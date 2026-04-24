from ..utils import *
from ..data_load.data_loader import *
from torch.utils.data import DataLoader

import os
from contextlib import nullcontext
from torch.profiler import profile, record_function, ProfilerActivity


class TrainBatchProcessor():
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.dataset = TrainDatasetMarginLoss(args, kg)
        self.shuffle_mode = True
        # num_workers>0 parallelizes data collation (~20ms/batch CPU in profile)
        # with the GPU forward. persistent_workers avoids respawning workers at
        # the start of every epoch. We guard persistent_workers behind nw>0
        # because PyTorch errors when it's True with nw=0.
        nw = int(getattr(self.args, "num_workers", 2))
        loader_kwargs = dict(
            shuffle=self.shuffle_mode,
            batch_size=int(self.args.batch_size),
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.random_seed)),
            pin_memory=True,
            num_workers=nw,
        )
        if nw > 0:
            loader_kwargs["persistent_workers"] = True
        self.data_loader = DataLoader(self.dataset, **loader_kwargs)

    def process_epoch(self, model, optimizer):
        model.train()
        if self.args.model_name == "LoraKGE_Layers" and (self.args.using_various_ranks or self.args.using_various_ranks_reverse):
            if model.lora_ent_embeddings_list != None:
                for lora_model in model.lora_ent_embeddings_list:
                    lora_model.train(True)
        """ Start training """
        total_loss = 0.0
        lora_stats = {
            "a_grad_norm_sum": 0.0,
            "b_grad_norm_sum": 0.0,
            "a_grad_steps": 0,
            "b_grad_steps": 0,
            "a_param_norm": 0.0,
            "b_param_norm": 0.0,
            "a_param_count": 0,
            "b_param_count": 0,
            "lr_min": min([group["lr"] for group in optimizer.param_groups], default=0.0),
            "lr_max": max([group["lr"] for group in optimizer.param_groups], default=0.0),
            "lr_group_count": len(optimizer.param_groups),
        }
        if self.args.record:
            loss_save_path = "/data/my_cl_kge/save/" + str(self.args.snapshot) + ".txt"
            with open(loss_save_path, "a", encoding="utf-8") as wf:
                wf.write(str(self.args.epoch))
                wf.write("\t")
        lora_a_params, lora_b_params = self._split_lora_params(model)

        # ---- P1 profiling setup ----------------------------------------------
        # Activates only when (snapshot, epoch) matches user-specified profile
        # target. Captures first profile_num_batches batches (first 2 = warmup)
        # then exports a chrome trace + logger table, and early-stops the loop
        # to keep trace size manageable.
        prof_ctx, profile_active, profile_num_batches = self._maybe_build_profiler()
        # ----------------------------------------------------------------------

        # log_lora_stats: old code called _accumulate_lora_grad_stats every
        # batch, which runs 16+ param.grad.norm(2).item() calls per batch --
        # each .item() is a CUDA sync that stalls the GPU pipeline (profile
        # showed ~25 aten::item/batch, dominating CPU-side overhead). New
        # contract: default = stats only after the last batch (still samples
        # real post-step grads); lora_stats_every > 0 keeps the old sampling
        # rate if a user actually needs fine-grained grad curves.
        log_stats = bool(getattr(self.args, "log_lora_stats", False))
        stats_every = int(getattr(self.args, "lora_stats_every", 0))
        last_batch_id = -1
        clip_val = float(getattr(self.args, "lora_grad_clip", 0.0) or 0.0)
        do_clip = clip_val > 0.0 and (lora_a_params or lora_b_params)

        with prof_ctx as prof:
            for b_id, batch in enumerate(self.data_loader):
                if profile_active and b_id >= profile_num_batches:
                    break
                last_batch_id = b_id
                with record_function("batch/total"):
                    """ Get loss """
                    bh, br, bt, by = batch
                    with record_function("batch/to_device"):
                        bh_d = bh.to(self.args.device, non_blocking=True)
                        br_d = br.to(self.args.device, non_blocking=True)
                        bt_d = bt.to(self.args.device, non_blocking=True)
                        by_d = by.to(self.args.device, non_blocking=True) if by is not None else by
                    optimizer.zero_grad()
                    with record_function("batch/forward_loss"):
                        batch_loss = model.loss(bh_d, br_d, bt_d, by_d)
                    """ updata """
                    with record_function("batch/backward"):
                        batch_loss.backward()
                    if log_stats and stats_every > 0 and (b_id % stats_every == 0):
                        self._accumulate_lora_grad_stats(lora_a_params, lora_b_params, lora_stats)
                    if do_clip:
                        torch.nn.utils.clip_grad_norm_(
                            lora_a_params + lora_b_params, max_norm=clip_val
                        )
                    with record_function("batch/optimizer_step"):
                        optimizer.step()
                    total_loss += batch_loss.item()
                    """ post processing """
                    model.epoch_post_processing(bh.size(0))
                    if self.args.record:
                        with open(loss_save_path, "a", encoding="utf-8") as wf:
                            wf.write(str(batch_loss.item()))
                            wf.write("\t")
                if profile_active and prof is not None:
                    prof.step()

        if profile_active and prof is not None:
            self._dump_profile_results(prof)

        if self.args.record:
            with open(loss_save_path, "a", encoding="utf-8") as wf:
                wf.write("\n")
        if log_stats:
            # Default low-overhead mode: sample grads once per epoch using
            # gradients from the last batch (still in .grad -- zero_grad is
            # called at the start of the next batch, not after step()).
            if stats_every <= 0 and last_batch_id >= 0:
                self._accumulate_lora_grad_stats(lora_a_params, lora_b_params, lora_stats)
            self._finalize_lora_param_stats(lora_a_params, lora_b_params, lora_stats)
        return total_loss, lora_stats

    # ------------------------------------------------------------------
    # P1 profiling helpers
    # ------------------------------------------------------------------
    def _maybe_build_profiler(self):
        """Return (context_manager, active_flag, profile_num_batches).

        When inactive, context_manager is a nullcontext() so the batch loop is
        zero-overhead (record_function calls become near no-ops without an
        attached profiler).
        """
        target_snap = int(getattr(self.args, "profile_snapshot", -1))
        target_epoch = int(getattr(self.args, "profile_epoch", -1))
        cur_snap = int(getattr(self.args, "snapshot", -999))
        cur_epoch = int(getattr(self.args, "epoch", -999))
        active = (target_snap >= 0 and target_snap == cur_snap and target_epoch == cur_epoch)
        if not active:
            return nullcontext(), False, 0

        num_batches = max(1, int(getattr(self.args, "profile_num_batches", 20)))
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        # No `schedule` on purpose: with a schedule, events from the active
        # phase aren't guaranteed to be flushed into key_averages() unless the
        # cycle completes AND we call prof.step() one more time (or on_trace_
        # ready fires). Previous run showed an empty top-ops table because of
        # this. Recording the whole captured window is fine here since we cap
        # at num_batches anyway; the trace is small.
        # V2.11: record_shapes=True so the follow-up analysis can discriminate
        # addmm/index_select calls by tensor shape (worth ~2% overhead, useful).
        prof = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
        logger = getattr(self.args, "logger", None)
        if logger is not None:
            logger.info(
                f"[profile] ENABLED snapshot={cur_snap} epoch={cur_epoch} "
                f"batches={num_batches} "
                f"activities={[a.name for a in activities]}"
            )
        return prof, True, num_batches

    def _dump_profile_results(self, prof):
        """Print full CPU/CUDA breakdowns and export a chrome trace.

        V2.11 upgrade (over V2.7 profiler output):
          - Always emit BOTH a top-CPU table and a top-CUDA table, regardless
            of which backend is hot. Previous version only printed one combined
            table sorted by CUDA time, but PyTorch silently drops CUDA columns
            when events don't satisfy some internal threshold -> we were left
            staring at a CPU-only view and thought GPU data was lost.
          - Add a one-line "budget" summary (wall / self-CPU / self-CUDA / #
            kernels) so the CPU-vs-GPU-bound question is answered at a glance.
          - record_shapes=True upstream means the kernel tables are now
            shape-aware: same aten::addmm at (B,D) vs (B,E) shows as two rows.
        """
        out_dir = str(getattr(self.args, "profile_out", "./logs/profile"))
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            pass
        ds = str(getattr(self.args, "dataset", "UNKNOWN"))
        snap = int(getattr(self.args, "snapshot", -1))
        epoch = int(getattr(self.args, "epoch", -1))
        trace_path = os.path.join(out_dir, f"{ds}_s{snap}_e{epoch}.json")

        logger = getattr(self.args, "logger", None)
        _log = (lambda msg: logger.info(msg)) if logger is not None else print

        try:
            events = prof.key_averages()
        except Exception as e:  # noqa: BLE001
            _log(f"[profile] key_averages() failed: {e}")
            return

        total_self_cpu_us = sum(getattr(ev, "self_cpu_time_total", 0) for ev in events)
        total_self_cuda_us = sum(getattr(ev, "self_cuda_time_total", 0) for ev in events)
        cuda_event_count = sum(
            1 for ev in events if getattr(ev, "self_cuda_time_total", 0) > 0
        )

        _log(f"[profile] dataset={ds} snapshot={snap} epoch={epoch} trace={trace_path}")
        _log(
            "[profile] budget: self_cpu={:.1f}ms  self_cuda={:.1f}ms  "
            "cuda_events={}  (ratio CUDA/CPU={:.2f})".format(
                total_self_cpu_us / 1e3,
                total_self_cuda_us / 1e3,
                cuda_event_count,
                (total_self_cuda_us / total_self_cpu_us) if total_self_cpu_us > 0 else 0.0,
            )
        )
        if cuda_event_count == 0 and torch.cuda.is_available():
            _log(
                "[profile] WARN: no CUDA events captured. Training is either "
                "CPU-bound (common for small KGE models), or the profiler "
                "missed the launch window (try -profile_num_batches 40)."
            )

        def _safe_table(sort_by):
            try:
                return events.table(sort_by=sort_by, row_limit=25)
            except Exception as e:  # noqa: BLE001
                return f"(key_averages().table failed on sort_by={sort_by}: {e})"

        _log("[profile] top ops by self_cpu_time_total:\n" + _safe_table("self_cpu_time_total"))
        if cuda_event_count > 0:
            _log(
                "[profile] top ops by self_cuda_time_total:\n"
                + _safe_table("self_cuda_time_total")
            )

        try:
            prof.export_chrome_trace(trace_path)
            _log(f"[profile] chrome trace saved -> {trace_path}")
        except Exception as e:  # noqa: BLE001
            _log(f"[profile] export_chrome_trace failed: {e}")

    def _split_lora_params(self, model):
        lora_a_params = []
        lora_b_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" not in name:
                continue
            if "lora_B" in name or "lora_embedding_B" in name:
                lora_b_params.append(param)
            else:
                lora_a_params.append(param)
        return lora_a_params, lora_b_params

    def _accumulate_lora_grad_stats(self, lora_a_params, lora_b_params, lora_stats):
        a_grad_sq = 0.0
        b_grad_sq = 0.0
        has_a_grad = False
        has_b_grad = False

        for param in lora_a_params:
            if param.grad is None:
                continue
            grad_norm = param.grad.data.norm(2).item()
            a_grad_sq += grad_norm * grad_norm
            has_a_grad = True

        for param in lora_b_params:
            if param.grad is None:
                continue
            grad_norm = param.grad.data.norm(2).item()
            b_grad_sq += grad_norm * grad_norm
            has_b_grad = True

        if has_a_grad:
            lora_stats["a_grad_norm_sum"] += a_grad_sq ** 0.5
            lora_stats["a_grad_steps"] += 1
        if has_b_grad:
            lora_stats["b_grad_norm_sum"] += b_grad_sq ** 0.5
            lora_stats["b_grad_steps"] += 1

    def _finalize_lora_param_stats(self, lora_a_params, lora_b_params, lora_stats):
        a_param_sq = 0.0
        b_param_sq = 0.0

        for param in lora_a_params:
            param_norm = param.data.norm(2).item()
            a_param_sq += param_norm * param_norm
        for param in lora_b_params:
            param_norm = param.data.norm(2).item()
            b_param_sq += param_norm * param_norm

        lora_stats["a_param_norm"] = a_param_sq ** 0.5
        lora_stats["b_param_norm"] = b_param_sq ** 0.5
        lora_stats["a_param_count"] = len(lora_a_params)
        lora_stats["b_param_count"] = len(lora_b_params)

class DevBatchProcessor():
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = 100
        """ prepare data """
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.random_seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        if self.args.model_name == "LoraKGE_Layers" and self.args.using_various_ranks or self.args.using_various_ranks_reverse:
            if model.lora_ent_embeddings_list != None:
                for lora_model in model.lora_ent_embeddings_list:
                    lora_model.train(False)
        num = 0
        results = {}
        hr2t = self.kg.snapshots[self.args.snapshot].hr2t_all
        """ Start evaluation """
        with torch.no_grad():
            for batch in self.data_loader:
            # head: (batch_size, 1)
                head, relation, tail, label = batch
                head = head.to(self.args.device)
                relation = relation.to(self.args.device)
                tail = tail.to(self.args.device)
                label = label.to(self.args.device) # (batch_size, ent_num)
                num += len(head)
                stage = "Valid" if self.args.valid else "Test"
                """ Get prediction scores """
                pred = model.predict(head, relation, stage=stage) # (batch_size, num_ent)
                """ filter: """
                batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
                target_pred = pred[batch_size_range, tail]
                pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[batch_size_range, tail] = target_pred
                if self.args.predict_result and stage == "Test":
                    logits_sorted, indices_sorted = torch.sort(pred, dim=-1, descending=True)
                    predict_result_path = "/data2/jun/lora_clkge/save/predict_result/" + "lora_kge/" + str(self.args.snapshot) + "_" + str(self.args.snapshot_test) + ".txt"
                    with open(predict_result_path, "a", encoding="utf-8") as af:
                        batch_num = len(head)
                        for i in range(batch_num):
                            top1 = indices_sorted[i][0]
                            top2 = indices_sorted[i][1]
                            top3 = indices_sorted[i][2]
                            af.write(self.kg.id2entity[head[i].detach().cpu().item()])
                            af.write("\t")
                            af.write(self.kg.id2relation[relation[i].detach().cpu().item()])
                            af.write("\t")
                            af.write(self.kg.id2entity[tail[i].detach().cpu().item()])
                            af.write("\n")
                            af.write(self.kg.id2entity[top1.detach().cpu().item()])
                            af.write("\t")
                            af.write(self.kg.id2entity[top2.detach().cpu().item()])
                            af.write("\t")
                            af.write(self.kg.id2entity[top3.detach().cpu().item()])
                            af.write("\n")
                            af.write("----------------------------------------------------------")
                            af.write("\n")
                """ rank all candidate entities """
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[batch_size_range, tail]
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results[f'hits{k + 1}'] = torch.numel(
                        ranks[ranks <= (k + 1)]
                    ) + results.get(f'hits{k + 1}', 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results