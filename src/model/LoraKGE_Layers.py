import math
import os
import re
from collections import defaultdict, deque
from typing import cast

import torch.nn.functional as F
from torch.profiler import record_function

from .BaseModel import *


def _init_nonempty(fn, tensor, *args, **kwargs):
    """Call a torch.nn.init-style initializer only when the tensor has elements.

    Avoids the `Initializing zero-element tensors is a no-op` UserWarning that
    triggers whenever a snapshot adds new entities but no new relations (or
    vice versa), leaving a (0, emb_dim) / (r, 0) shaped parameter.
    """
    if tensor is None or tensor.numel() == 0:
        return tensor
    return fn(tensor, *args, **kwargs)


class DoRALoraEmbedding(loralib.Embedding):
    """
    DoRA-inspired adapter: keep frozen base embedding + standard LoRA (B @ A),
    multiply the low-rank delta by a learnable per-dimension scale lora_g (shape emb_dim).
    Named lora_g so loralib.lora_state_dict() keeps it in checkpoints.
    merge_weights=False: merged weight path cannot represent g-scaled deltas.
    Optional use_rs_lora: replace loralib's alpha/r scaling with alpha/sqrt(r) (RS-LoRA), DoRA path only.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = False,
        use_rs_lora: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            r,
            lora_alpha=lora_alpha,
            merge_weights=merge_weights,
            **kwargs,
        )
        if r > 0 and use_rs_lora:
            self.scaling = self.lora_alpha / math.sqrt(float(self.r))
        if r > 0:
            self.lora_g = nn.Parameter(
                torch.ones(embedding_dim, dtype=self.weight.dtype, device=self.weight.device)
            )
        else:
            self.register_parameter("lora_g", None)

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x,
                self.lora_A.transpose(0, 1),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            delta = (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            result = result + delta * self.lora_g
            return result
        return nn.Embedding.forward(self, x)


class LoraKGE_Layers(BaseModel):
    def __init__(self, args, kg) -> None:
        super(LoraKGE_Layers, self).__init__(args, kg)
        self.lora_ent_embeddings_list = None
        # P0 Graph Layering (rel-side): always a ModuleList. When num_rel_layers==1
        # the list has a single module and the pipeline is numerically identical
        # to the pre-P0 single-module path.
        self.lora_rel_embeddings_list = None
        self.lora_rel_len = 0
        self.new_rel_embeddings_len = 0
        self.ent_layer_difficulty_scores = None
        self.ent_layer_lr_scales = None
        self._new_entity_stats = None
        self.interlayer_ent_scales = None
        self.interlayer_ent_projs = None
        # P1/P2 efficiency caches: precomputed once per snapshot transition and
        # reused across every batch of that snapshot. Without these, margin_loss
        # / predict / get_lora_embeddings were rebuilding the same tensors per
        # batch (torch.arange(N).to(device), list(range(old_num_ent)) +
        # new_ordered_entities, python-list fancy index).
        self._ent_arange_cache = None     # torch.arange(lora_ent_len) on device
        self._rel_arange_list = None      # list of torch.arange(rows) per rel layer
        self._ent_canonical_perm = None   # maps canonical-new-ent-idx -> lora row

    def store_old_parameters(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def expand_embedding_size(self):
        _dtype = model_dtype(self.args)
        ent_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_ent, self.args.emb_dim).to(self.args.device, dtype=_dtype)
        rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_rel, self.args.emb_dim).to(self.args.device, dtype=_dtype)
        xavier_normal_(ent_embeddings.weight)
        xavier_normal_(rel_embeddings.weight)
        return ent_embeddings, rel_embeddings

    def _get_lora_factor_params(self, module: nn.Module):
        lora_a_param = getattr(module, "lora_embedding_A", None)
        lora_b_param = getattr(module, "lora_embedding_B", None)
        if lora_a_param is None or lora_b_param is None:
            lora_a_param = getattr(module, "lora_A", None)
            lora_b_param = getattr(module, "lora_B", None)
        return lora_a_param, lora_b_param

    def _write_low_rank_matrix(self, module: nn.Module, target_matrix: torch.Tensor, base_matrix: torch.Tensor | None = None):
        with torch.no_grad():
            module_weight = cast(torch.Tensor, getattr(module, "weight"))
            if base_matrix is None:
                module_weight.zero_()
            else:
                module_weight.copy_(base_matrix.to(device=module_weight.device, dtype=module_weight.dtype))

            lora_a_param, lora_b_param = self._get_lora_factor_params(module)
            if lora_a_param is None or lora_b_param is None:
                return

            lora_a = cast(torch.Tensor, lora_a_param.data)
            lora_b = cast(torch.Tensor, lora_b_param.data)
            lora_a.zero_()
            lora_b.zero_()

            if target_matrix.numel() == 0:
                return

            target_matrix = target_matrix.to(device=module_weight.device, dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(target_matrix, full_matrices=False)
            module_rank = int(cast(int, getattr(module, "r", 0)))
            effective_r = min(module_rank, Vh.size(0), U.size(1))
            if effective_r <= 0:
                return

            us = U[:, :effective_r] @ torch.diag(S[:effective_r])
            vh = Vh[:effective_r, :]
            if lora_a.shape[1] == target_matrix.shape[1] and lora_b.shape[0] == target_matrix.shape[0]:
                lora_a[:effective_r, :] = vh.to(device=lora_a.device, dtype=lora_a.dtype)
                lora_b[:, :effective_r] = us.to(device=lora_b.device, dtype=lora_b.dtype)
            elif lora_a.shape[1] == target_matrix.shape[0] and lora_b.shape[0] == target_matrix.shape[1]:
                lora_a[:effective_r, :] = us.T.to(device=lora_a.device, dtype=lora_a.dtype)
                lora_b[:, :effective_r] = vh.T.to(device=lora_b.device, dtype=lora_b.dtype)
            else:
                _init_nonempty(xavier_normal_, module_weight)

    def _build_reference_block(self, source_weight: torch.Tensor, rows: int, layer_idx: int, total_layers: int) -> torch.Tensor:
        if rows <= 0:
            return torch.zeros((0, self.args.emb_dim), device=self.args.device, dtype=torch.float32)
        if source_weight.numel() == 0 or source_weight.shape[0] == 0:
            return torch.zeros((rows, self.args.emb_dim), device=self.args.device, dtype=torch.float32)

        source_weight = source_weight.detach().to(device=self.args.device, dtype=torch.float32)
        source_rows = int(source_weight.shape[0])
        sample_pos = torch.linspace(0, max(source_rows - 1, 0), steps=rows, device=source_weight.device)
        sample_idx = sample_pos.round().long()
        if total_layers > 1 and source_rows > 1:
            stride = max(1, source_rows // total_layers)
            sample_idx = (sample_idx + layer_idx * stride) % source_rows
        return source_weight.index_select(0, sample_idx)

    def _quantize_reference_block(self, reference_block: torch.Tensor) -> torch.Tensor:
        bits = max(2, int(getattr(self.args, "quant_init_bits", 8)))
        quant_levels = (1 << (bits - 1)) - 1
        if quant_levels <= 0 or reference_block.numel() == 0:
            return reference_block.clone()

        granularity = str(getattr(self.args, "quant_init_granularity", "row")).lower()
        if granularity == "tensor":
            scale = reference_block.abs().max().view(1, 1)
        else:
            scale = reference_block.abs().amax(dim=1, keepdim=True)
        scale = scale.clamp_min(1e-8) / float(quant_levels)
        quantized = torch.round(reference_block / scale).clamp(-quant_levels, quant_levels)
        return quantized * scale

    def _init_quant_svd_module(
        self,
        module: nn.Module,
        source_weight: torch.Tensor,
        rows: int,
        layer_idx: int,
        total_layers: int,
    ):
        reference_block = self._build_reference_block(source_weight, rows, layer_idx, total_layers)
        quantized_block = self._quantize_reference_block(reference_block)
        residual_block = reference_block - quantized_block
        self._write_low_rank_matrix(module, residual_block, base_matrix=quantized_block)

    def _build_legacy_entity_stats(self):
        old_num_ent = int(self.kg.snapshots[self.args.snapshot].num_ent)
        next_num_ent = int(self.kg.snapshots[self.args.snapshot + 1].num_ent)
        new_entities = list(range(old_num_ent, next_num_ent))
        stats = {
            eid: {
                "distance": 100.0,
                "degree": 0.0,
                "old_support": 0.0,
                "new_relation": 0.0,
                "difficulty": 0.0,
            }
            for eid in new_entities
        }
        nodes_ordered_path = f"./data/{self.args.dataset}/{self.args.snapshot + 1}/train_distance_nodes.txt"
        if os.path.exists(nodes_ordered_path):
            with open(nodes_ordered_path, "r", encoding="utf-8") as f:
                lines = list(f.readlines())
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    line_list = line.split("\t")
                    if len(line_list) < 3:
                        continue
                    node, distance, score = int(line_list[0]), float(line_list[1]), float(line_list[2])
                    if node in stats:
                        stats[node]["distance"] = distance
                        stats[node]["degree"] = score
                        stats[node]["difficulty"] = distance + max(0.0, 1.0 / (score + 1.0))
            return stats
        return self._build_difficulty_entity_stats()

    def _build_difficulty_entity_stats(self):
        old_num_ent = int(self.kg.snapshots[self.args.snapshot].num_ent)
        old_num_rel = int(self.kg.snapshots[self.args.snapshot].num_rel)
        next_snapshot = self.kg.snapshots[self.args.snapshot + 1]
        next_num_ent = int(next_snapshot.num_ent)
        new_entities = list(range(old_num_ent, next_num_ent))
        new_entity_set = set(new_entities)

        stats = {
            eid: {
                "distance": 100.0,
                "degree": 0.0,
                "old_support": 0.0,
                "new_relation": 0.0,
                "difficulty": 0.0,
            }
            for eid in new_entities
        }
        adjacency = defaultdict(set)
        old_neighbor_sets = {eid: set() for eid in new_entities}
        relation_sets = {eid: set() for eid in new_entities}

        for h, r, t in next_snapshot.train:
            adjacency[h].add(t)
            adjacency[t].add(h)
            if h in new_entity_set:
                stats[h]["degree"] += 1.0
                if t < old_num_ent:
                    old_neighbor_sets[h].add(t)
                if r >= old_num_rel:
                    relation_sets[h].add(r)
            if t in new_entity_set:
                stats[t]["degree"] += 1.0
                if h < old_num_ent:
                    old_neighbor_sets[t].add(h)
                if r >= old_num_rel:
                    relation_sets[t].add(r)

        seed_nodes = [eid for eid in range(old_num_ent) if eid in adjacency]
        visited = set(seed_nodes)
        bfs_queue = deque((eid, 0) for eid in seed_nodes)
        while bfs_queue:
            node, dist = bfs_queue.popleft()
            for nxt in adjacency[node]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                next_dist = dist + 1
                if nxt in new_entity_set:
                    stats[nxt]["distance"] = float(next_dist)
                bfs_queue.append((nxt, next_dist))

        max_distance = max([value["distance"] for value in stats.values()] + [1.0])
        max_degree = max([value["degree"] for value in stats.values()] + [1.0])
        max_old_support = max([float(len(old_neighbor_sets[eid])) for eid in new_entities] + [1.0])
        max_new_rel = max([float(len(relation_sets[eid])) for eid in new_entities] + [1.0])

        for eid in new_entities:
            stats[eid]["old_support"] = float(len(old_neighbor_sets[eid]))
            stats[eid]["new_relation"] = float(len(relation_sets[eid]))
            dist_norm = stats[eid]["distance"] / max_distance
            degree_norm = stats[eid]["degree"] / max_degree
            old_support_norm = stats[eid]["old_support"] / max_old_support
            new_rel_norm = stats[eid]["new_relation"] / max_new_rel
            stats[eid]["difficulty"] = (
                dist_norm
                + 0.5 * (1.0 - degree_norm)
                + 0.75 * (1.0 - old_support_norm)
                + 0.35 * new_rel_norm
            )
        return stats

    def _compute_new_entity_stats(self):
        strategy = str(getattr(self.args, "entity_layering", "legacy")).lower()
        if strategy == "difficulty":
            return self._build_difficulty_entity_stats()
        return self._build_legacy_entity_stats()

    def _build_rank_list_from_difficulty(self, layer_scores):
        num_layers = max(1, int(self.args.num_ent_layers))
        base_rank = max(1, int(self.args.ent_r))
        if num_layers == 1:
            return [base_rank]
        base_per_layer = max(1, base_rank // num_layers)
        mean_score = sum(layer_scores) / max(1, len(layer_scores))
        scale = float(getattr(self.args, "difficulty_rank_scale", 0.5))
        ranks = []
        for score in layer_scores:
            centered = 0.0 if mean_score <= 1e-8 else (score - mean_score) / mean_score
            rank = int(round(base_per_layer * (1.0 + scale * centered)))
            ranks.append(max(1, rank))
        return ranks

    def _build_lr_scales_from_difficulty(self, layer_scores):
        mean_score = sum(layer_scores) / max(1, len(layer_scores))
        scale = float(getattr(self.args, "difficulty_lr_scale", 0.0))
        if scale <= 0.0:
            return [1.0] * len(layer_scores)
        lr_scales = []
        for score in layer_scores:
            centered = 0.0 if mean_score <= 1e-8 else (score - mean_score) / mean_score
            lr_scale = 1.0 + scale * centered
            lr_scales.append(max(0.25, float(lr_scale)))
        return lr_scales

    def _log_entity_layer_plan(self):
        logger = getattr(self.args, "logger", None)
        if logger is None:
            return
        if not bool(getattr(self.args, "log_layer_plan", False)):
            return
        logger.info(
            "分层规划 snapshot=%s strategy=%s ranks=%s lr_scales=%s difficulty=%s",
            self.args.snapshot + 1,
            getattr(self.args, "entity_layering", "legacy"),
            getattr(self.args, "ent_r_list", None),
            self.ent_layer_lr_scales,
            self.ent_layer_difficulty_scores,
        )

    def _rebuild_interlayer_modules(self):
        mode = str(getattr(self.args, "interlayer_lora_mode", "off")).lower()
        num_layers = max(1, int(self.args.num_ent_layers))
        if mode == "off" or num_layers <= 1:
            self.interlayer_ent_scales = None
            self.interlayer_ent_projs = None
            return
        init_value = float(getattr(self.args, "interlayer_init", 0.0))
        dtype = model_dtype(self.args)
        self.interlayer_ent_scales = nn.Parameter(
            torch.full((num_layers - 1,), init_value, device=self.args.device, dtype=dtype)
        )
        if mode == "gate":
            proj_list = []
            for _ in range(num_layers - 1):
                proj = nn.Linear(self.args.emb_dim, self.args.emb_dim, bias=False).to(self.args.device, dtype=dtype)
                nn.init.eye_(proj.weight)
                proj_list.append(proj)
            self.interlayer_ent_projs = nn.ModuleList(proj_list)
        else:
            self.interlayer_ent_projs = None

    def _fuse_with_previous_entity_layer(self, layer_idx: int, current_part: torch.Tensor, previous_part: torch.Tensor | None):
        mode = str(getattr(self.args, "interlayer_lora_mode", "off")).lower()
        if mode == "off" or layer_idx == 0 or previous_part is None or self.interlayer_ent_scales is None:
            return current_part
        previous_summary = previous_part.mean(dim=0, keepdim=True)
        if str(getattr(self.args, "interlayer_stopgrad", "on")).lower() == "on":
            previous_summary = previous_summary.detach()
        if mode == "residual":
            return current_part + self.interlayer_ent_scales[layer_idx - 1] * previous_summary
        if mode == "gate" and self.interlayer_ent_projs is not None:
            previous_summary = self.interlayer_ent_projs[layer_idx - 1](previous_summary)
            gate_value = torch.sigmoid(self.interlayer_ent_scales[layer_idx - 1])
            return current_part + gate_value * previous_summary
        return current_part

    def get_new_ordered_entities(self):
        all_new_entities = self._compute_new_entity_stats()
        self._new_entity_stats = all_new_entities
        strategy = str(getattr(self.args, "entity_layering", "legacy")).lower()
        if strategy == "difficulty":
            ordered = sorted(
                all_new_entities.items(),
                key=lambda kv: (kv[1]["difficulty"], kv[1]["distance"], -kv[1]["old_support"], kv[0]),
            )
            self.all_new_entities = {
                key: (value["distance"], value["difficulty"])
                for key, value in ordered
            }
        else:
            ordered = sorted(
                all_new_entities.items(),
                key=lambda kv: (kv[1]["distance"], kv[1]["degree"], kv[0]),
            )
            self.all_new_entities = {
                key: (value["distance"], value["degree"])
                for key, value in ordered
            }
        return list(self.all_new_entities.keys())

    def expand_lora_embeddings(self):
        self.new_ordered_entities = self.get_new_ordered_entities()
        new_ent_embeddings_len = self.kg.snapshots[self.args.snapshot + 1].num_ent - self.kg.snapshots[self.args.snapshot].num_ent
        new_rel_embeddings_len = self.kg.snapshots[self.args.snapshot + 1].num_rel - self.kg.snapshots[self.args.snapshot].num_rel
        self.lora_ent_len = (new_ent_embeddings_len + int(self.args.num_ent_layers) - 1) // int(self.args.num_ent_layers)
        self.ent_layer_difficulty_scores = []
        for i_layer in range(int(self.args.num_ent_layers)):
            start = i_layer * self.lora_ent_len
            end = min((i_layer + 1) * self.lora_ent_len, len(self.new_ordered_entities))
            layer_entities = self.new_ordered_entities[start:end]
            if len(layer_entities) == 0:
                self.ent_layer_difficulty_scores.append(0.0)
                continue
            assert self._new_entity_stats is not None
            layer_score = sum(self._new_entity_stats[eid]["difficulty"] for eid in layer_entities) / len(layer_entities)
            self.ent_layer_difficulty_scores.append(float(layer_score))
        
        tmp_r = self.args.ent_r
        self.args.ent_r = (self.lora_ent_len // 20) if (self.lora_ent_len // 20) > int(self.args.ent_r) else self.args.ent_r
        if self.args.explore:
            self.args.ent_r = tmp_r
        rank_policy = str(getattr(self.args, "ent_rank_policy", "legacy")).lower()
        if rank_policy == "difficulty":
            self.args.ent_r_list = self._build_rank_list_from_difficulty(self.ent_layer_difficulty_scores)
        elif rank_policy == "uniform":
            self.args.ent_r_list = [int(self.args.ent_r) // int(self.args.num_ent_layers)] * int(self.args.num_ent_layers)
            self.args.ent_r_list = [i if i else 1 for i in self.args.ent_r_list]
        elif self.args.using_various_ranks:
            ent_node_list = []
            for _, v in self.all_new_entities.items():
                ent_node_list.append(v[1])
            self.args.ent_r_list = []
            for i_layer in range(int(self.args.num_ent_layers)):
                self.args.ent_r_list.append(sum(ent_node_list[i_layer * self.lora_ent_len: (i_layer + 1) * self.lora_ent_len]))
            average_nodes = sum(self.args.ent_r_list) / len(self.args.ent_r_list)
            r_threshold = int(int(self.args.ent_r) * 0.9)
            self.args.ent_r_list = [int(self.args.ent_r) * i / average_nodes if int(self.args.ent_r) * i / average_nodes > r_threshold else r_threshold for i in self.args.ent_r_list]
            self.args.ent_r_list = [int(i) for i in self.args.ent_r_list]
            self.args.ent_r_list = [int(i) for i in self.args.ent_r_list]
            self.args.ent_r_list = [i if i else 1 for i in self.args.ent_r_list]
            assert len(self.args.ent_r_list) == int(self.args.num_ent_layers)
        elif self.args.using_various_ranks_reverse:
            self.args.ent_r_list = np.linspace(int(self.args.ent_r) // 2, int(self.args.ent_r) // 2 * 3, int(self.args.num_ent_layers)).tolist()
            self.args.ent_r_list = [int(i) for i in self.args.ent_r_list]
            self.args.ent_r_list = [i if i else 1 for i in self.args.ent_r_list]
            self.args.ent_r_list = self.args.ent_r_list[::-1]
            assert len(self.args.ent_r_list) == int(self.args.num_ent_layers)
        else:
            self.args.ent_r_list = [int(self.args.ent_r) // int(self.args.num_ent_layers)] * int(self.args.num_ent_layers)
            self.args.ent_r_list = [i if i else 1 for i in self.args.ent_r_list]
        self.ent_layer_lr_scales = self._build_lr_scales_from_difficulty(self.ent_layer_difficulty_scores)
        self._log_entity_layer_plan()

        lora_ent_embeddings_list = []
        reference_weight = self.ent_embeddings.weight.data.float()
        ref_mean = reference_weight.mean()
        ref_std = reference_weight.std().clamp_min(1e-6)
        init_method = str(getattr(self.args, "lora_init", "pissa")).lower()
        ent_source_weight = self.ent_embeddings.weight.data[: self.kg.snapshots[self.args.snapshot].num_ent].float()

        for layer_idx in range(int(self.args.num_ent_layers)):
            r = int(self.args.ent_r_list[layer_idx])
            if init_method == "dora":
                new_ent_embeddings = DoRALoraEmbedding(
                    self.lora_ent_len,
                    self.args.emb_dim,
                    r,
                    merge_weights=False,
                    use_rs_lora=bool(getattr(self.args, "use_rs_lora", False)),
                ).to(self.args.device, dtype=model_dtype(self.args))
                _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
            else:
                new_ent_embeddings = loralib.Embedding(self.lora_ent_len, self.args.emb_dim, r).to(self.args.device, dtype=model_dtype(self.args))
                if init_method == "xavier":
                    _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
                elif init_method == "zero_b":
                    # Canonical LoRA init: A ~ kaiming, B = 0 → delta = 0 at step 0,
                    # and grad_B ∝ A is immediately non-zero so LoRA+ (lr_B > lr_A) is meaningful.
                    _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
                    with torch.no_grad():
                        lora_a_param = getattr(new_ent_embeddings, "lora_embedding_A", None)
                        lora_b_param = getattr(new_ent_embeddings, "lora_embedding_B", None)
                        if lora_a_param is None or lora_b_param is None:
                            lora_a_param = getattr(new_ent_embeddings, "lora_A", None)
                            lora_b_param = getattr(new_ent_embeddings, "lora_B", None)
                        if lora_a_param is not None and lora_b_param is not None:
                            _init_nonempty(
                                nn.init.kaiming_uniform_,
                                cast(torch.Tensor, lora_a_param.data),
                                a=math.sqrt(5),
                            )
                            _init_nonempty(nn.init.zeros_, cast(torch.Tensor, lora_b_param.data))
                elif init_method == "quant_svd":
                    self._init_quant_svd_module(
                        new_ent_embeddings,
                        ent_source_weight,
                        self.lora_ent_len,
                        layer_idx,
                        int(self.args.num_ent_layers),
                    )
                elif init_method == "qr":
                    with torch.no_grad():
                        effective_r = min(r, self.lora_ent_len, self.args.emb_dim)
                        left_rand = torch.randn(self.lora_ent_len, effective_r, device=self.args.device, dtype=torch.float32)
                        right_rand = torch.randn(self.args.emb_dim, effective_r, device=self.args.device, dtype=torch.float32)
                        left_q, _ = torch.linalg.qr(left_rand, mode='reduced')
                        right_q, _ = torch.linalg.qr(right_rand, mode='reduced')
                        us = left_q[:, :effective_r] * ref_std
                        vh = right_q[:, :effective_r].T
                        lora_a_param = getattr(new_ent_embeddings, "lora_embedding_A", None)
                        lora_b_param = getattr(new_ent_embeddings, "lora_embedding_B", None)
                        if lora_a_param is None or lora_b_param is None:
                            lora_a_param = getattr(new_ent_embeddings, "lora_A", None)
                            lora_b_param = getattr(new_ent_embeddings, "lora_B", None)
                        if lora_a_param is not None and lora_b_param is not None:
                            lora_a = cast(torch.Tensor, lora_a_param.data)
                            lora_b = cast(torch.Tensor, lora_b_param.data)
                            if lora_a.shape[1] == self.args.emb_dim and lora_b.shape[0] == self.lora_ent_len:
                                lora_a[:effective_r, :] = vh.clone()
                                lora_b[:, :effective_r] = us.clone()
                            elif lora_a.shape[1] == self.lora_ent_len and lora_b.shape[0] == self.args.emb_dim:
                                lora_a[:effective_r, :] = us.T.clone()
                                lora_b[:, :effective_r] = vh.T.clone()
                            else:
                                _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
                            new_ent_embeddings.weight.data.zero_()
                        else:
                            _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
                else:
                    with torch.no_grad():
                        temp_w = torch.randn(self.lora_ent_len, self.args.emb_dim, device=self.args.device, dtype=torch.float32) * ref_std + ref_mean
                        U, S, Vh = torch.linalg.svd(temp_w, full_matrices=False)
                        effective_r = min(r, Vh.size(0), U.size(1))
                        lora_a_param = getattr(new_ent_embeddings, "lora_embedding_A", None)
                        lora_b_param = getattr(new_ent_embeddings, "lora_embedding_B", None)
                        if lora_a_param is None or lora_b_param is None:
                            lora_a_param = getattr(new_ent_embeddings, "lora_A", None)
                            lora_b_param = getattr(new_ent_embeddings, "lora_B", None)
                        if lora_a_param is not None and lora_b_param is not None:
                            lora_a = cast(torch.Tensor, lora_a_param.data)
                            lora_b = cast(torch.Tensor, lora_b_param.data)
                            us = U[:, :effective_r] @ torch.diag(S[:effective_r])
                            vh = Vh[:effective_r, :]
                            if lora_a.shape[1] == self.args.emb_dim and lora_b.shape[0] == self.lora_ent_len:
                                lora_a[:effective_r, :] = vh.clone()
                                lora_b[:, :effective_r] = us.clone()
                            elif lora_a.shape[1] == self.lora_ent_len and lora_b.shape[0] == self.args.emb_dim:
                                lora_a[:effective_r, :] = us.T.clone()
                                lora_b[:, :effective_r] = vh.T.clone()
                            else:
                                _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
                            new_ent_embeddings.weight.data.zero_()
                        else:
                            _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
            lora_ent_embeddings_list.append(new_ent_embeddings)

        # ---- Relation side: P0 Graph Layering --------------------------------
        # Split new relations into `num_rel_layers` chunks in canonical insertion
        # order; each chunk owns its own LoRA adapter. When num_rel_layers == 1
        # this loop produces a single module of shape (new_rel_len, emb_dim, rel_r)
        # which is bit-identical to the pre-P0 path (same rows, same rank, same
        # RNG sequence for init).
        num_rel_layers = max(1, int(self.args.num_rel_layers))
        self.lora_rel_len = (new_rel_embeddings_len + num_rel_layers - 1) // num_rel_layers
        self.new_rel_embeddings_len = new_rel_embeddings_len
        # Per-layer rank. Uniform split mirrors the ent-side default path when
        # using_various_ranks is off; each layer gets rel_r // num_rel_layers rank.
        # For num_rel_layers==1 this is just [rel_r], matching current behavior.
        base_rel_r = int(self.args.rel_r)
        if num_rel_layers == 1:
            rel_r_list = [base_rel_r]
        else:
            rel_r_list = [max(1, base_rel_r // num_rel_layers)] * num_rel_layers
        self.args.rel_r_list = rel_r_list

        lora_rel_embeddings_list = []
        rel_source_weight = self.rel_embeddings.weight.data[: self.kg.snapshots[self.args.snapshot].num_rel].float()
        for layer_i in range(num_rel_layers):
            layer_rows = self.lora_rel_len if num_rel_layers > 1 else new_rel_embeddings_len
            layer_r = int(rel_r_list[layer_i])
            lora_rel_embeddings_list.append(
                self._build_rel_lora_module(layer_rows, layer_r, init_method, ref_mean, ref_std, rel_source_weight, layer_i, num_rel_layers)
            )

        return list(lora_ent_embeddings_list), lora_rel_embeddings_list

    def _build_rel_lora_module(
        self,
        rows: int,
        r: int,
        init_method: str,
        ref_mean,
        ref_std,
        rel_source_weight: torch.Tensor,
        layer_idx: int,
        total_layers: int,
    ):
        """Create one relation-side LoRA adapter of shape (rows, emb_dim, r) and
        initialize it with the requested scheme.

        Extracted from the original inline block in expand_lora_embeddings so the
        layered (num_rel_layers>1) path can construct each layer independently.
        For num_rel_layers==1 with rows=new_rel_embeddings_len and r=rel_r this is
        numerically identical to the pre-P0 initialization sequence.
        """
        if init_method == "dora":
            mod = DoRALoraEmbedding(
                rows,
                self.args.emb_dim,
                r,
                merge_weights=False,
                use_rs_lora=bool(getattr(self.args, "use_rs_lora", False)),
            ).to(self.args.device, dtype=model_dtype(self.args))
            _init_nonempty(xavier_normal_, mod.weight)
            return mod

        mod = loralib.Embedding(rows, self.args.emb_dim, r).to(self.args.device, dtype=model_dtype(self.args))
        if init_method == "xavier":
            _init_nonempty(xavier_normal_, mod.weight)
        elif init_method == "zero_b":
            _init_nonempty(xavier_normal_, mod.weight)
            with torch.no_grad():
                rel_a_param = getattr(mod, "lora_embedding_A", None)
                rel_b_param = getattr(mod, "lora_embedding_B", None)
                if rel_a_param is None or rel_b_param is None:
                    rel_a_param = getattr(mod, "lora_A", None)
                    rel_b_param = getattr(mod, "lora_B", None)
                if rel_a_param is not None and rel_b_param is not None:
                    _init_nonempty(
                        nn.init.kaiming_uniform_,
                        cast(torch.Tensor, rel_a_param.data),
                        a=math.sqrt(5),
                    )
                    _init_nonempty(nn.init.zeros_, cast(torch.Tensor, rel_b_param.data))
        elif init_method == "quant_svd":
            self._init_quant_svd_module(mod, rel_source_weight, rows, layer_idx, total_layers)
        elif init_method == "qr":
            with torch.no_grad():
                eff_r = min(r, rows, self.args.emb_dim)
                left_rand = torch.randn(rows, eff_r, device=self.args.device, dtype=torch.float32)
                right_rand = torch.randn(self.args.emb_dim, eff_r, device=self.args.device, dtype=torch.float32)
                left_q, _ = torch.linalg.qr(left_rand, mode='reduced')
                right_q, _ = torch.linalg.qr(right_rand, mode='reduced')
                rel_us = left_q[:, :eff_r] * ref_std
                rel_vh = right_q[:, :eff_r].T
                rel_a_param = getattr(mod, "lora_embedding_A", None)
                rel_b_param = getattr(mod, "lora_embedding_B", None)
                if rel_a_param is None or rel_b_param is None:
                    rel_a_param = getattr(mod, "lora_A", None)
                    rel_b_param = getattr(mod, "lora_B", None)
                if rel_a_param is not None and rel_b_param is not None:
                    rel_a = cast(torch.Tensor, rel_a_param.data)
                    rel_b = cast(torch.Tensor, rel_b_param.data)
                    if rel_a.shape[1] == self.args.emb_dim and rel_b.shape[0] == rows:
                        rel_a[:eff_r, :] = rel_vh.clone()
                        rel_b[:, :eff_r] = rel_us.clone()
                    elif rel_a.shape[1] == rows and rel_b.shape[0] == self.args.emb_dim:
                        rel_a[:eff_r, :] = rel_us.T.clone()
                        rel_b[:, :eff_r] = rel_vh.T.clone()
                    else:
                        _init_nonempty(xavier_normal_, mod.weight)
                    mod.weight.data.zero_()
                else:
                    _init_nonempty(xavier_normal_, mod.weight)
        else:
            with torch.no_grad():
                rel_temp_w = torch.randn(rows, self.args.emb_dim, device=self.args.device, dtype=torch.float32) * ref_std + ref_mean
                U, S, Vh = torch.linalg.svd(rel_temp_w, full_matrices=False)
                eff_r = min(r, Vh.size(0), U.size(1))
                rel_a_param = getattr(mod, "lora_embedding_A", None)
                rel_b_param = getattr(mod, "lora_embedding_B", None)
                if rel_a_param is None or rel_b_param is None:
                    rel_a_param = getattr(mod, "lora_A", None)
                    rel_b_param = getattr(mod, "lora_B", None)
                if rel_a_param is not None and rel_b_param is not None:
                    rel_a = cast(torch.Tensor, rel_a_param.data)
                    rel_b = cast(torch.Tensor, rel_b_param.data)
                    rel_us = U[:, :eff_r] @ torch.diag(S[:eff_r])
                    rel_vh = Vh[:eff_r, :]
                    if rel_a.shape[1] == self.args.emb_dim and rel_b.shape[0] == rows:
                        rel_a[:eff_r, :] = rel_vh.clone()
                        rel_b[:, :eff_r] = rel_us.clone()
                    elif rel_a.shape[1] == rows and rel_b.shape[0] == self.args.emb_dim:
                        rel_a[:eff_r, :] = rel_us.T.clone()
                        rel_b[:, :eff_r] = rel_vh.T.clone()
                    else:
                        _init_nonempty(xavier_normal_, mod.weight)
                    mod.weight.data.zero_()
                else:
                    _init_nonempty(xavier_normal_, mod.weight)
        return mod

    def get_lora_embeddings(self):
        raise NotImplementedError("子类需要实现 LoRA 嵌入聚合逻辑")

    def switch_snapshot(self):
        if self.lora_ent_embeddings_list is not None:
            assert self.lora_rel_embeddings_list is not None
            new_ent_embeddings = self.ent_embeddings.weight.data
            new_rel_embeddings = self.rel_embeddings.weight.data
            old_num_ent = int(self.kg.snapshots[self.args.snapshot - 1].num_ent)
            old_num_rel = int(self.kg.snapshots[self.args.snapshot - 1].num_rel)
            lora_ent_embeddings, lora_rel_embeddings = self.get_lora_embeddings()
            new_ent_embeddings[old_num_ent:] = Parameter(lora_ent_embeddings.detach().clone())
            new_rel_embeddings[old_num_rel:] = Parameter(lora_rel_embeddings.detach().clone())
            self.ent_embeddings.weight = Parameter(new_ent_embeddings)
            self.rel_embeddings.weight = Parameter(new_rel_embeddings)
        self.store_old_parameters()
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = Parameter(
            self.ent_embeddings.weight.data
        )
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = Parameter(
            self.rel_embeddings.weight.data
        )
        self.ent_embeddings.weight = Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = Parameter(new_rel_embeddings)
        self.ent_embeddings.weight.requires_grad = False
        self.rel_embeddings.weight.requires_grad = False
        lora_ent_list_tmp, lora_rel_list_tmp = self.expand_lora_embeddings()
        self.lora_ent_embeddings_list = nn.ModuleList(lora_ent_list_tmp)
        self.lora_rel_embeddings_list = nn.ModuleList(lora_rel_list_tmp)
        self._rebuild_interlayer_modules()
        self._rebuild_lora_cache()

    def _rebuild_lora_cache(self):
        """Precompute per-snapshot tensors used in the hot margin_loss /
        predict / get_lora_embeddings path.

        Before this cache, every batch rebuilt three things that are actually
        constant throughout a snapshot:
          1. `torch.arange(self.lora_ent_len).to(device)`  (one per ent layer)
          2. `list(range(old_num_ent)) + self.new_ordered_entities`  (Python
             list of size ~70k on HYBRID S3)
          3. `all_ent[ent_indices]` with that Python list as fancy index
             (forces a host->device index transfer each batch)

        Observation: `ent_indices[:old_num_ent]` is identity on the old /
        frozen part, so the permutation truly only acts on the new-entity
        section. We collapse it into a small `new_num_ent`-sized long tensor
        `_ent_canonical_perm` applied directly to `lora_ent_embeddings` inside
        `get_lora_embeddings` -- margin_loss can then just `torch.cat` + gather
        with no Python hop per batch.
        """
        if self.lora_ent_embeddings_list is None or self.lora_rel_embeddings_list is None:
            self._ent_arange_cache = None
            self._rel_arange_list = None
            self._ent_canonical_perm = None
            return

        device = self.args.device

        # (1) entity-side arange: all ent layers share the same lora_ent_len
        # (last layer may hold padding rows that are trimmed via the canonical
        # perm below, so this arange covers the full padded length).
        ent_len = int(self.lora_ent_len)
        self._ent_arange_cache = torch.arange(ent_len, device=device, dtype=torch.long)

        # (2) relation-side arange per layer: last layer may have fewer rows
        # when new_rel_len % num_rel_layers != 0.
        rel_arange_list = []
        for i in range(len(self.lora_rel_embeddings_list)):
            layer_mod = cast(nn.Module, self.lora_rel_embeddings_list[i])
            rows = int(cast(torch.Tensor, layer_mod.weight).shape[0])  # type: ignore[attr-defined]
            rel_arange_list.append(torch.arange(rows, device=device, dtype=torch.long))
        self._rel_arange_list = rel_arange_list

        # (3) canonical-order permutation for new entities. switch_snapshot is
        # called at end of snap s iteration, so args.snapshot = s here and
        # `self.new_ordered_entities` holds global ids in
        # [num_ent_s, num_ent_{s+1}) sorted by (distance, centrality). At
        # training time (snap s+1) margin_loss does the equivalent of
        # `all_ent[new_ordered_entities[j]]` i.e. row
        # `(new_ordered_entities[j] - num_ent_s)` of the lora layer output.
        old_num_ent = int(self.kg.snapshots[int(self.args.snapshot)].num_ent)
        perm = [int(gid) - old_num_ent for gid in self.new_ordered_entities]
        self._ent_canonical_perm = torch.tensor(perm, device=device, dtype=torch.long)

    def _get_param_lr_scale(self, name: str) -> float:
        if self.ent_layer_lr_scales is None:
            return 1.0
        ent_match = re.search(r"lora_ent_embeddings_list\.(\d+)\.", name)
        if ent_match is not None:
            layer_idx = int(ent_match.group(1))
            if 0 <= layer_idx < len(self.ent_layer_lr_scales):
                return float(self.ent_layer_lr_scales[layer_idx])
        interlayer_proj_match = re.search(r"interlayer_ent_projs\.(\d+)\.", name)
        if interlayer_proj_match is not None:
            layer_idx = int(interlayer_proj_match.group(1)) + 1
            if 0 <= layer_idx < len(self.ent_layer_lr_scales):
                return float(self.ent_layer_lr_scales[layer_idx])
        return 1.0

    # LoRA+ optimizer: separate lr for A / B, independent weight decay per group,
    # and keep DoRA scale parameter lora_g in a dedicated group (no weight decay).
    def get_lora_plus_optimizer(
        self,
        base_lr=1e-3,
        loraplus_ratio=16.0,
        weight_decay=0.0,
        lora_wd_a=0.0,
        lora_wd_b=0.0,
    ):
        param_A = []
        param_B = []
        param_g = []
        param_other = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            lr_scale = self._get_param_lr_scale(name)
            if 'lora_B' in name or 'lora_embedding_B' in name:
                param_B.append((param, lr_scale))
            elif 'lora_A' in name or 'lora_embedding_A' in name:
                param_A.append((param, lr_scale))
            elif name.endswith('lora_g') or '.lora_g' in name:
                param_g.append((param, lr_scale))
            else:
                param_other.append((param, lr_scale))

        def _append_groups(param_entries, lr_value, wd_value):
            grouped = {}
            for param, lr_scale in param_entries:
                key = round(float(lr_scale), 6)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(param)
            for lr_scale, params in grouped.items():
                param_groups.append(
                    {
                        'params': params,
                        'lr': lr_value * float(lr_scale),
                        'weight_decay': wd_value,
                    }
                )

        param_groups = []
        if param_A:
            _append_groups(param_A, base_lr, lora_wd_a)
        if param_B:
            _append_groups(param_B, base_lr * loraplus_ratio, lora_wd_b)
        if param_g:
            _append_groups(param_g, base_lr, 0.0)
        if param_other:
            _append_groups(param_other, base_lr, weight_decay)

        if not param_groups:
            # Fallback: ensure the optimizer has at least one (possibly empty) group
            # so callers can still call .step(). AdamW requires >=1 group.
            param_groups.append({'params': [], 'lr': base_lr, 'weight_decay': weight_decay})

        # AdamW with wd=0 is numerically equivalent to Adam; with wd>0 we get
        # decoupled weight decay (better behaved for LoRA B norm growth).
        optimizer = torch.optim.AdamW(param_groups)
        return optimizer


class TransE(LoraKGE_Layers):
    def __init__(self, args, kg) -> None:
        super(TransE, self).__init__(args, kg)
        self.huber_loss = torch.nn.HuberLoss(reduction='sum')

    def new_loss(self, head, rel, tail=None, label=None):
        return self.margin_loss(head, rel, tail, label) / head.size(0)

    def score_fun(self, h, r, t):
        h = self.norm_ent(h)
        r = self.norm_rel(r)
        t = self.norm_ent(t)
        return torch.norm(h + r - t, 1, -1)

    def split_pn_score(self, score, label):
        p_score = score[torch.where(label > 0)]
        n_score = (score[torch.where(label < 0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def get_lora_embeddings(self):
        """Forward each LoRA layer on its cached arange tensor and return the
        per-entity / per-relation deltas already re-ordered to match the
        canonical global-id layout used by the rest of the pipeline.

        Ent output shape: (new_ent_embeddings_len, emb_dim). Row j corresponds
        to the j-th new entity in canonical order (i.e. global id
        old_num_ent + j). Rel output shape: (new_rel_embeddings_len, emb_dim),
        already in canonical order since rel has no centrality sort.
        """
        assert self.lora_ent_embeddings_list is not None
        assert self.lora_rel_embeddings_list is not None
        assert self._ent_arange_cache is not None, "LoRA cache not built; did switch_snapshot run?"
        assert self._rel_arange_list is not None
        assert self._ent_canonical_perm is not None

        with record_function("lora/ent_layers_forward"):
            num_ent_layers = int(self.args.num_ent_layers)
            arange_ent = self._ent_arange_cache
            if num_ent_layers == 1:
                lora_ent_concat = self.lora_ent_embeddings_list[0].forward(arange_ent)
            else:
                ent_parts = []
                previous_part = None
                for i in range(num_ent_layers):
                    current_part = cast(nn.Module, self.lora_ent_embeddings_list[i]).forward(arange_ent)
                    current_part = self._fuse_with_previous_entity_layer(i, current_part, previous_part)
                    ent_parts.append(current_part)
                    previous_part = current_part
                lora_ent_concat = torch.cat(ent_parts, dim=0)
            # Canonical reordering (drops phantom rows from the last padded
            # layer as a side effect since perm only references valid rows).
            lora_ent_embeddings = lora_ent_concat.index_select(0, self._ent_canonical_perm)

        with record_function("lora/rel_layers_forward"):
            num_rel_layers = len(self.lora_rel_embeddings_list)
            if num_rel_layers == 1:
                layer_mod = cast(nn.Module, self.lora_rel_embeddings_list[0])
                lora_rel_embeddings = layer_mod(self._rel_arange_list[0])
            else:
                rel_parts = []
                for i in range(num_rel_layers):
                    layer_mod = cast(nn.Module, self.lora_rel_embeddings_list[i])
                    rel_parts.append(layer_mod(self._rel_arange_list[i]))
                lora_rel_embeddings = torch.cat(rel_parts, dim=0)[: self.new_rel_embeddings_len]
        return lora_ent_embeddings, lora_rel_embeddings

    def embedding(self, stage=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.args.snapshot == 0:
            ent_embeddings = self.ent_embeddings.weight
            rel_embeddings = self.rel_embeddings.weight
        else:
            ent_embeddings = cast(torch.Tensor, self.old_data_ent_embeddings_weight)
            rel_embeddings = cast(torch.Tensor, self.old_data_rel_embeddings_weight)
        return ent_embeddings, rel_embeddings

    def predict(self, head, relation, stage='Valid'):
        if stage != 'Test':
            num_ent = self.kg.snapshots[self.args.snapshot_valid].num_ent
        else:
            num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
        if self.args.snapshot == 0:
            ent_embeddings, rel_embeddings = self.embedding(stage)
            h = torch.index_select(ent_embeddings, 0, head)
            r = torch.index_select(rel_embeddings, 0, relation)
            t_all = ent_embeddings[:num_ent]
        else:
            ent_embeddings, rel_embeddings = self.embedding(stage)
            lora_ent_embeddings, lora_rel_embeddings = self.get_lora_embeddings()
            # lora_* already in canonical order; see get_lora_embeddings.
            all_ent_embeddings = torch.cat([ent_embeddings, lora_ent_embeddings], dim=0)
            all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
            h = torch.index_select(all_ent_embeddings, 0, head)
            r = torch.index_select(all_rel_embeddings, 0, relation)
            t_all = all_ent_embeddings[:num_ent]

        h = self.norm_ent(h)
        r = self.norm_rel(r)
        t_all = self.norm_ent(t_all)

        pred_t = h + r
        chunk_size = 2048
        score_chunks = []
        for start in range(0, t_all.size(0), chunk_size):
            end = min(start + chunk_size, t_all.size(0))
            t_chunk = t_all[start:end]
            chunk_score = 9.0 - torch.norm(pred_t.unsqueeze(1) - t_chunk, p=1, dim=2)
            score_chunks.append(torch.sigmoid(chunk_score))
        return torch.cat(score_chunks, dim=1)

    def _lookup_ent_with_lora(
        self,
        idx: torch.Tensor,
        ent_old: torch.Tensor,
        lora_new: torch.Tensor,
    ) -> torch.Tensor:
        """P2 sparse-gather: return `torch.cat([ent_old, lora_new])[idx]` without
        ever materializing the (old_num+new_num, D) concat.

        On HYBRID S3 (old=70k, new=~1000, D=200) the cat was ~56 MB per batch;
        a batch only touches batch_size * (1+neg_ratio) ~ 11k rows, so >99% of
        the cat was wasted. We gather from ent_old and lora_new independently
        and merge with torch.where on a bool mask.

        Correctness: for any row i with idx[i] < old_num the returned row equals
        ent_old[idx[i]] (is_new == False branch); for idx[i] >= old_num it
        equals lora_new[idx[i] - old_num] (is_new == True branch). The clamps
        keep the "wrong side" gather on an in-range index so it doesn't fault;
        torch.where then discards that garbage row.
        """
        old_num = ent_old.size(0)
        is_new = idx >= old_num
        safe_old_idx = torch.clamp(idx, max=old_num - 1)
        safe_new_idx = torch.clamp(idx - old_num, min=0)
        h_old = ent_old.index_select(0, safe_old_idx)
        h_new = lora_new.index_select(0, safe_new_idx)
        return torch.where(is_new.unsqueeze(-1), h_new, h_old)

    def margin_loss(self, head, rel, tail, label=None):
        if self.args.snapshot == 0:
            ent_embeddings, rel_embeddings = self.embedding('Train')
            h = torch.index_select(ent_embeddings, 0, head)
            r = torch.index_select(rel_embeddings, 0, rel)
            t = torch.index_select(ent_embeddings, 0, tail)
        else:
            with record_function("lora/embedding_base"):
                ent_embeddings, rel_embeddings = self.embedding('Train')
            with record_function("lora/get_lora_embeddings"):
                lora_ent_embeddings, lora_rel_embeddings = self.get_lora_embeddings()
            # lora_ent_embeddings is already in canonical new-entity order
            # (see get_lora_embeddings). Relation table is small (a few hundred
            # rows) so the cat on the relation side is cheap -- we only skip
            # the entity-side cat which is the one that costs memory.
            use_sparse = str(getattr(self.args, "lora_sparse_gather", "on")) == "on"
            if use_sparse:
                with record_function("lora/gather_ent_mixed"):
                    h = self._lookup_ent_with_lora(head, ent_embeddings, lora_ent_embeddings)
                    t = self._lookup_ent_with_lora(tail, ent_embeddings, lora_ent_embeddings)
                with record_function("lora/gather_rel"):
                    all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
                    r = torch.index_select(all_rel_embeddings, 0, rel)
            else:
                with record_function("lora/assemble_full_embeddings"):
                    all_ent_embeddings = torch.cat([ent_embeddings, lora_ent_embeddings], dim=0)
                    all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
                with record_function("lora/index_select"):
                    h = torch.index_select(all_ent_embeddings, 0, head)
                    r = torch.index_select(all_rel_embeddings, 0, rel)
                    t = torch.index_select(all_ent_embeddings, 0, tail)
        with record_function("lora/score_fun"):
            score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y)

    def get_TransE_loss(self, head, relation, tail, label):
        return self.new_loss(head, relation, tail, label)

    def loss(self, head, relation, tail=None, label=None):
        loss = self.get_TransE_loss(head, relation, tail, label)
        return loss