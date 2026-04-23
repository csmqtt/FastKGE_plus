import math
import shutil
from datetime import datetime
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.utils import *
from src.parse_args import args
from src.data_load.KnowledgeGraph import KnowledgeGraph
from src.model.LoraKGE_Layers import TransE as LoraKGE_Layers
from src.train import *
from src.test import *
from src.plot_loss import plot_loss_curve


def _build_lr_scheduler(optimizer, args):
    """Linear warmup -> cosine decay on a shared multiplier (preserves LoRA+ ratio across groups).

    V2.4: by default the scheduler is applied ONLY to snapshot 0 (the base TransE
    training). On LoRA snapshots (snap >= 1), constant lr matches the V1.x
    baseline and avoids the ~35% effective-training-budget cut that cosine
    decay introduces — empirically that cut causes a ~5-6pp MRR regression on
    FB_CKGE S1/S2/S3. Set -scheduler_lora_snapshots to opt back into the old
    "scheduler everywhere" behavior.
    """
    if not getattr(args, "use_lr_scheduler", False):
        return None
    if int(getattr(args, "snapshot", 0)) > 0 and not getattr(
        args, "scheduler_lora_snapshots", False
    ):
        return None
    warmup = max(1, int(getattr(args, "warmup_epochs", 5)))
    total = max(1, int(args.epoch_num))
    min_ratio = float(getattr(args, "min_lr_ratio", 0.1))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

class Instructor():
    """ The instructor of the model """
    def __init__(self, args) -> None:

        self.args = args
        self.loss_history = {}

        """ 1. Prepare for path, logger and device """
        self.prepare()

        """ 2. Load data """
        self.kg = KnowledgeGraph(args)

        """ 3. Create models and optimizer """
        self.model, self.optimizer = self.create_model()

        self.args.logger.info(self.args)

    def _make_optimizer(self):
        return self.model.get_lora_plus_optimizer(  # type: ignore[attr-defined]
            base_lr=float(self.args.learning_rate),
            loraplus_ratio=float(getattr(self.args, "loraplus_ratio", 16.0)),
            weight_decay=float(self.args.l2),
            lora_wd_a=float(getattr(self.args, "lora_wd_a", 0.0)),
            lora_wd_b=float(getattr(self.args, "lora_wd_b", 0.0)),
        )

    def create_model(self):
        """ Create KGE model and optimizer """
        if self.args.model_name == "LoraKGE_Layers":
            model = LoraKGE_Layers(self.args, self.kg)
        else:
            model = LoraKGE_Layers(self.args, self.kg)
        model.to(self.args.device)
        self.model = model

        optimizer = self._make_optimizer()
        self.scheduler = _build_lr_scheduler(optimizer, self.args)
        return model, optimizer

    def reset_model(self, model=False, optimizer=False):
        """
        Reset model or optimizer
        :param model: If True: reset the model and optimizer
        :param optimizer: If True: reset the optimizer
        """
        if model:
            self.model, self.optimizer = self.create_model()
        if optimizer:
            self.optimizer = self._make_optimizer()
            self.scheduler = _build_lr_scheduler(self.optimizer, self.args)

    def prepare(self):
        """ Set data path """
        if not os.path.exists(args.data_path):
            os.mkdir(args.data_path)
        self.args.data_path = args.data_path + args.dataset + "/"

        """ Set save path """
        self.args.save_path = args.save_path + args.dataset
        if os.path.exists(args.save_path):
            shutil.rmtree(args.save_path, True)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        if self.args.note != '':
            self.args.save_path += self.args.note
        if os.path.exists(args.save_path):
            shutil.rmtree(args.save_path, True)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        """ Set log path """
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        self.args.log_path = args.log_path + datetime.now().strftime("%Y%m%d%H%M%S/")
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        self.args.log_path = args.log_path + args.dataset
        if self.args.note != "":
            self.args.log_path += self.args.note

        """ Set logger """
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        logging_file_name = f'{args.log_path}.log'
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

        """ Set device """
        torch.cuda.set_device(int(args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device

    def next_snapshot_setting(self):
        """ Prepare for next snapshot """
        self.model.switch_snapshot()

    def run(self):
        """ Run the instructor of the model. The training process on all snapshots """
        report_results = PrettyTable()
        report_results.field_names = ['Snapshot', 'Time', 'Whole_MRR', 'Whole_Hits@1', 'Whole_Hits@3', 'Whole_Hits@10']
        test_results = []
        training_times = []
        BWT = [] # h(n, i) - h(i, i)
        FWT = [] # h(i- 1, i)
        first_learning_res = []

        """ training process """
        for ss_id in range(int(self.args.snapshot_num)):
            best_checkpoint = os.path.join(
                self.args.save_path, f'{str(ss_id - 1)}model_best.tar'
            )

            self.args.snapshot = ss_id
            self.args.snapshot_test = ss_id
            self.args.snapshot_valid = ss_id


            """ preprocess before training on a snapshot """
            self.model.pre_snapshot()

            if ss_id > 0:
                self.args.test_FWT = True
                res_before = self.test()
                FWT.append(res_before['mrr'])
            self.args.test_FWT = False

            training_time = self.train()

            """ prepare result table """
            test_res = PrettyTable()
            test_res.field_names = [
                f'Snapshot:{str(ss_id)}',
                'MRR',
                'Hits@1',
                'Hits@3',
                'Hits@5',
                'Hits@10',
            ]

            best_checkpoint = os.path.join(
                self.args.save_path, f'{str(ss_id)}model_best.tar'
            )
            self.load_checkpoint(best_checkpoint)

            self.model.snapshot_post_processing()

            reses = []
            for test_ss_id in range(ss_id + 1):
                self.args.snapshot_test = test_ss_id
                res = self.test()
                if test_ss_id == ss_id:
                    first_learning_res.append(res['mrr'])
                test_res.add_row([
                    test_ss_id, res['mrr'], res['hits1'], res['hits3'], res['hits5'], res['hits10']
                ])
                reses.append(res)
            if ss_id == self.args.snapshot_num - 1:
                BWT.extend(
                    reses[iid]['mrr'] - first_learning_res[iid]
                    for iid in range(self.args.snapshot_num - 1)
                )
            self.args.logger.info(f"\n{test_res}")
            test_results.append(test_res)

            """ record report results """
            whole_mrr, whole_hits1, whole_hits3, whole_hits10 = self.get_report_results(reses)
            report_results.add_row([ss_id, training_time, whole_mrr, whole_hits1, whole_hits3, whole_hits10])
            training_times.append(training_time)

            if self.args.snapshot < int(self.args.snapshot_num) - 1:
                self.next_snapshot_setting()
                # V2.4.1 fix: advance args.snapshot BEFORE rebuilding the
                # optimizer/scheduler, so _build_lr_scheduler's snapshot-based
                # gating sees the snapshot this optimizer is actually for.
                # Without this, the scheduler built here for the *next* snapshot
                # reads the current (old) snapshot id and is always built for
                # snap 0's semantics.
                self.args.snapshot = ss_id + 1
                self.reset_model(optimizer=True)
        self.args.logger.info(f'Final Result:\n{test_results}')
        self.args.logger.info(f'Report Result:\n{report_results}')
        self.args.logger.info(f'Sum_Training_Time:{sum(training_times)}')
        self.args.logger.info(f'Every_Training_Time:{training_times}')
        self.args.logger.info(
            f'Forward transfer: {sum(FWT) / len(FWT)} Backward transfer: {sum(BWT) / len(BWT)}'
        )

    def get_report_results(self, results):
        mrrs, hits1s, hits3s, hits10s, num_test = [], [], [], [], []
        for idx, result in enumerate(results):
            mrrs.append(result['mrr'])
            hits1s.append(result['hits1'])
            hits3s.append(result['hits3'])
            hits10s.append(result['hits10'])
            num_test.append(len(self.kg.snapshots[idx].test))
        whole_mrr = sum(
            mrr * num_test[i] for i, mrr in enumerate(mrrs)
            ) / sum(num_test)
        whole_hits1 = sum(
            hits1 * num_test[i] for i, hits1 in enumerate(hits1s)
        ) / sum(num_test)
        whole_hits3 = sum(
            hits3 * num_test[i] for i, hits3 in enumerate(hits3s)
        ) / sum(num_test)
        whole_hits10 = sum(
            hits10 * num_test[i] for i, hits10 in enumerate(hits10s)
        ) / sum(num_test)
        return round(whole_mrr, 3), round(whole_hits1, 3), round(whole_hits3, 3), round(whole_hits10, 3)

    def train(self):
        """ Training process, return training time """
        start_time = time.time()
        print("Start training =============================")
        self.best_valid = 0.0
        self.stop_epoch = 0
        trainer = Trainer(self.args, self.kg, self.model, self.optimizer)
        epoch_losses = []

        """ Trainign iteration """
        for epoch in range(int(self.args.epoch_num)):
            self.args.epoch = epoch
            """ training """
            loss, valid_res, lora_stats = trainer.run_epoch()
            epoch_losses.append(float(loss))
            """ per-epoch LR schedule (warmup + cosine) """
            scheduler = getattr(self, "scheduler", None)
            if scheduler is not None:
                scheduler.step()
            """ early stop """
            if self.args.debug:
                if epoch > 0:
                    break
            # Early-stop counter is suppressed during warmup: while warming up,
            # lr_A ramps 0→base, so MRR can appear "flat" for several epochs
            # even though the optimizer is progressing. Counting stop_epoch here
            # would early-stop before training has had any real chance to
            # update, which was observed on FB_CKGE snapshots >= 2 (stopping at
            # epoch 3). Only active when the scheduler is actually running for
            # this snapshot (see _build_lr_scheduler for V2.4 snapshot gating).
            scheduler_active = getattr(self, "scheduler", None) is not None
            warmup = int(getattr(self.args, "warmup_epochs", 0)) if scheduler_active else 0
            in_warmup = epoch < warmup
            # V2.10: decouple "best checkpoint" from "stop-counter reset".
            # Any improvement (even 1e-5 raw MRR) still updates the saved best
            # checkpoint so the final model uses the strongest weights we ever saw,
            # but only improvements >= es_min_delta reset stop_epoch. This removes
            # the sensitivity of early-stop to floating-point noise observed
            # between V2.8 and V2.9 (S0 stopping at 10 vs 7 epochs from a 0.05pp
            # wiggle), making per-epoch efficiency measurements meaningful.
            current = valid_res[self.args.valid_metrics]
            min_delta = float(getattr(self.args, "es_min_delta", 1e-3))
            is_best = current > self.best_valid
            is_meaningful = current > self.best_valid + min_delta
            if is_best:
                self.best_valid = current
                if self.args.snapshot == 0:
                    self.save_model(is_best=True, lora=False)
                else:
                    self.save_model(is_best=True, lora=True)
            else:
                if self.args.snapshot == 0:
                    self.save_model(lora=False)
                else:
                    self.save_model(lora=True)
            if is_meaningful:
                self.stop_epoch = 0
            else:
                if not in_warmup:
                    self.stop_epoch += 1
                if self.stop_epoch >= self.args.patience:
                    self.args.logger.info(
                        f'Early Stopping! Snapshot:{self.args.snapshot} Epoch: {epoch} Best Results: {round(self.best_valid * 100, 3)}'
                    )
                    break
            """ logging """
            if epoch % 1 == 0:
                base_log = (
                    f"Snapshot:{self.args.snapshot}\tEpoch:{epoch}\tLoss:{round(loss, 3)}"
                    f"\tMRR:{round(valid_res['mrr'] * 100, 3)}\tHits@10:{round(valid_res['hits10'] * 100, 3)}"
                    f"\tBest:{round(self.best_valid * 100, 3)}"
                )
                if getattr(self.args, "log_lora_stats", False):
                    a_grad_avg = lora_stats["a_grad_norm_sum"] / max(1, lora_stats["a_grad_steps"])
                    b_grad_avg = lora_stats["b_grad_norm_sum"] / max(1, lora_stats["b_grad_steps"])
                    base_log += (
                        f"\tLoRAStats("
                        f"lrA={lora_stats['lr_group_0']:.2e},lrB={lora_stats['lr_group_1']:.2e},"
                        f"gradA={a_grad_avg:.3e},gradB={b_grad_avg:.3e},"
                        f"paramA={lora_stats['a_param_norm']:.3e},paramB={lora_stats['b_param_norm']:.3e},"
                        f"countA={lora_stats['a_param_count']},countB={lora_stats['b_param_count']}"
                        f")"
                    )
                self.args.logger.info(base_log)
        end_time = time.time()
        self.loss_history[self.args.snapshot] = epoch_losses
        return end_time - start_time

    def test(self):
        tester = Tester(self.args, self.kg, self.model)
        return tester.test()

    def save_model(self, is_best=False, lora=False):
        if lora == False:
            checkpoint_dict = {'state_dict': self.model.state_dict()}
            checkpoint_dict['epoch_id'] = self.args.epoch
            out_tar = os.path.join(
                self.args.save_path,
                f'{str(self.args.snapshot)}checkpoint-{self.args.epoch}.tar',
            )
            torch.save(checkpoint_dict, out_tar)
            if is_best:
                best_path = os.path.join(
                    self.args.save_path, f'{str(self.args.snapshot)}model_best.tar'
                )
                shutil.copyfile(out_tar, best_path)
        else:
            out_tar = os.path.join(
                self.args.save_path,
                f'{str(self.args.snapshot)}checkpoint-{self.args.epoch}.tar',
            )
            torch.save(loralib.lora_state_dict(self.model), out_tar)
            if is_best:
                best_path = os.path.join(
                    self.args.save_path, f'{str(self.args.snapshot)}model_best.tar'
                )
                shutil.copyfile(out_tar, best_path)


    def load_checkpoint(self, input_file):
        if self.args.snapshot == 0:
            if os.path.isfile(input_file):
                logging.info(f"=> loading checkpoint \'{input_file}\'")
                checkpoint = torch.load(input_file, map_location=f"cuda:{self.args.gpu}")
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                logging.info(f'=> no checking found at \'{input_file}\'')
        else:
            if os.path.isfile(input_file):
                logging.info(f"=> loading checkpoint \'{input_file}\'")
                checkpoint = torch.load(input_file, map_location=f"cuda:{self.args.gpu}")
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                logging.info(f'=> no checking found at \'{input_file}\'')


""" Main function """
if __name__ == "__main__":
    set_seeds(args.random_seed)
    ins = Instructor(args)
    ins.run()
    # Plot after all training/testing logs have been emitted.
    plot_loss_curve(ins.loss_history, ins.args.log_path, ins.args.logger)