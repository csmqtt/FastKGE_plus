import argparse
import sys
parser = argparse.ArgumentParser(description="Parser For Arguments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# paths
parser.add_argument("-save_path", dest="save_path", default="./checkpoint/", help="Path of saved models")
parser.add_argument("-log_path", dest="log_path", default="./logs/", help="Path of saved logs")
parser.add_argument("-data_path", dest="data_path", default="./data/", help="Path of dataset")

# global setting
parser.add_argument("-random_seed", dest="random_seed", type=int, default=3407, help="Set random seeds")
parser.add_argument("-dataset", dest="dataset", default="ENTITY", help="dataset name")
parser.add_argument("-gpu", dest="gpu", default=0, help="number of gpu")

# model setting
parser.add_argument("-model_name", dest="model_name", default="LoraKGE", help="name of model")
parser.add_argument("-batch_size", dest="batch_size", type=int, default=1024, help="Set the batch size")
parser.add_argument("-learning_rate", dest="learning_rate", type=float, default=1e-4, help="Set the learning rate")
parser.add_argument("-epoch_num", dest="epoch_num", type=int, default=200, help="Set the epoch")
parser.add_argument("-note", dest='note', default='', help='The note of log file name')
parser.add_argument("-snapshot_num", dest="snapshot_num", type=int, default=5, help="The number of snapshots")
parser.add_argument("-emb_dim", dest="emb_dim", type=int, default=200, help="embedding dimension")
parser.add_argument("-margin", dest="margin", type=float, default=8.0, help="The margin of MarginLoss")
parser.add_argument("-neg_ratio", dest="neg_ratio", type=int, default=10, help="the ratio of negtive/postive facts")
parser.add_argument("-l2", dest='l2', type=float, default=0.0, help="optimizer l2")
parser.add_argument("-num_layer", dest="num_layer", type=int, default=1, help='MAE layer')
parser.add_argument("-skip_previous", dest="skip_previous", default="False", help="Allow re-training and snapshot_only models skip previous training")
parser.add_argument("-train_new", dest="train_new", default=True, help="True: Training on new facts; False: Training on all seen facts")
parser.add_argument("-valid_metrics", dest="valid_metrics", default="mrr")
parser.add_argument("-patience", dest="patience", type=int, default=3, help="early stop step")

# new updates
parser.add_argument("-debug", dest="debug", default=False, help="test mode")
parser.add_argument("-record", dest="record", default=False, help="Record the loss of different layers")
parser.add_argument("-predict_result", dest="predict_result", default=False, help="The result of predict")
parser.add_argument("-r", dest="r", type=int, default=100, help="The rank of lora")
parser.add_argument("-ent_r", dest="ent_r", type=int, default=100, help="The rank of ent lora")
parser.add_argument("-rel_r", dest="rel_r", type=int, default=10, help="The rank of rel lora")
parser.add_argument("-r_fixed", dest="r_fixed", default=True, help="fix the r")
parser.add_argument("-using_multi_layers", dest="using_multi_layers", default=False, help="Use multi_layers or not")
parser.add_argument("-multi_layers_path", dest="multi_layers_path", default="train_sorted_by_edges_betweenness.txt", help="New_path")
parser.add_argument("-num_ent_layers", dest="num_ent_layers", type=int, default=10, help="The length of ent embeddings list")
parser.add_argument("-num_rel_layers", dest="num_rel_layers", type=int, default=1, help="The length of rel embeddings list")
parser.add_argument("-using_various_ranks", dest="using_various_ranks", default=False, help="Using various ranks or not")
parser.add_argument("-using_various_ranks_reverse", dest="using_various_ranks_reverse", default=False, help="Using reverse various ranks or not")
parser.add_argument("-explore", dest="explore", default=False, help="Explorable experiments")
parser.add_argument("-lora_init", dest="lora_init", default="xavier", choices=["pissa", "xavier", "qr", "dora", "zero_b", "quant_svd"], help="LoRA 初始化方式：pissa 为随机矩阵 SVD，xavier 为随机初始化，qr 为正交初始化，dora 为逐维缩放的 LoRA，zero_b 为 A 随机 B 置零，quant_svd 为量化残差 SVD 初始化")
parser.add_argument("-use_rs_lora", dest="use_rs_lora", action="store_true", help="Rank-stabilized scaling (alpha/sqrt(r)) on DoRALoraEmbedding only; no effect unless lora_init=dora")
parser.add_argument("-log_lora_stats", dest="log_lora_stats", action="store_true", help="Log per-epoch LoRA A/B grad/param norms and optimizer group learning rates")
parser.add_argument("-quant_init_bits", dest="quant_init_bits", type=int, default=8, help="量化残差初始化的量化位宽，仅在 lora_init=quant_svd 时生效")
parser.add_argument("-quant_init_granularity", dest="quant_init_granularity", default="row", choices=["row", "tensor"], help="量化残差初始化的量化粒度：row 表示逐行量化，tensor 表示整张量量化")
parser.add_argument("-entity_layering", dest="entity_layering", default="legacy", choices=["legacy", "difficulty"], help="实体分层策略：legacy 使用原始距离排序，difficulty 使用综合难度排序")
parser.add_argument("-ent_rank_policy", dest="ent_rank_policy", default="legacy", choices=["legacy", "uniform", "difficulty"], help="实体 LoRA 秩分配策略：legacy 兼容旧逻辑，uniform 均匀分配，difficulty 按层难度分配")
parser.add_argument("-difficulty_rank_scale", dest="difficulty_rank_scale", type=float, default=0.5, help="难度感知秩分配缩放系数，越大越偏向高难层")
parser.add_argument("-difficulty_lr_scale", dest="difficulty_lr_scale", type=float, default=0.0, help="难度感知学习率缩放系数，0 表示关闭")
parser.add_argument("-interlayer_lora_mode", dest="interlayer_lora_mode", default="off", choices=["off", "residual", "gate"], help="实体侧层间协同模式：off 关闭，residual 为标量残差融合，gate 为门控投影融合")
parser.add_argument("-interlayer_stopgrad", dest="interlayer_stopgrad", default="on", choices=["on", "off"], help="层间协同时是否对上一层摘要停止梯度")
parser.add_argument("-interlayer_init", dest="interlayer_init", type=float, default=0.0, help="层间协同参数初始值")
parser.add_argument("-log_layer_plan", dest="log_layer_plan", action="store_true", help="记录每个快照的分层顺序、难度、秩分配和学习率缩放")

# P1 profiling (torch.profiler). Off by default. Activates only for a single
# (snapshot, epoch) pair so we capture one representative training step without
# producing gigantic chrome traces. Typical use on HYBRID S3:
#   python main.py ... -profile_snapshot 3 -profile_epoch 1 -profile_num_batches 20
parser.add_argument("-profile_snapshot", dest="profile_snapshot", type=int, default=-1, help="If >=0, run torch.profiler on this snapshot (and this epoch). -1 disables profiling.")
parser.add_argument("-profile_epoch", dest="profile_epoch", type=int, default=1, help="Epoch index on profile_snapshot at which to attach the profiler (defaults to epoch 1 to skip first-epoch CUDA lazy init).")
parser.add_argument("-profile_num_batches", dest="profile_num_batches", type=int, default=20, help="How many batches to record when the profiler is active (first 2 are warmup).")
parser.add_argument("-profile_out", dest="profile_out", default="./logs/profile", help="Directory for chrome traces and profile summary files.")

# Efficiency knobs
parser.add_argument("-precision", dest="precision", default="fp32", choices=["fp32", "fp64"], help="Training precision. fp32 is 3-10x faster on consumer GPUs for small KGE models (RTX 30xx/40xx FP64 is ~1/32 of FP32). fp64 reproduces the original paper's double-precision setting.")
parser.add_argument("-num_workers", dest="num_workers", type=int, default=2, help="DataLoader num_workers for the training loop. 0 disables multiprocessing.")
parser.add_argument("-lora_stats_every", dest="lora_stats_every", type=int, default=0, help="When >0 and -log_lora_stats is set, compute LoRA grad stats every N batches instead of every batch (0 = only after the last batch of the epoch, the default low-overhead mode).")
parser.add_argument("-lora_sparse_gather", dest="lora_sparse_gather", default="off", choices=["on", "off"], help="P2: skip the per-batch torch.cat([ent_old, lora_new]) materialization in margin_loss. 'on' uses masked gather over ent_old and lora_new separately (numerically equivalent on CPU, but GPU kernel order differs and measured per-epoch speedup was only ~3% on HYBRID S3, so default is 'off' as of V2.10). 'off' uses the V2.7 dense cat path.")
parser.add_argument("-es_min_delta", dest="es_min_delta", type=float, default=1e-3, help="V2.10: minimum raw MRR improvement required to reset the early-stop counter (1e-3 == 0.1pp in the printed metric). Smaller 'improvements' still update the saved best checkpoint but do NOT reset stop_epoch, so early-stop is robust to floating-point noise (e.g. V2.8 vs V2.9 S0 diverging at 0.05pp). Set 0 for the strict > comparison used before V2.10.")

# LoRA+ / optimizer tuning
parser.add_argument("-loraplus_ratio", dest="loraplus_ratio", type=float, default=16.0, help="LoRA+ ratio: lr_B = lr_A * loraplus_ratio")
parser.add_argument("-lora_wd_a", dest="lora_wd_a", type=float, default=0.0, help="weight_decay for LoRA A group (AdamW)")
parser.add_argument("-lora_wd_b", dest="lora_wd_b", type=float, default=0.0, help="weight_decay for LoRA B group (AdamW)")
parser.add_argument("-lora_grad_clip", dest="lora_grad_clip", type=float, default=0.0, help="If >0, clip grad norm of LoRA params before optimizer.step()")

# LR scheduler (warmup + cosine)
parser.add_argument("-use_lr_scheduler", dest="use_lr_scheduler", action="store_true", help="Enable linear-warmup + cosine-decay scheduler (per-epoch step). By default only applies to snapshot 0; set -scheduler_lora_snapshots to also apply it to LoRA snapshots.")
parser.add_argument("-warmup_epochs", dest="warmup_epochs", type=int, default=5, help="Number of warmup epochs for LR scheduler")
parser.add_argument("-min_lr_ratio", dest="min_lr_ratio", type=float, default=0.1, help="Min lr ratio at the end of cosine decay (relative to base lr)")
parser.add_argument("-scheduler_lora_snapshots", dest="scheduler_lora_snapshots", action="store_true", help="Also apply the LR scheduler to LoRA snapshots (snap >= 1). Off by default because cosine decay empirically costs ~5pp MRR on FB_CKGE S1/S2/S3 by cutting effective training budget to ~65%.")

args = parser.parse_args()

# ---------------- FB/WN dataset-aware presets ----------------
# Only fill in values for flags the user did NOT pass explicitly on the CLI.
# This lets `-arg value` always win over presets while still giving FB/WN a
# stronger default recipe that targets their "large initial snapshot + many
# new entities per snapshot" regime.
def _user_cli_flags() -> set:
    seen: set[str] = set()
    for tok in sys.argv[1:]:
        if not tok.startswith("-"):
            continue
        name = tok.lstrip("-").split("=", 1)[0]
        if name:
            seen.add(name)
    return seen

_cli = _user_cli_flags()


def _maybe_set(name: str, value) -> None:
    if name not in _cli:
        setattr(args, name, value)


if args.dataset in ("FB_CKGE", "WN_CKGE"):
    # V2.3: structural knobs reverted to the V1.x baseline that actually worked
    # on FB/WN (see recordforstep for the full V2.2-vs-baseline regression table).
    # Keep only the ORTHOGONAL improvements: warmup+cosine scheduler, warmup-aware
    # early-stop (implemented in main.py), and the AdamW/logging plumbing.
    # lora_wd_b and lora_grad_clip are left at 0 here so this version is a clean
    # ablation on top of baseline structure; flip them on via CLI if desired.
    _maybe_set("num_ent_layers", 2)
    _maybe_set("ent_r", 150)
    _maybe_set("rel_r", 20)
    _maybe_set("lora_init", "dora")
    # Scheduler: linear warmup -> cosine decay. Helps S0 (base embeddings) most
    # in our runs; harmless for LoRA snapshots as long as min_lr_ratio is high
    # enough that late epochs can still climb.
    _maybe_set("use_lr_scheduler", True)
    _maybe_set("warmup_epochs", 5)
    _maybe_set("min_lr_ratio", 0.3)
    # Ensure patience is larger than warmup_epochs; otherwise early stopping
    # can fire before warmup ends (observed on FB_CKGE snapshots 2/3/4 where
    # MRR stayed flat during the first warmup epochs).
    if "patience" not in _cli:
        args.patience = max(int(args.patience), int(args.warmup_epochs) + 3)