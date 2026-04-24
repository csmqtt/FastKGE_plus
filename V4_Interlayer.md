# V4 层间协作（Interlayer）工作说明与阶段成果

本文档单独记录以**实体侧层间协作**为主线的第四版工作（简称 **V4_Interlayer**）：实现要点、实验设计、已有日志结论、与已搁置方向的边界说明。便于开题、中期或论文方法章节直接引用。

---

## 1. 目标与定位

在**不依赖图先验**的前提下，在持续快照知识图谱嵌入训练中，为实体侧多段 `LoRA` 适配器增加**层间信息传递**，使各层输出在几何上更一致、快照迁移更稳，并系统观察对**排序指标**（如 `Whole_MRR`）与**训练时间**的影响。

**当前主线**：仅保留并深化**层间协作模块**；难度感知分层、量化残差奇异值分解初始化等方向已从主实验矩阵中撤出（代码中部分仍可保留作历史分支，见第 4 节）。

---

## 2. 实现概要（代码落点）

- **核心文件**：`src/model/LoraKGE_Layers.py`
  - 快照切换后构造多层实体 `LoRA`；在 `TransE.get_lora_embeddings` 中逐层前向，并通过 `_fuse_with_previous_entity_layer` 将**上一层输出的行均值摘要**注入当前层（残差或门控）。
  - `switch_snapshot` 写回主嵌入时使用聚合后的 `get_lora_embeddings()`，避免层间信息在写回时丢失。
  - 层间参数：`interlayer_lora_mode`（`off` / `residual` / `gate`）、`interlayer_stopgrad`、`interlayer_init`；门控模式下另有 `interlayer_ent_projs`（线性投影）。
- **命令行入口**：`src/parse_args.py` 中上述开关；`main.py` 打「结构配置」日志行便于复现。
- **优化器**：`get_lora_plus_optimizer` 中按参数名匹配层索引；若曾启用难度感知学习率缩放，则 `interlayer_ent_projs` 与对应层共用缩放（主线实验建议 `difficulty_lr_scale=0` 以保持对照纯净）。

---

## 3. 层间协作机制（简述）

- **作用范围**：仅**实体侧**、**多个实体 `LoRA` 层**（`num_ent_layers>1` 时生效）；关系侧多层为另一套逻辑，与实体层间协同独立。
- **融合方式**：对上一层完整输出在「行」维求均值得到 `(1, emb_dim)`，再与当前层输出相加（残差标量强度可学）或经线性投影后由门控强度注入。
- **`interlayer_stopgrad`**：`on` 时摘要不回传到上一层，降低跨层耦合与优化震荡；`off` 时梯度可穿过融合项，耦合更强。
- **精度与效率**：前向额外主要是行均值与广播加法，门控模式增加小批量线性层；对总时间的影响更多来自**优化路径与早停轮数**变化，而非单步算子本身量级剧增。

---

## 4. 与已探索、非主线方向的边界

下列内容曾实现或跑过实验，**不作为 V4_Interlayer 正文主结论的依赖条件**：

| 方向 | 说明 |
|------|------|
| 难度感知分层 | `entity_layering=difficulty`、`ent_rank_policy=difficulty` 及难度绑定秩、学习率等；实验上未形成稳定优于原始排序的结论。 |
| 量化残差奇异值分解初始化 | `lora_init=quant_svd` 及 `quant_init_bits` 等；在 `logs/ablation_quant_init` 中 `ENTITY`/`FB_CKGE` 上未带来稳定末快照收益，`ENTITY` 上总训练时间明显增加。主线已不保留该实验矩阵。 |

若论文只写 V4，建议在正文写清：**主贡献为层间协作**；上表可作为「已探索但未采用」或附录一句带过。

---

## 5. 实验脚本与日志路径

### 5.1 层间相关消融（含难度组合的历史脚本）

- **脚本**：`ablation_interlayer.sh`
- **日志根目录**：`logs/ablation_interlayer/`
- **典型子目录**：`interlayer_only`、`difficulty_only`、`interlayer_difficulty`（时间戳子目录内为各数据集 `.log`）。

### 5.2 原始结构对照（论文级基线日志）

- **路径**：`logs/Nowbest/`（例如 `ENTITY` / `HYBRID` / `RELATION` 各一次完整运行的 `*.log`）。
- **用途**：与 `interlayer_only` 等消融在相同指标表（`Report Result`）下对照。

### 5.3 四组「原始 + 层间 + 量化奇异值分解初始化」实验（已不作为主线）

- **脚本**：`ablation_quant_init.sh`
- **日志根目录**：`logs/ablation_quant_init/`
- **子目录**：`original_structure`、`interlayer_only`、`quant_svd_only`、`interlayer_quant_svd`。

**实验卫生提醒**：若多组共用同一 `./checkpoint/<数据集>`，后跑会覆盖先跑的检查点；正式论文表建议每组独立 `-save_path` 或 `-note` 重跑关键对比。

---

## 6. 阶段成果（基于已有日志的量化摘要）

以下数值均来自各日志末尾 **`Report Result` 表的 `Whole_MRR`** 与 **`Sum_Training_Time`**（秒），便于与论文表格一致。

### 6.1 与 Nowbest 原始结构的对照（历史对话中已对齐）

数据集 **ENTITY** / **HYBRID** / **RELATION** 上，`logs/Nowbest/` 中原始配方与 `logs/ablation_interlayer/interlayer_only/` 等对比的结论是：

- **HYBRID**：层间协同与原始在**末快照 `Whole_MRR` 上可持平**，且总训练时间更短。
- **ENTITY**、**RELATION**：末快照上**原始结构略优或互有胜负**，层间协同更多体现为**中间快照或时间上的折中**，而非全面压倒原始。

（具体数字以对应 `*.log` 内 `Report Result` 为准；写论文时请从文件直接拷贝，避免手抄误差。）

### 6.2 `logs/ablation_quant_init`（ENTITY 与 FB_CKGE，四组完整）

**ENTITY**

| 组别 | 末快照 Whole_MRR | Sum_Training_Time |
|------|------------------|-------------------|
| 原始结构 | **0.233** | 411.29 |
| 仅层间协同 | 0.229 | **337.14** |
| 仅 quant_svd 初始化 | 0.228 | 735.35 |
| 层间 + quant_svd | 0.229 | 829.22 |

**FB_CKGE**

| 组别 | 末快照 Whole_MRR | Sum_Training_Time |
|------|------------------|-------------------|
| 原始结构 | **0.215** | 397.03 |
| 仅层间协同 | **0.215** | **385.23** |
| 仅 quant_svd 初始化 | 0.214 | 440.70 |
| 层间 + quant_svd | 0.214 | 406.43 |

**解读（写入 V4 结论时可采用）**

- **层间协作**：在 `FB_CKGE` 上与原始末指标持平且总时间略优；在 `ENTITY` 上节省时间但末指标略低于本脚本下的「原始结构」组。整体更符合「改变训练动力学与效率，末指标未必单调上升」。
- **量化残差奇异值分解初始化**：本批结果不支持其作为主线；V4 文档将其标记为**已搁置**。

---

## 7. 后续工作建议（仅层间主线）

1. **主表固定为两组或三组**：原始结构；仅层间（`residual`）；可选门控（`gate`）或 `stopgrad` 开闭小消融。
2. **每组独立检查点路径**，保证可复现与公平对比。
3. **统一非结构变量**：同一数据集上 `lora_init`、调度器、`patience` 等与层间无关的开关保持一致，只改 `interlayer_*`。
4. **报告维度**：除末快照 `Whole_MRR` 外，可补充各快照 `Time` 之和、前向迁移等，避免单一指标掩盖「更快收敛」类收益。

---

## 8. 文件与版本说明

- **文档名称**：`V4_Interlayer`（本文件：`V4_Interlayer.md`）
- **内容性质**：工作说明与阶段成果汇总；具体超参与数值以运行时的 `Namespace` 行及 `Report Result` 表为准。
- **更新建议**：每完成一轮正式主表实验，在本文件第 6 节增补数据集、时间戳路径与表格行引用。

---

*文档生成说明：按当前仓库状态与用户决策整理；若删除或下线 `quant_svd` 等代码分支，请同步修订第 4、5.3、6.2 节表述。*
