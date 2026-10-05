# C2G 参数高尔夫 — 项目交付包

## 项目信息

| 字段 | 值 |
|------|-----|
| 挑战 | C2G 参数高尔夫 — 极限约束下的语言模型训练 |
| 目标等级 | Level 2（BPB < 1.18） |
| 赛道 | track_10min_16mb |
| 基线 BPB | 1.2244 (Naive Baseline) |
| 当前 SOTA | 1.0810 (SP8192 + 3LayerRecur + ParResid + QK5.25 + TTT) |

---

## 交付物清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `方案草案.md` | 算力申请用方案草案（≥500字，4个门槛问题） | ✓ 完成 |
| `方案设计.md` | 技术选型、实验矩阵、16MB 预算分析 | ✓ 完成 |
| `消融实验报告.md` | 基于 33 条官方榜单记录的组件收益分析 | ✓ 完成 |
| `leaderboard对比表.md` | 完整榜单演进 + 本方案 vs baseline vs SOTA 对比 | ✓ 完成 |
| `AI日志.md` | 7 轮人-AI 协作记录（Round 0-6） | ✓ 完成 |
| `拿来说明.md` | 代码/思路溯源（nanoGPT、官方 repo、Muon、历史方案） | ✓ 完成 |
| `AAR复盘.md` | 七维复盘 + 失败经验 + 下一步计划 | ✓ 完成 |
| `train_gpt.py` | 改进版训练脚本（SP8192 + SW eval + QK-Gain 5.0） | ✓ 完成 |
| `README.md` | 本文件 | ✓ 完成 |

---

## 改进方案概述

基于对 OpenAI parameter-golf 官方 repo 中 33 条历史提交的系统分析，本方案采用渐进式优化策略：

### 核心改动

| 改动 | 基线值 | 改进值 | 预期收益 |
|------|--------|--------|----------|
| Tokenizer | SP-1024 | SP-8192 | −0.03 ~ −0.05 BPB |
| QK-Gain | 1.5 | 5.0 | −0.005 ~ −0.008 BPB |
| 评估策略 | 标准 | Sliding Window (stride=64) | −0.005 ~ −0.01 BPB |
| 优化器 | AdamW (已内置 Muon) | Muon hybrid (已内置) | −0.004 ~ −0.015 BPB |

### 预期 BPB

- **保守估计：** 1.16 ~ 1.18（Level 2）
- **乐观估计：** 1.12 ~ 1.14（Level 3 区间）

---

## 代码改动说明

`train_gpt.py` 基于官方 NaiveBaseline 版本，改动如下：

| 改动 | 位置 | 说明 |
|------|------|------|
| 默认 vocab_size | Hyperparameters 类 | 1024 → 8192 |
| 默认 data_path | Hyperparameters 类 | fineweb10B_sp1024 → fineweb10B_sp8192 |
| 默认 tokenizer_path | Hyperparameters 类 | fineweb_1024_bpe.model → fineweb_8192_sp.model |
| QK-Gain 默认值 | Hyperparameters 类 | 1.5 → 5.0 |
| eval_stride 参数 | Hyperparameters 类 | 新增，默认 64 |
| eval_val 函数 | 评估模块 | 支持 sliding window 评估（stride < seq_len 时启用） |

其余代码（Muon 优化器、int8 量化、数据加载、Transformer 架构）保持与官方 baseline 一致。

---

## 运行方式

### 环境要求

```bash
pip install torch>=2.3 sentencepiece numpy tqdm
pip install flash-attn --no-build-isolation
```

### Baseline 复现（SP-1024）

```bash
# 使用官方默认参数
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
QK_GAIN_INIT=1.5 \
EVAL_STRIDE=1024 \
torchrun --nproc_per_node=8 train_gpt.py
```

### 改进方案（SP-8192 + SW Eval）

```bash
# 需要先准备 SP-8192 数据
python data/download_hf_docs_and_tokenize.py --vocab_size 8192

# 运行训练
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_sp.model \
VOCAB_SIZE=8192 \
QK_GAIN_INIT=5.0 \
EVAL_STRIDE=64 \
torchrun --nproc_per_node=8 train_gpt.py
```

---

## 实验状态

| 字段 | 值 |
|------|-----|
| local_bpb_measured | not_measured |
| local_hardware | 无 GPU（CPU only） |
| official_leaderboard | false |
| track | local_analysis_only |
| seeds_run | 0 |
| artifact_built | not_measured |

**说明：** 当前环境无 8×H100 或同等 GPU 资源。所有 BPB 预期值均来自官方榜单历史数据分析，非本地实测。方案设计和实验矩阵已准备就绪，待获取算力后可立即执行。

---

## 文件命名规范

按挑战要求，所有文件应命名为 `姓名拼音_C2G_内容描述.扩展名` 格式。当前交付包使用通用名称，提交前需批量重命名：

```
方案草案.md           → [姓名拼音]_C2G_方案草案.md
方案设计.md           → [姓名拼音]_C2G_方案设计.md
消融实验报告.md       → [姓名拼音]_C2G_消融实验报告.md
leaderboard对比表.md  → [姓名拼音]_C2G_leaderboard对比表.md
AI日志.md             → [姓名拼音]_C2G_AI日志.md
拿来说明.md           → [姓名拼音]_C2G_拿来说明.md
AAR复盘.md            → [姓名拼音]_C2G_AAR复盘.md
train_gpt.py          → [姓名拼音]_C2G_train_gpt.py
README.md             → [姓名拼音]_C2G_README.md
submission.json       → [姓名拼音]_C2G_submission.json
```

---

## 参考资源

- [openai/parameter-golf](https://github.com/openai/parameter-golf) — 官方 repo
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — 基线代码祖先
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — Muon 优化器
- [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb) — 训练/评估数据
