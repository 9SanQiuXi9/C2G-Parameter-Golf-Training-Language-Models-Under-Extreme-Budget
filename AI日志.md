# C2G 参数高尔夫 — AI 使用日志

## 概述

本日志记录了 C2G 参数高尔夫挑战开发全过程中的 AI 使用情况。所有代码编写、架构分析、实验设计、文档撰写均通过人-AI 协作完成。

**AI 工具：** Claude (Anthropic) / GPT-4 (OpenAI)
**使用模式：** 多轮迭代对话，人工决策 + AI 执行

---

## Round 0：项目理解与代码审查

**时间：** 2026-10-05
**目标：** 理解 parameter-golf 官方 repo 的代码结构和基线实现

### Prompt（人工）
```
请逐段分析 train_gpt.py 的代码结构，重点关注：
1. 数据加载和 tokenization 流程
2. Transformer block 的具体实现（attention、MLP、norm 位置）
3. 训练循环（optimizer、lr schedule、grad clip）
4. 量化和 artifact 打包流程
5. 评估函数（BPB 计算方式）
对每个模块，说明它对最终 BPB 的可能影响。
```

### AI 输出摘要
- 识别出 baseline 架构：9L×512d, KV4, tied embeddings, SP-1024
- 数据流：FineWeb → SentencePiece tokenize → pack into 1024-len sequences → batch 524288 tokens/step
- Attention：标准 GQA (8 heads, 4 KV heads)，RoPE 位置编码
- MLP：2-layer MLP with mult=2, GELU activation
- Norm：RMSNorm，pre-norm 位置
- 量化：训练后 int8 round-to-nearest + zlib 压缩
- BPB 计算：`val_loss * log2(e) / log2(256)` = cross-entropy 转换为 bits-per-byte

### 人工决策
- 确认 tokenizer 升级是第一步方向（改动最小、收益最确定）
- 要求 AI 列出所有 33 条历史提交的 BPB 排名

### AI 踩坑记录
- AI 最初将 BPB 公式写为 `val_loss / ln(256)`，遗漏了 `log2(e)` 的转换因子。经人工核对官方 eval_bpb.py 后纠正为 `val_loss * log2(e) / 8`。

---

## Round 1：榜单数据分析

**时间：** 2026-10-05
**目标：** 从 33 条历史提交中提取各组件的独立贡献

### Prompt（人工）
```
读取 records/track_10min_16mb/ 下所有 submission.json，按时间排序。
对每对相邻提交，比较 technique_summary 的差异，记录 BPB 变化。
生成一张组件-收益对照表。
注意：很多提交同时改了多个东西，需要标注"无法隔离"的情况。
```

### AI 输出摘要
- 生成了完整的 33 条提交排名表
- 识别出 7 个主要组件：SP8192、Muon、ParRes、DepthRec、QK-Gain、SW Eval、TTT
- 标注了哪些对比是"较干净的单变量对照"，哪些是"多变量混合"

### 人工决策
- 确认保守估计策略：只取可隔离对照中的最小收益值
- 决定在消融报告中明确标注置信度（高/中/低）

### 迭代过程
- 第 1 轮：AI 给出的组件收益估计偏高（把多变量的改善全归因于单组件）
- 第 2 轮：人工指出问题，要求 AI 只取"其他条件相同、仅一个变量不同"的对比
- 第 3 轮：AI 修正后给出保守估计，与人工交叉验证后确认

---

## Round 2：SP8192 Tokenizer 实现设计

**时间：** 2026-10-05
**目标：** 设计将 baseline tokenizer 从 SP-1024 升级到 SP-8192 的代码改动方案

### Prompt（人工）
```
基于 train_gpt.py 的当前实现，设计 SP8192 的替换方案。
需要回答：
1. 哪些代码段需要修改？
2. Embedding 矩阵维度变化对 artifact 大小的影响？
3. 是否需要配合量化来控制 artifact < 16MB？
4. 数据预处理管线的改动？
给出具体的代码 diff 方案。
```

### AI 输出摘要
- 需修改 5 处：tokenizer 加载、vocab_size 常量、embedding 层定义、data loader 的 token 序列、量化目标
- Embedding 矩阵：1024×512×2B = 1MB → 8192×512×2B = 8MB（bf16）
- 8MB embedding 已占 artifact 的 50%，必须量化到 int8（4MB）或 int6（3MB）
- 数据管线：需要重新 tokenize FineWeb 训练集（SP-8192 model），但评估集是预计算的

### 人工决策
- 选择 int8 量化 embedding（比 int6 实现简单，精度损失可接受）
- 决定先实现代码，等拿到 GPU 后再跑实验验证

### AI 踩坑记录
- AI 最初建议用 GPTQ 量化 embedding，但 GPTQ 需要 Hessian 矩阵计算，实现复杂度高。人工决定先用简单的 round-to-nearest int8，降低实现风险。

---

## Round 3：Muon 优化器接入

**时间：** 2026-10-05
**目标：** 设计 Muon 优化器的接入方案

### Prompt（人工）
```
阅读 Muon 优化器的原始实现（github.com/KellerJordan/Muon），
设计在 parameter-golf train_gpt.py 中的接入方案。
注意：
1. Muon 只适用于 2D 权重矩阵，embedding 和 1D 参数仍用 AdamW
2. Newton-Schulz 迭代次数（默认 6）对速度的影响
3. 学习率和 weight decay 的推荐值
4. 是否需要在 10 分钟约束内调整 batch size 或步数？
```

### AI 输出摘要
- Muon 适用于：attention 的 QKV/O 投影矩阵、MLP 的 up/down 投影矩阵
- AdamW 适用于：embedding、layer norm 参数、tied output head
- 推荐超参：lr=0.02, wd=0.04, momentum=0.95, NS iterations=6
- 每步额外开销：~15-30%（Newton-Schulz 迭代），需要监控 wallclock

### 人工决策
- 采用 hybrid optimizer 方案（Muon for 2D, AdamW for 1D/embedding）
- 预留 20% wallclock buffer：如果 NS=6 超时，降级到 NS=4

### 迭代过程
- 第 1 轮：AI 给出的 Muon 实现直接对所有参数使用，包括 embedding
- 第 2 轮：人工指出 embedding 是 2D 矩阵但不应使用 Muon（因为 tied embedding 的特殊性），要求 AI 修改为只对 attention+MLP 的 2D 权重使用 Muon
- 第 3 轮：确认最终方案

---

## Round 4：实验矩阵设计

**时间：** 2026-10-05
**目标：** 设计完整的实验矩阵和消融方案

### Prompt（人工）
```
基于前面的分析，设计完整的实验矩阵。要求：
1. 每个实验有明确的假设
2. 包含消融对照（leave-one-out）
3. 每个配置至少 3 个 seed
4. 估算总实验次数和算力预算
5. 明确每个实验的判据（成功/失败/需要调整的标准）
```

### AI 输出摘要
- 设计了 4 Phase 共 12 个实验配置
- Phase 0: baseline 复现 (1 run)
- Phase 1: SP8192 (3 seeds × 2 configs = 6 runs)
- Phase 2: +Muon (3 seeds × 2 configs = 6 runs)
- Phase 3: +SW eval (3 seeds × 1 config = 3 runs)
- Phase 4: 最终提交 (3 seeds = 3 runs)
- 总计：约 19 runs，约 3.2 小时 8×H100

### 人工决策
- 精简到 12 runs（去掉部分冗余配置）
- 确认成功判据：每阶段 BPB 必须比上一阶段低 ≥ 0.005

---

## Round 5：文档撰写

**时间：** 2026-10-05
**目标：** 撰写方案草案、方案设计、消融报告等文档

### Prompt（人工）
```
基于前面所有分析和决策，撰写方案草案。
要求：
1. ≥500 字
2. 回答 4 个门槛问题
3. 引用具体的榜单数据作为证据
4. 诚实标注当前无 GPU 的状态
5. 不要用空洞的承诺，每个预期收益都要有数据来源
```

### 迭代过程
- 第 1 轮：AI 生成的草案引用了论文中的理论值
- 第 2 轮：人工要求全部改用榜单实际数据，删除论文引用
- 第 3 轮：AI 修正后，人工审核通过

---

## Round 6：代码审查与自检

**时间：** 2026-10-05
**目标：** 审查训练代码的正确性

### Prompt（人工）
```
审查修改后的 train_gpt.py，检查：
1. Muon 优化器的 Newton-Schulz 迭代实现是否正确
2. int8 量化的 round-to-nearest 是否有 off-by-one 错误
3. sliding window eval 的 stride 计算是否正确
4. artifact 打包时是否正确计算了总大小
5. BPB 计算公式是否与官方 eval 一致
```

### AI 输出摘要
- 发现 2 个问题：
  1. Muon 的 NS 迭代中缺少正交化后的 scale factor
  2. SW eval 的最后一个 window 可能越界
- 其余检查通过

### 人工决策
- 修复上述 2 个问题
- 增加恒等式自检（量化前后矩阵 Frobenius 范数差异 < 阈值）

---

## AI 使用统计

| 维度 | 数值 |
|------|------|
| 总对话轮次 | 7 (Round 0-6) |
| 人工发起的纠错 | 5 次 |
| AI 自主发现的问题 | 2 次 |
| 被人工拒绝的 AI 建议 | 3 次（GPTQ 量化、全参数 Muon、论文引用） |
| 采纳的 AI 建议 | 12 次 |
| 采纳/拒绝比 | 80% / 20% |

---

## 关键决策中的人工 vs AI 分工

| 决策 | 人工角色 | AI 角色 |
|------|----------|---------|
| 选择 SP8192 方向 | **最终决策** | 提供数据分析和选项 |
| 保守估计策略 | **制定原则** | 执行计算 |
| int8 vs GPTQ | **最终决策** | 提供两种方案对比 |
| 实验矩阵设计 | 审核 + 精简 | 生成初稿 |
| 代码审查 | 发现方向性问题 | 检查实现细节 |
| 文档撰写 | 定框架 + 审核 | 生成文本 + 数据填充 |
