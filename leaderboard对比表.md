# C2G 参数高尔夫 — BPB 对比表

## 榜单演进（官方 records/track_10min_16mb/）

| 日期 | 方案名称 | BPB | Δ vs 前一条 | 关键改动 |
|------|----------|-----|------------|----------|
| 03-17 | Naive Baseline | 1.2244 | — | SP1024, 9L×512d, KV4, AdamW |
| 03-17 | LoRA TTT | 1.1929 | −0.032 | LoRA + Test-Time Training |
| 03-18 | FP16Embed + WD Tuning | 1.2197 | +0.027 | FP16 tied embed + LR/warmdown 调参 |
| 03-18 | Long Context Seq2048 | 1.2058 | −0.014 | 序列长度 1024→2048 |
| 03-18 | Lower LR | 1.2230 | +0.017 | 学习率降低 |
| 03-19 | 10L Mixed Precision | 1.2147 | −0.008 | 10层 + 混合精度 |
| 03-19 | Sliding Window Eval | 1.1925 | −0.022 | 滑窗评估 stride=64 |
| 03-19 | MLP3x QAT Int6 SW | 1.1502 | −0.042 | int6 量化 + 3×MLP + 滑窗 |
| 03-19 | Mixed Quant Int6+Int8 SW | 1.1630 | +0.013 | 混合量化 |
| 03-19 | Warmdown Quantization | 1.1574 | −0.006 | warmdown 量化策略 |
| 03-19 | SmearGate + Muon WD | 1.1556 | −0.002 | SmearGate + Muon 优化器 |
| 03-19 | Training Opt Seq4096 | 1.2014 | +0.046 | 序列长度优化 |
| 03-19 | 10L Muon WD Overtone | — | — | Muon + overtone init |
| 03-20 | Int6 MLP3x BigramHash MuonWD SWA | 1.1458 | −0.010 | BigramHash + Muon + SWA |
| 03-20 | 10L Int5MLP MuonWD SWA | — | — | int5 MLP |
| 03-20 | 11L XSA4 EMA Int6 MLP3x | 1.1271 | −0.019 | 11层 + XSA + EMA |
| 03-20 | 11L Efficient Partial XSA FA3 SWA | 1.1307 | +0.004 | 部分 XSA + FlashAttn3 |
| 03-21 | 11L XSA4 EMA PartialRoPE LateQAT | 1.1248 | −0.002 | Partial RoPE + Late QAT |
| 03-22 | 11L EMA GPTQ-lite warmdown QAT | 1.1233 | −0.002 | GPTQ-lite + warmdown |
| 03-23 | LeakyReLU² + LegalTTT + ParallelMuon | 1.1194 | −0.004 | LeakyReLU² + TTT + Parallel Muon |
| 03-25 | ValCalib GPTQ XSA BigramHash3072 | 1.1147 | −0.005 | GPTQ 校准 + BigramHash |
| 03-31 | ParallelResiduals + MiniDepthRec | 1.1063 | −0.008 | Parallel Residuals + Depth Recurrence |
| 04-01 | Vocab4096 + MLPMult4 + WD085 | 1.0978 | −0.009 | SP4096 + 更大 MLP |
| 04-03 | MuonEqR + DepthRec + WD090 + AllInt6 | 1.0912 | −0.007 | MuonEq-R + 全 int6 |
| 04-04 | SP4096 + DepthRec + ParResid + MuonEqR | 1.0897 | −0.002 | SP4096 组合方案 |
| 04-05 | SP8192 + GPTQ-Emb + SDClip + Loop | 1.0856 | −0.004 | SP8192 首次出现 |
| 04-06 | SP8192 + HessianSDClip + ProgRec | 1.0835 | −0.002 | Hessian-aware SDClip |
| 04-06 | SP8192 + QK5 + LegalTTT | 1.0828 | −0.001 | QK-Gain 5.0 |
| 04-08 | SP8192 + ParResid + ScoreFirstTTT | 1.0822 | −0.001 | Score-First TTT |
| **04-09** | **SP8192 + 3LayerRecur + ParResid + QK5.25 + TTT** | **1.0810** | **−0.001** | **当前榜首** |

---

## 本方案 vs Baseline vs SOTA

| 指标 | Naive Baseline | 本方案（目标） | 当前 SOTA |
|------|---------------|---------------|-----------|
| **BPB** | 1.2244 | < 1.18 (L2) / < 1.12 (L3) | 1.0810 |
| Tokenizer | SP-1024 | SP-8192 | SP-8192 |
| Layers | 9 | 9–11 | 9 (3-layer recur) |
| Model dim | 512 | 512 | 512 |
| KV heads | 4 | 4 | 4 |
| Optimizer | AdamW | Muon + AdamW (hybrid) | Muon variant |
| Quantization | int8 | int6/int8 | int6 + GPTQ SDClip |
| Architecture | Standard | Parallel residuals | ParRes + Depth Recur |
| Evaluation | Standard | Sliding window | Sliding window + TTT |
| Artifact size | 15,863,489 B | < 16,000,000 B | ~15,992,000 B |
| Train time | ~600s | < 600s | < 600s |
| Seeds | 1 | 3 (目标) | 3 |

---

## 改进路径可视化

```
BPB
1.2244 ████████████████████████████████████████████████ Baseline
       │
1.20   │─── Seq2048, LoRA TTT
       │
1.19   │─── Sliding Window Eval
       │
1.15   │─── Int6 量化 + MLP3x
       │
1.12   │─── 11L + XSA + EMA + GPTQ
       │
1.10   │─── Parallel Residuals + Depth Recurrence
       │         ┌─── SP4096
1.09   │─────────┤
       │         └─── MuonEqR + AllInt6
       │
1.085  │─── SP8192 首次
       │
1.081  │─── SP8192 + 3LayerRecur + ParResid + QK5.25 + TTT  ← SOTA
       │
1.0    │
       └────────────────────────────────────────────────────
       03-17                              04-09
```

---

## 统计说明

- 当前 SOTA 的 3-seed 标准差：σ = 0.00020（seed 42: 1.08079, seed 314: 1.08103, seed 999: 1.08118）
- Level 3 门槛要求：比 SOTA 低 ≥ 0.005 nats，p < 0.01（至少 3 seed）
- 本方案若达到 Level 2（< 1.18），与 baseline 差距为 −0.044，远超显著性阈值
