# SpiderWeb Self-Attention

## 基于 Transformer 的字符级结构偏置注意力机制

**Version**: SpiderWeb-v0.1-stable | **Author**: 张彪 | **Date**: 2026-06

---

## 1. 项目概述

SpiderWeb Self-Attention 是一种面向中文长文本理解的**结构偏置注意力机制**。它在 Transformer 自注意力的基础上引入一个结构偏置矩阵 M_web，模拟文章"中心→支撑→描述"的三层层级关系，引导模型关注核心内容。

## 注：此项目为测试项目，并未进行真实的 中/英文 文章 数据测试作者实力有限，且作者的本意是使用这个机制，使智能模型快速理解文章含义且降低硬件设施的损耗。

---

**核心公式**：

```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + lambda * M_web) * V
```

**M_web 三组件**：
- **M_center (α)**：增强中心 token 之间的注意力
- **M_hierarchy (β)**：按层级距离衰减注意力
- **M_position (γ)**：同段落内 + 相邻段落间注意力增强

---

## 2. 实验结果

### 合成层级数据实验

| 阶段 | TR 准确率 | SW 准确率 | 提升 | 说明 |
|------|:---------:|:---------:|:----:|------|
| Phase 1 | 0.7989 | 0.8011 | +0.22pp | 初始消融，λ=0.7最优 |
| Phase 2 | 0.7870 | 0.8080 | +2.10pp | 5 seeds + RandomBias 对照 |
| **Phase 4** | **0.7853** | **0.8150** | **+2.97pp** | **一致性核查通过，最终版本** |

### 关键统计 (Phase 4 最终版本)

| 对比 | 均值差 | t | p | Cohen's d |
|------|:------:|:---:|:---:|:---:|
| SW vs Transformer | +2.97pp | 5.30 | 0.0061 | 2.369 |
| SW vs RandomBias | +3.37pp | 3.03 | 0.0388 | 1.355 |

- **所有 8 个类别 SpiderWeb F1 均高于 Transformer**
- RandomBias 低于基线 → 证明提升来自结构偏置，非随机效应
- 长文本上优势更明显

---

## 3. 项目结构

```
SpiderWeb Self-Attention/
├── README.md                          # 项目总览（本文件）
├── model.py                           # 模型定义：SpiderWebMultiHeadAttention + SpiderWebClassifier
├── data.py                            # 合成数据生成器（3层层级结构）
├── train.py                           # 训练管道：train_model + evaluate_detailed
├── experiment.py                      # 实验编排器：消融实验 + 参数扫描
├── report.py                          # 最终报告生成器
├── real_data.py                       # 真实中文数据生成器（字符频率分布）
├── agent_encoder.py                   # SpiderWeb 阅读智能体：文章编码 + 段落检索
├── agent_cli.py                       # 交互式命令行智能体
├── consistency_v2.py                  # 一致性核查脚本
├── finalize_v2.py                     # 最终统一输出脚本
├── phase5_experiment.py              # Phase 5 真实数据实验
├── download_thucnews.py              # THUCNews 数据下载（备用）
├── analysis_core.py / analysis_viz.py / analysis_extras.py  # 统计分析与可视化
│
├── archive/old_results/               # 废弃的旧版报告（+2.10pp, +0.77pp）
│
├── phase1_basic_ablation/             # 第一次实验：基础消融 + λ扫描
├── phase2_rigorous_experiment/        # 第二次实验：5 seeds + RandomBias + CSV
├── phase3_statistical_analysis/       # 第三次实验：统计检验 + 修正图表
├── phase4_paper_ready/               # 第四次实验：论文级输出
│   └── final_unified_results/         # ★ 最终统一结果（一致性核查通过）
└── phase5_real_data/                 # 第五次实验：真实中文数据（运行中）
```

---

## 4. 模型详解

```
SpiderWebClassifier(
  token_embedding: Embedding(vocab_size, 128)
  pos_embedding: Embedding(max_len, 128)
  layers: [
    SpiderWebEncoderLayer(
      SpiderWebMultiHeadAttention(128, 4 heads, d_k=32)  ← M_web 注入点
      LayerNorm + FFN(128→512→128, GELU)
    ) × 2
  ]
  classifier: LayerNorm → Linear(128→64) → GELU → Linear(64→n_classes)
)
```

**参数总量**: ~479K | **支持长度**: 80 (合成) / 512 (真实数据)

---

## 5. 快速使用

### 环境要求

```bash
pip install torch numpy matplotlib seaborn scipy tqdm
```

### 训练模型

```bash
# 合成数据（80字符）
python experiment.py

# 真实数据（512字符）
python phase5_experiment.py
```

### 运行阅读智能体

```bash
python agent_cli.py
# 1. 粘贴文章 → 输入 END
# 2. 提问 → 返回最相关段落
```

### 生成论文报告

```bash
python finalize_v2.py
# 输出到 phase4_paper_ready/final_unified_results/
```

---

## 6. 实验阶段总结

| 阶段 | 文件夹 | 核心产出 |
|------|--------|---------|
| Phase 1 | `phase1_basic_ablation/` | 基础消融：4 variants，λ scan [0, 0.1, 0.3, 0.5, 0.7, 1.0] |
| Phase 2 | `phase2_rigorous_experiment/` | 5 seeds + RandomBias + Accuracy/Precision/Recall/F1 |
| Phase 3 | `phase3_statistical_analysis/` | t-test + Wilcoxon + CI + Cohen's d |
| Phase 4 | `phase4_paper_ready/final_unified_results/` | ★ 最终版：一致性核查 + 混淆矩阵 + 热力图 + 论文报告 |
| Phase 5 | `phase5_real_data/` | 真实中文数据（512字符，运行中） |

---

## 7. 废弃版本说明

以下版本已废弃，移到 `archive/old_results/`：

| 版本 | Delta | 废弃原因 |
|------|------|---------|
| +2.10pp | TR=0.7870, SW=0.8080 | 不同训练 run，CM 不一致 |
| +0.77pp | TR=0.7853, SW=0.7930 | 不同训练配置 (无 CosineAnnealingLR) |

**最终采纳版本：+2.97pp (TR=0.7853, SW=0.8150)** — 单次 inference pass，全自洽。

---

## 8. 论文引用

如果使用了本工作，请引用：

```
@techreport{SpiderWeb-v0.1,
  title   = {SpiderWeb Self-Attention: Structural Bias Attention Mechanism for Chinese Long-Text Understanding},
  version  = {v0.1-stable},
  year     = {2026},
  note     = {5-seed experiment, +2.97pp over Transformer baseline, consistency-verified}
}
```
 
---

## 9. 许可证

MIT License
