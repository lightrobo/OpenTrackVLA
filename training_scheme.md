# OpenTrackVLA 训练方案

## 核心设计：直接回归，不需要 Tokenization

OpenTrackVLA **不需要将连续的 (x, y, θ) 离散化或 tokenize**。

模型使用 LLM 作为**特征编码器**，而非自回归生成器：
- 输入：视觉 embedding + 文本 embedding（全是连续向量）
- 输出：从 [ACT] token 位置提取隐状态，送入 MLP 直接回归

---

## 训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                       数据准备                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   JSON/JSONL 数据集包含:                                        │
│   • images: 历史帧路径列表 (31帧)                               │
│   • current: 当前帧路径                                         │
│   • waypoints: GT轨迹 [(x,y,θ), ...] × 8                       │
│   • instruction: 文本指令                                       │
│   • valid_mask: 有效 waypoint 掩码                              │
│                                                                 │
│   预处理:                                                       │
│   • precache_frames.py 预计算 DINO+SigLIP 特征                  │
│   • 粗粒度: 4 tokens/帧, 细粒度: 64 tokens/帧                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       前向传播                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 视觉特征投影                                               │
│      vis_c = Projector(coarse_tokens)  # (B, H×4, D)            │
│      vis_f = Projector(fine_tokens)    # (B, 64, D)             │
│                                                                 │
│   2. 时间标记插入 (TVI)                                         │
│      vis_c = interleave_tvi(vis_c, time_idx, kind=0)            │
│      vis_f = interleave_tvi(vis_f, time_idx, kind=1)            │
│                                                                 │
│   3. 序列拼接                                                   │
│      seq = [文本emb] + [历史粗视觉] + [当前细视觉] + [ACT]      │
│                                                                 │
│   4. LLM 编码 (注意: 使用 inputs_embeds, 不是 input_ids!)       │
│      out = LLM(inputs_embeds=seq)                               │
│      h_act = out.last_hidden_state[:, -1, :]  # [ACT]位置       │
│                                                                 │
│   5. MLP 回归                                                   │
│      a_hat = PlannerHead(h_act)         # (B, 8, 3)             │
│      tau_pred = a_hat * alpha_task      # 缩放到实际单位        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       损失计算                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Loss = β_nav × MSE_masked(pred, gt, valid_mask)               │
│                                                                 │
│   其中:                                                         │
│   • β_nav = 10.0 (导航损失权重)                                 │
│   • MSE_masked: 仅在 valid_mask=True 的 waypoint 上计算         │
│                                                                 │
│   def mse_masked(pred, target, mask):                           │
│       se = (pred - target).pow(2)                               │
│       se = se[mask.expand_as(se)]                               │
│       return se.mean()                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       反向传播                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   可训练参数:                                                   │
│   • CrossModalityProjector (视觉投影器)                         │
│   • TVIEmbedder (时间-视角嵌入)                                 │
│   • act_token (动作查询 token)                                  │
│   • PlannerHead3L (规划 MLP)                                    │
│   • LLM (可选: freeze_llm=False 时微调)                         │
│                                                                 │
│   优化器: AdamW                                                 │
│   • lr = 2e-5 ~ 3e-4                                            │
│   • weight_decay = 0.01                                         │
│   • grad_clip = 1.0                                             │
│                                                                 │
│   混合精度: bfloat16 (可选)                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 为什么不需要 Tokenize (x, y, θ)?

### 传统 LLM 生成方式 vs 本模型方式

| 方面 | 传统自回归 LLM | OpenTrackVLA |
|------|---------------|--------------|
| 输入 | `input_ids` (离散 token) | `inputs_embeds` (连续向量) |
| 输出 | 逐 token 生成 | 单次 MLP 回归 |
| 动作表示 | 需要离散化/tokenize | 直接连续值 |
| 推理 | 多次前向 (自回归) | 单次前向 |

### 关键代码

```python
# 不是这样用 LLM:
# tokens = tokenizer.encode(text)
# output_ids = model.generate(tokens)

# 而是这样用:
seq = torch.cat([txt_emb, vis_c, vis_f, act_token], dim=1)  # 全是 embedding
out = self.llm(inputs_embeds=seq)  # 直接传入 embedding
h_act = out.last_hidden_state[:, -1, :]  # 取最后位置的隐状态
waypoints = self.planner(h_act)  # MLP 输出连续值
```

---

## 数据归一化策略

### alpha_xy 缩放

为了让 MSE loss 在合理范围内，对 x, y 维度进行缩放：

```
alpha_xy = 2.0  (默认值，或从数据集 95 百分位自动计算)

训练时:
  pred_norm[..., 0:2] = pred[..., 0:2] / alpha_xy
  gt_norm[..., 0:2] = gt[..., 0:2] / alpha_xy
  loss = MSE(pred_norm, gt_norm)

推理时:
  raw_output = planner(h_act)  # [-1, 1] (tanh 限制)
  waypoints = raw_output * alpha_task  # 缩放回实际单位
```

### tanh 输出限制

```python
class PlannerHead3L(nn.Module):
    def forward(self, act_h):
        y = self.mlp(act_h)
        if self.use_tanh:
            y = torch.tanh(y)  # 限制到 [-1, 1]
        return y.view(-1, self.nw, self.ad)
```

---

## 训练配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llm_name` | Qwen/Qwen3-0.6B | LLM 骨干 |
| `freeze_llm` | False | 是否冻结 LLM |
| `n_waypoints` | 8 | 预测 waypoint 数量 |
| `history` | 31 | 历史帧数量 |
| `batch_size` | 2~12 | 批大小 |
| `lr` | 2e-5 ~ 3e-4 | 学习率 |
| `epochs` | 1~5 | 训练轮数 |
| `beta_nav` | 10.0 | 导航损失权重 |
| `alpha_xy` | 2.0 | XY 缩放系数 |
| `grad_clip` | 1.0 | 梯度裁剪 |
| `mixed_precision` | True | 混合精度训练 |

---

## 评估指标

训练过程中监控：

| 指标 | 说明 |
|------|------|
| `L_nav` | 归一化空间的 MSE 损失 |
| `mse_total` | 绝对空间的 MSE |
| `per_dim_mse` | 各维度 (x, y, θ) 分别的 MSE |
| `mask_cov` | valid_mask 覆盖率 |
| `final_EPE` | 最后一个 waypoint 的端点误差 |
| `hit_rate` | EPE < threshold 的比例 |

---

## 与 Diffusion Policy 的对比

| 方面 | OpenTrackVLA (MLP 回归) | Diffusion Policy |
|------|------------------------|------------------|
| 输出分布 | 单点预测 | 可建模多峰分布 |
| 训练复杂度 | 简单 (MSE) | 复杂 (去噪过程) |
| 推理速度 | 快 (单次前向) | 慢 (多步去噪) |
| 多模态动作 | ❌ 会平均化 | ✅ 可采样不同模式 |
| 实现难度 | 低 | 高 |

### 何时考虑改用 Diffusion?

1. 任务存在多种合理轨迹（如绕障碍左/右都行）
2. 需要采样多样化的动作序列
3. 当前 MLP 回归出现 "regression to the mean" 问题

---

## 训练命令示例

```bash
python train.py \
    --train_json data/sample/jsonl \
    --cache_root data/sample/vision_cache \
    --out_dir ckpt_sample \
    --epochs 2 \
    --batch_size 8 \
    --n_waypoints 8 \
    --history 31 \
    --lr 2e-5 \
    --mixed_precision \
    --save_trajectories
```

---

## LLM Backbone 训练策略

### 默认配置：全量微调 (Full Fine-tuning)

```python
# train.py 第 269 行
@dataclass
class ModelConfig:
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = False  # ← 训练时默认不冻结

# train.py 第 284 行
self.llm.requires_grad_(not cfg.freeze_llm)  # False → 全部可训练
```

### 训练策略对比

| 问题 | 答案 |
|------|------|
| LLM 冻结吗？ | **默认不冻结** (`freeze_llm=False`) |
| 使用 LoRA 吗？ | **没有**，代码中无任何 peft/lora |
| 部分冻结？ | **没有**，只有全冻结或全解冻两种模式 |
| 训练哪些参数？ | **全部参数** (LLM + Projector + TVI + Planner) |

### 两种模式

| 配置 | 可训练参数 | 用途 |
|------|-----------|------|
| `freeze_llm=False` (train.py 默认) | LLM + Projector + TVI + Planner | 训练时全量微调 |
| `freeze_llm=True` (model.py 默认) | Projector + TVI + Planner | 推理时冻结，或轻量训练 |

### 参数统计日志

训练开始时会打印参数统计：

```python
# train.py 第 837-840 行
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[PARAMS] total={total_params:,} trainable={trainable_params:,} ({pct:.2f}%)")
```

当 `freeze_llm=False` 时，输出类似：
```
[PARAMS] total=600,000,000 trainable=600,000,000 (100.00%)
[PARAMS groups] llm:600000000/600000000 proj:xxx/xxx tvi:xxx/xxx planner:xxx/xxx
```

### 注意：不同文件默认值不同

| 文件 | 默认值 | 用途 |
|------|--------|------|
| `train.py` | `freeze_llm=False` | 训练脚本，全量微调 |
| `model.py` | `freeze_llm=True` | 推理脚本，冻结 LLM |
| `open_trackvla_hf/` | `freeze_llm=True` | HuggingFace 格式，冻结 LLM |

### 如需添加 LoRA 支持

当前代码**不支持 LoRA**，需要手动添加：

```python
from peft import LoraConfig, get_peft_model

# 在 OpenTrackVLA.__init__ 中添加
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)
self.llm = get_peft_model(self.llm, lora_config)
```

这样可以大幅减少可训练参数量（约 0.1%~1%），同时保持较好的性能。

---

## 总结

1. **不需要 tokenize 连续值** — LLM 接收 `inputs_embeds`，不是 `input_ids`
2. **LLM 是特征编码器** — 不做自回归生成，只提取 [ACT] 位置的隐状态
3. **MLP 直接回归** — 输出 (x, y, θ) × 8 个 waypoints
4. **Loss 是 MSE** — 简单的均方误差，带有效 waypoint 掩码
5. **训练简单稳定** — 不需要复杂的 diffusion scheduler 或采样策略
6. **LLM 默认全量微调** — `freeze_llm=False`，无 LoRA，无部分冻结

