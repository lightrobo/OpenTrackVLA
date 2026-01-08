# 基于 GridPool 的长短期记忆模块

## 核心设计思想

**用空间分辨率换取时间长度**：历史帧不需要精细细节，当前帧需要高分辨率信息。

这不是什么复杂的 attention 记忆机制，而是最朴素的空间池化 + 滑动窗口队列。

---

## 1. GridPool 实现原理

### 1.1 核心函数

```python
# cache_gridpool.py: 163-180
def grid_pool_tokens(patch_tokens: torch.Tensor, Hp: int, Wp: int, out_tokens: int) -> torch.Tensor:
    """Average-pool patch tokens on a HxW grid down to out_tokens (must be square)."""
    B, P, C = patch_tokens.shape
    s = sqrt(out_tokens)  # 目标网格边长
    # (B, P, C) -> (B, C, Hp, Wp)
    feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, Hp, Wp)
    feat = F.adaptive_avg_pool2d(feat, output_size=(s, s))  # (B, C, s, s)
    pooled = feat.flatten(2).transpose(1, 2).contiguous()    # (B, s*s, C)
    return pooled
```

### 1.2 两种粒度的 Token

| 粒度类型 | Token 数量 | 网格尺寸 | 用途 |
|---------|-----------|---------|------|
| **粗粒度 (Coarse)** | 4 | 2×2 | 历史帧 (长期记忆) |
| **细粒度 (Fine)** | 64 | 8×8 | 当前帧 (短期记忆) |

**特征来源**：DINOv2 + SigLIP 拼接后的特征 (1536维)

```python
# trained_agent.py: 397-399
Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)  # (1, P, C_total)
Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)[0]   # (64, C)
Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)[0]  # (4, C)
```

---

## 2. 长短期记忆结构

### 2.1 数据结构

```python
# trained_agent.py: 202, 207
self.history = 31  # 历史帧数量
self._coarse_hist_tokens: deque = deque(maxlen=self.history)  # 滑动窗口队列
```

### 2.2 记忆组成

```
┌─────────────────────────────────────────────────────────────────────┐
│                        模型输入序列                                  │
├────────────────────────────────────┬────────────────────────────────┤
│        长期记忆 (历史帧)            │       短期记忆 (当前帧)         │
│   31帧 × 4 tokens = 124 tokens     │   1帧 × 64 tokens = 64 tokens  │
│      粗粒度，捕获全局运动           │     细粒度，保留空间细节        │
├────────────────────────────────────┼────────────────────────────────┤
│  t=0  t=1  t=2  ...  t=29  t=30   │           t=31                 │
│  [4]  [4]  [4]  ...  [4]   [4]    │           [64]                 │
└────────────────────────────────────┴────────────────────────────────┘
                                    
总输入: 124 + 64 = 188 visual tokens
```

---

## 3. 连续帧推理流程

### 3.1 单帧处理步骤

```
输入: RGB 图像 (H, W, 3)
           ↓
    ┌──────────────────┐
    │  DINOv2 编码器    │ → tok_dino (1, 576, 1024)
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  SigLIP 编码器    │ → tok_sigl (1, 576, 512)
    └──────────────────┘
           ↓
    特征拼接: Vt_cat (1, 576, 1536)
           ↓
    ┌──────────────────────────────────┐
    │         GridPool                  │
    │  ├── 粗粒度: 576 → 4 tokens      │
    │  └── 细粒度: 576 → 64 tokens     │
    └──────────────────────────────────┘
           ↓
    Vcoarse (4, 1536), Vfine (64, 1536)
```

### 3.2 时间线演示

假设 `history=31`，机器人从第1帧开始连续观察：

---

**T=1 (第一帧)**

```
输入: Frame 1
编码: Vc1(4tok), Vf1(64tok)

队列状态: [Vc1] (长度=1)

组装输入 (左填充):
┌─────────────────────────────────────────────────────────────────────┐
│ 历史 (粗粒度) - 需要31帧，只有1帧，用第一帧填充                       │
│ t=0:  Vc1 ←填充                                                     │
│ t=1:  Vc1 ←填充                                                     │
│  ...                                                                │
│ t=29: Vc1 ←填充                                                     │
│ t=30: Vc1 ←真实数据                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ 当前 (细粒度)                                                        │
│ t=31: Vf1 (64 tokens)                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

**T=5 (第五帧)**

```
输入: Frame 5
编码: Vc5(4tok), Vf5(64tok)

队列状态: [Vc1, Vc2, Vc3, Vc4, Vc5] (长度=5)

组装输入 (左填充):
┌─────────────────────────────────────────────────────────────────────┐
│ 历史 (粗粒度)                                                        │
│ t=0~25:  Vc1 ←填充 (重复26次)                                        │
│ t=26:    Vc1 ←真实                                                  │
│ t=27:    Vc2                                                        │
│ t=28:    Vc3                                                        │
│ t=29:    Vc4                                                        │
│ t=30:    Vc5                                                        │
├─────────────────────────────────────────────────────────────────────┤
│ 当前 (细粒度)                                                        │
│ t=31: Vf5 (64 tokens)                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

**T=32 (队列已满)**

```
输入: Frame 32
编码: Vc32(4tok), Vf32(64tok)

队列状态: [Vc2, Vc3, ..., Vc31, Vc32] (长度=31, Vc1被挤出)

组装输入 (无需填充):
┌─────────────────────────────────────────────────────────────────────┐
│ 历史 (粗粒度) - 全部是真实数据                                        │
│ t=0:  Vc2  ←最早的历史                                              │
│ t=1:  Vc3                                                           │
│  ...                                                                │
│ t=29: Vc31                                                          │
│ t=30: Vc32                                                          │
├─────────────────────────────────────────────────────────────────────┤
│ 当前 (细粒度)                                                        │
│ t=31: Vf32 (64 tokens)                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

**T=100 (稳态运行)**

```
输入: Frame 100
编码: Vc100(4tok), Vf100(64tok)

队列状态: [Vc70, Vc71, ..., Vc99, Vc100] (长度=31, 滑动窗口)

组装输入:
┌─────────────────────────────────────────────────────────────────────┐
│ 历史 (粗粒度)                                                        │
│ t=0:  Vc70  (约3秒前 @10Hz)                                         │
│ t=1:  Vc71                                                          │
│  ...                                                                │
│ t=30: Vc100                                                         │
├─────────────────────────────────────────────────────────────────────┤
│ 当前 (细粒度)                                                        │
│ t=31: Vf100 (64 tokens)                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心代码实现

### 4.1 推理时的队列更新

```python
# trained_agent.py: 424-434
# 将当前帧的粗粒度tokens加入历史队列
self._coarse_hist_tokens.append(Vc.cpu())

# 构建历史序列 (左填充)
H = self.history  # 31
hist = list(self._coarse_hist_tokens)
if len(hist) < H:
    pad_needed = H - len(hist)
    first = hist[0] if hist else Vc
    hist = [first] * pad_needed + hist  # 用最早的帧填充
else:
    hist = hist[-H:]  # 保留最近H帧
```

### 4.2 组装模型输入

```python
# trained_agent.py: 435-445
# 粗粒度历史 tokens + 时间索引
coarse_list = []
coarse_tidx = []
for t, tok4 in enumerate(hist):
    coarse_list.append(tok4.to(device))
    coarse_tidx.append(torch.full((4,), fill_value=t, dtype=torch.long))

coarse_tokens = torch.cat(coarse_list, dim=0).unsqueeze(0)  # (1, 31*4, C)
coarse_tidx = torch.cat(coarse_tidx, dim=0).unsqueeze(0)    # (1, 31*4)

# 细粒度当前帧 tokens + 时间索引
fine_tokens = Vf.unsqueeze(0)  # (1, 64, C)
fine_tidx = torch.full((1, 64), fill_value=H, dtype=torch.long)  # t=31
```

### 4.3 模型前向传播

```python
# trained_agent.py: 447-452
with torch.inference_mode():
    tau = self.planner_model(
        coarse_tokens, coarse_tidx,   # 历史 (1, 124, C), (1, 124)
        fine_tokens, fine_tidx,        # 当前 (1, 64, C), (1, 64)
        instr                          # 文本指令
    )  # 输出: (1, 8, 3) 预测轨迹
```

---

## 5. 设计优势

### 5.1 计算效率

| 方案 | 每帧 Token 数 | 32帧总 Token 数 |
|------|-------------|----------------|
| 全分辨率 (576tok/帧) | 576 | 18,432 |
| **本方案** | 4 (历史) + 64 (当前) | 31×4 + 64 = **188** |

**压缩比: 98×**

### 5.2 信息保留

- **长期记忆 (粗粒度)**: 保留全局语义（房间布局、走廊方向、目标大致位置）
- **短期记忆 (细粒度)**: 保留空间细节（障碍物边缘、目标精确位置、可通行区域）

### 5.3 滑动窗口的优势

- 固定内存占用: O(H) 而非 O(T)
- 自然遗忘: 久远的历史自动丢弃
- 实时性: 每帧只需编码一次，复用历史编码

---

## 6. 与训练数据的对应

训练时从轨迹中采样帧序列，模拟推理时的滑动窗口：

```python
# train.py (数据加载逻辑)
# 采样 history+1 帧图像
# 前 history 帧 → 编码为粗粒度 (每帧4 tokens)
# 最后1帧 → 编码为细粒度 (64 tokens)
# 标签: 未来轨迹点
```

这确保了训练和推理时的输入分布一致。

---

## 7. 关键问题：视觉 Tokens 没有 Token ID！

### 7.1 核心问题

> 这些视觉特征不在 Qwen3 的词表里，它们的 token id 是多少？

**答案：它们根本没有 token id！**

### 7.2 原理解释

LLM（如 Qwen3）有两种输入方式：

| 输入方式 | 适用场景 | 数据类型 |
|---------|---------|---------|
| `input_ids` | 文本输入 | 离散整数，通过词表查找嵌入 |
| `inputs_embeds` | 多模态输入 | 连续向量，**绕过词表** |

本模型使用 **`inputs_embeds`** 方式，直接把连续的嵌入向量送入 Transformer：

```python
# model.py: 286
out = self.llm(inputs_embeds=seq, attention_mask=attn, ...)
```

### 7.3 视觉特征投影流程

```
视觉特征 (DINOv2+SigLIP)
     │
     │  维度: (B, N_tokens, 1536)
     ▼
┌─────────────────────────────────────────┐
│     CrossModalityProjector              │
│  ┌─────────────────────────────────┐    │
│  │  LayerNorm(1536)                │    │
│  │       ↓                         │    │
│  │  Linear(1536 → D) + GELU        │    │
│  │       ↓                         │    │
│  │  Linear(D → D)                  │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
     │
     │  维度: (B, N_tokens, D)
     │  D = LLM 隐藏维度 (Qwen3-0.6B: 896)
     ▼
直接与文本嵌入拼接，送入 LLM
```

### 7.4 模型前向传播关键代码

```python
# model.py: 257-291 (简化版)
def forward(self, coarse_tokens, fine_tokens, instructions, ...):
    
    # 1. 视觉特征投影 (连续向量 → LLM 嵌入空间)
    vis_c = self.proj(coarse_tokens)   # (B, 124, D)  ← 无 token id
    vis_f = self.proj(fine_tokens)     # (B, 64, D)   ← 无 token id
    
    # 2. 文本嵌入 (离散 token id → 查词表 → 嵌入)
    tok = self.tokenizer(instructions, ...)
    txt_emb = self.llm.get_input_embeddings()(tok['input_ids'])  # (B, L_txt, D)
    
    # 3. 特殊 ACT token (可学习参数，无 token id)
    act = self.act_token.expand(B, 1, -1)  # (B, 1, D)
    
    # 4. 拼接所有嵌入
    seq = torch.cat([txt_emb, vis_c, vis_f, act], dim=1)  # (B, L_total, D)
    
    # 5. 使用 inputs_embeds 而非 input_ids！
    out = self.llm(inputs_embeds=seq, attention_mask=attn, ...)
```

### 7.5 各部分的 Token ID 情况

| 组件 | 是否有 Token ID | 来源 |
|------|----------------|------|
| 文本指令 | ✅ 有 | Qwen3 Tokenizer 编码 |
| 粗粒度视觉 tokens | ❌ **无** | CrossModalityProjector 投影 |
| 细粒度视觉 tokens | ❌ **无** | CrossModalityProjector 投影 |
| TVI 时间标记 | ❌ **无** | nn.Embedding(max_time, D) |
| ACT token | ❌ **无** | nn.Parameter (可学习) |

### 7.6 训练时哪些部分被优化？

```python
# model.py: 195-207
self.llm.requires_grad_(not cfg.freeze_llm)  # LLM 可选冻结
self.proj.requires_grad_(True)               # Projector 必须训练！
self.tvi.requires_grad_(True)                # TVI Embedding 训练
self.planner.requires_grad_(True)            # Planner MLP 训练
```

**关键**：`CrossModalityProjector` 是必须训练的，它负责将视觉特征对齐到 LLM 的嵌入空间。这就是多模态模型的"桥梁"。

### 7.7 为什么这样设计？

1. **灵活性**：视觉编码器（DINOv2/SigLIP）和语言模型可以独立预训练
2. **效率**：不需要为每种视觉特征分配离散的 token id
3. **连续性**：视觉信息本质上是连续的，强制离散化会丢失信息
4. **标准做法**：LLaVA、Flamingo、PaLM-E 等多模态模型都采用类似方法

### 7.8 与 Qwen3-VL 原生视觉编码的区别

| 方面 | Qwen3-VL 原生 | 本模型 |
|------|--------------|-------|
| 视觉编码器 | Qwen 自带的 ViT | DINOv2 + SigLIP |
| 投影方式 | 内置 projector | 外置 CrossModalityProjector |
| 特征维度 | Qwen 内部统一 | 1536 → D 投影 |
| 是否冻结 LLM | 通常冻结 | 可选 (`freeze_llm`) |

本模型**替换了 Qwen3-VL 的视觉编码器**，使用自定义的 DINOv2+SigLIP 组合，通过可训练的 Projector 对齐到语言模型空间。

---

## 8. 深入理解：`input_ids` vs `inputs_embeds`

### 8.1 Transformer 的输入层架构

无论哪种输入方式，最终进入 Transformer 注意力层的都是**连续的嵌入向量**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Transformer 核心层                                │
│                                                                     │
│   Self-Attention → Feed-Forward → ... → Self-Attention → Output    │
│                                                                     │
│                         ↑                                           │
│                    嵌入向量 (B, L, D)                                │
│                         ↑                                           │
│           ┌─────────────┴─────────────┐                             │
│           │                           │                             │
│     input_ids 路径              inputs_embeds 路径                  │
│           │                           │                             │
└───────────┼───────────────────────────┼─────────────────────────────┘
            │                           │
```

### 8.2 方式一：`input_ids` (传统文本输入)

#### 工作流程

```
"Hello world"
     │
     ▼ Tokenizer
[15496, 995]  ← 离散的 token ID (整数)
     │
     ▼ Embedding Layer (词表查找)
     │
     │  词表: nn.Embedding(vocab_size, hidden_dim)
     │  vocab_size = 151,936 (Qwen3)
     │  hidden_dim = 896 (Qwen3-0.6B)
     │
     ▼
[[0.12, -0.34, ...],   ← 连续向量 (896维)
 [0.56, 0.78, ...]]
     │
     ▼ 进入 Transformer
```

#### 代码示例

```python
# 传统文本输入方式
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

text = "Hello world"
tokens = tokenizer(text, return_tensors='pt')
# tokens['input_ids'] = tensor([[15496, 995]])  ← 离散整数

output = model(**tokens)  # 内部: embedding_layer(input_ids) → transformer
```

#### 关键：词表是有限的

```python
print(tokenizer.vocab_size)  # 151,936
# 每个 token id 必须在 [0, 151935] 范围内
# 词表是预定义的，训练后固定
```

### 8.3 方式二：`inputs_embeds` (直接嵌入输入)

#### 工作流程

```
任意来源的特征
(图像、音频、传感器...)
     │
     ▼ 自定义投影层
连续向量 (B, L, D)  ← 直接是嵌入，跳过词表！
     │
     ▼ 进入 Transformer
```

#### 代码示例

```python
# 直接嵌入输入方式
import torch

# 假设我们有一些来自视觉编码器的特征
visual_features = torch.randn(1, 10, 896)  # (batch=1, 10个token, 896维)

# 直接作为嵌入输入，不经过词表！
output = model(inputs_embeds=visual_features)
```

### 8.4 核心区别对比图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    input_ids 路径                                    │
│                                                                     │
│  "follow the person"                                                │
│         │                                                           │
│         ▼                                                           │
│  Tokenizer: [12345, 678, 9012]  ← 必须在词表中存在                   │
│         │                                                           │
│         ▼                                                           │
│  nn.Embedding(vocab_size, D)   ← 查表操作，离散→连续                 │
│         │                                                           │
│         ▼                                                           │
│  [e_12345, e_678, e_9012]      ← 连续向量 (3, D)                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    inputs_embeds 路径                                │
│                                                                     │
│  视觉特征 (DINOv2 + SigLIP)                                         │
│         │                                                           │
│         ▼                                                           │
│  CrossModalityProjector        ← 可训练的 MLP                       │
│  (1536 → D)                                                         │
│         │                                                           │
│         ▼                                                           │
│  [v_1, v_2, ..., v_64]         ← 连续向量 (64, D)                   │
│                                                                     │
│  注意：没有 token id！没有词表查找！                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.5 为什么多模态模型必须用 `inputs_embeds`？

#### 问题：视觉特征无法用 token id 表示

```python
# 假设我们想给图像分配 token id
image_feature = encode_image(img)  # 得到 (64, 1536) 的连续特征

# 问题1：这是连续向量，不是离散整数
# 问题2：即使量化成整数，词表只有 151,936 个位置
# 问题3：图像特征的语义空间和文本完全不同

# 错误尝试：
fake_ids = torch.tensor([151937, 151938, ...])  # 超出词表范围！会报错
```

#### 解决方案：绕过词表

```python
# 正确做法：直接投影到嵌入空间
projector = nn.Linear(1536, 896)  # 视觉维度 → LLM 嵌入维度
visual_embeds = projector(image_feature)  # (64, 896)

# 与文本嵌入拼接
text_embeds = model.get_input_embeddings()(text_ids)  # (L_text, 896)
combined = torch.cat([text_embeds, visual_embeds], dim=0)

# 使用 inputs_embeds 输入
output = model(inputs_embeds=combined)
```

### 8.6 本模型的实际实现

```python
# model.py: 257-286 (简化)
def forward(self, coarse_tokens, fine_tokens, instructions, ...):
    
    # ===== 视觉路径：inputs_embeds =====
    # coarse_tokens: (B, 124, 1536) - 来自 DINOv2+SigLIP，无 token id
    # fine_tokens:   (B, 64, 1536)  - 来自 DINOv2+SigLIP，无 token id
    vis_c = self.proj(coarse_tokens)  # → (B, 124, 896)
    vis_f = self.proj(fine_tokens)    # → (B, 64, 896)
    
    # ===== 文本路径：input_ids → embedding =====
    tok = self.tokenizer(instructions, ...)  # 得到 input_ids
    txt_emb = self.llm.get_input_embeddings()(tok['input_ids'])  # 查词表
    # txt_emb: (B, L_text, 896)
    
    # ===== 合并：全部变成连续嵌入 =====
    seq = torch.cat([txt_emb, vis_c, vis_f, act_token], dim=1)
    # seq: (B, L_text + 124 + 64 + 1, 896)
    
    # ===== 使用 inputs_embeds 输入 LLM =====
    out = self.llm(inputs_embeds=seq, ...)  # 不用 input_ids！
```

### 8.7 形象比喻

把 LLM 想象成一个**只懂中文的翻译官**：

| 方式 | 比喻 |
|------|------|
| `input_ids` | 给翻译官一本**中文词典**的页码列表，他自己去查每个词 |
| `inputs_embeds` | 直接给翻译官**写好的中文句子**，跳过查词典的步骤 |

对于视觉输入：
- 图像"说的是另一种语言"（1536维特征空间）
- `CrossModalityProjector` 是一个**实时翻译器**，把图像语言翻译成中文
- 翻译好的结果直接交给翻译官（`inputs_embeds`），不需要查词典

### 8.8 两种方式的完整对比

| 特性 | `input_ids` | `inputs_embeds` |
|------|-------------|-----------------|
| 数据类型 | 离散整数 | 连续向量 |
| 是否需要词表 | ✅ 必须 | ❌ 不需要 |
| 适用场景 | 纯文本 | 多模态/自定义嵌入 |
| 灵活性 | 受限于词表大小 | 任意维度投影 |
| 梯度流向 | 词表 embedding 可训练 | 投影层可训练 |
| 本模型使用 | 文本指令 | 视觉特征 + TVI + ACT |

### 8.9 为什么 Hugging Face Transformers 支持这两种方式？

```python
# transformers 库的 forward 函数签名 (简化)
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,      # 方式1
    inputs_embeds: Optional[torch.FloatTensor] = None, # 方式2
    attention_mask: Optional[torch.Tensor] = None,
    ...
):
    # 内部逻辑：
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)  # 查词表
    # 否则直接使用 inputs_embeds
    
    # 然后送入 transformer 层
    hidden_states = inputs_embeds
    for layer in self.layers:
        hidden_states = layer(hidden_states, attention_mask, ...)
```

**设计哲学**：Transformer 本质上只处理连续向量，`input_ids` 只是为了方便文本处理而提供的"语法糖"。`inputs_embeds` 才是更底层、更灵活的接口。

### 8.10 核心洞察

> **`inputs_embeds` 让 LLM 能够处理任何可以投影到其嵌入空间的信息，这是多模态模型的基础。**

本模型正是利用这一特性，将 DINOv2+SigLIP 的视觉特征通过 `CrossModalityProjector` 投影后，与文本嵌入无缝拼接，让 Qwen3 能够"看见"图像并做出导航决策。
