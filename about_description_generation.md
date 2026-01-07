# OpenTrackVLA Instruction 生成机制分析

## 核心结论

**Instruction（自然语言指令）在这个仓库里是"黑箱"——它不是在代码里生成的，而是预先定义好的静态数据。**

---

## 数据流追踪

### Instruction 的来源链路

```
train.json.gz (episode数据集)
    └─ episode.info['instruction'] = "Track the woman wearing a light gray long-sleeve top..."
                │
                ▼
baseline_agent.py 第58行:
    instruction = env.current_episode.info.get('instruction', None)
                │
                ▼
episode.json 第141行:
    result['instruction'] = instruction  # 采集时直接复制
                │
                ▼
make_tracking_data.py 第291-295行:
    status = load_episode_status(status_path)
    instr_candidate = status.get("instruction")  # 从 episode.json 读取
```

### Episode 数据集结构

`train.json.gz` 里的实际数据：

```json
{
  "episode_id": "4",
  "scene_id": "hm3d/train/00083-16tymPtM7uS/16tymPtM7uS.basis.glb",
  "info": {
    "main_humanoid_name": "female_20",
    "main_human_semantic_id": 1035,
    "extra_humanoid_names": ["female_15", "female_12", "male_32", "male_2"],
    "instruction": "Track the woman wearing a light gray long-sleeve top and white pants.",
    "episode_mode": "stt",
    "human_num": 4,
    ...
  }
}
```

### Humanoid 元数据结构

`humanoid_infos.json` 里只有基础信息，**没有外观描述**：

```json
{
  "name": "female_20",
  "gender": "female",
  "texture": "ATLAS_testset/test_100/xxx.png",
  "semantic_id": 1035
}
```

---

## 实际示例

| episode_id | main_humanoid_name | instruction |
|------------|-------------------|-------------|
| 4 | female_20 | "Track the woman wearing a light gray long-sleeve top and white pants." |
| 5 | female_22 | "Pursue the person in a black leather bodysuit." |
| 17 | female_21 | "Chase the woman in a gold metallic top and brown boots." |
| 23 | female_24 | "Chase the woman in a green and brown top with a necklace." |
| 28 | male_23 | "Pursue the individual dressed in a green and purple superhero costume." |

---

## 关键洞察

| 数据类型 | 来源 | 生成方式 | 代码位置 |
|---------|------|---------|---------|
| 轨迹 waypoints | navmesh + PID 控制 | 仿真自动生成 ✅ | `baseline_agent.py` |
| 人物初始位置 | episode 数据集 | 预定义 | `train.json.gz` |
| 人物 waypoints | episode 数据集 | 预定义 | `train.json.gz` |
| **instruction 文本** | episode 数据集 | **外部预标注 ❌** | 代码里找不到生成逻辑 |

---

## Instruction 可能的生成方式（推测）

这些 instruction 描述很可能来自以下方式之一：

### 1. 人工标注
有人查看每个 humanoid 的纹理贴图（texture），手写外观描述。

### 2. VLM 自动生成
使用 GPT-4V、LLaVA 等视觉语言模型，输入纹理图片，生成外观描述。

### 3. ATLAS 数据集自带
texture 路径是 `ATLAS_testset/test_100/xxx.png`，可能 ATLAS 数据集本身包含人物描述。

**但无论哪种方式，这个仓库里都没有相关代码。**

---

## 如果你想自己生成 Instruction

### 方法 1：使用 VLM 自动生成

```python
import random
from PIL import Image

def generate_humanoid_instruction(humanoid_name: str, texture_path: str, vlm_client) -> str:
    """
    使用 VLM 为 humanoid 生成自然语言描述
    
    Args:
        humanoid_name: humanoid 名称，如 "female_20"
        texture_path: 纹理图片路径
        vlm_client: VLM API 客户端（如 OpenAI GPT-4V）
    
    Returns:
        instruction: 自然语言指令
    """
    # 1. 加载纹理图片
    image = Image.open(texture_path)
    
    # 2. 调用 VLM 生成描述
    prompt = """Describe this person's clothing and appearance in a concise phrase.
    Focus on: clothing style, colors, distinctive features.
    Format: "a [gender] wearing [clothing description]"
    Example: "a woman wearing a light gray long-sleeve top and white pants"
    """
    
    description = vlm_client.generate(
        image=image,
        prompt=prompt,
        max_tokens=50
    )
    
    # 3. 组合成 instruction
    action_verbs = ["Track", "Follow", "Chase", "Pursue"]
    instruction = f"{random.choice(action_verbs)} {description}."
    
    return instruction


# 使用示例
# instruction = generate_humanoid_instruction(
#     humanoid_name="female_20",
#     texture_path="data/humanoids/textures/female_20.png",
#     vlm_client=openai_client
# )
# >>> "Track the woman wearing a light gray long-sleeve top and white pants."
```

### 方法 2：创建静态描述映射

如果 humanoid 数量有限，可以手动创建描述映射：

```python
HUMANOID_DESCRIPTIONS = {
    "female_0": "a woman in a casual blue dress",
    "female_1": "a woman wearing a red blouse and jeans",
    "female_20": "a woman wearing a light gray long-sleeve top and white pants",
    "male_0": "a man in a black suit",
    "male_23": "an individual dressed in a green and purple superhero costume",
    # ... 添加所有 humanoid
}

def get_instruction(humanoid_name: str) -> str:
    action_verbs = ["Track", "Follow", "Chase", "Pursue"]
    description = HUMANOID_DESCRIPTIONS.get(
        humanoid_name, 
        "the target person"  # fallback
    )
    return f"{random.choice(action_verbs)} {description}."
```

### 方法 3：修改 Episode 生成流程

如果你要生成自己的 episode 数据集，需要在生成器里加入 instruction 生成：

```python
def create_episode(scene_id, humanoid_name, ...):
    """创建单个 episode"""
    
    # 生成 instruction（你需要实现这部分）
    instruction = generate_humanoid_instruction(humanoid_name, texture_path, vlm_client)
    
    episode = {
        "episode_id": str(episode_id),
        "scene_id": scene_id,
        "info": {
            "main_humanoid_name": humanoid_name,
            "instruction": instruction,  # 关键！
            # ... 其他字段
        },
        # ... 其他字段
    }
    return episode
```

---

## 总结

```
┌─────────────────────────────────────────────────────────────┐
│  这个仓库开源了：                                            │
│    ✅ 训练代码（数据格式转换、模型训练）                      │
│    ✅ 轨迹采集（navmesh + PID 控制）                         │
│    ✅ 视觉编码（DINO + SiGLIP）                              │
│                                                             │
│  这个仓库没有开源：                                          │
│    ❌ Episode 数据集生成器                                   │
│    ❌ Instruction 生成逻辑（最有价值的部分）                  │
│    ❌ Humanoid 外观描述的原始数据/生成方法                   │
└─────────────────────────────────────────────────────────────┘
```

**真正的数据工程工作——为 100+ 种 humanoid 生成高质量的自然语言描述——被藏在 `train.json.gz` 这个预打包的数据集里了。**

如果你要复现或扩展这个工作，instruction 生成这一步需要自己实现。

---

## 相关文件

| 文件 | 作用 |
|------|------|
| `data/datasets/track/STT/train/train.json.gz` | Episode 数据集，包含预定义的 instruction |
| `humanoid_infos.json` | Humanoid 基础信息（无外观描述）|
| `baseline_agent.py` | 数据采集，第58行读取 instruction |
| `make_tracking_data.py` | 数据处理，第291行读取 instruction |

