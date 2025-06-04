# Style2Code

**Style2Code** 是一种面向可控代码风格生成的模型框架，结合了显式风格向量提取、对比式表示学习和大语言模型的强大生成能力。与以往依赖隐式提示或手工模板的方法不同，Style2Code 通过连续向量显式编码代码风格，实现了细粒度、可解释且用户可控的代码风格迁移与生成。

📄 论文: [arXiv:2505.19442](https://arxiv.org/abs/2505.19442)
📦 Hugging Face 模型权重: [点击访问](https://huggingface.co/DUTAOZHANG/Styele2Code_model2/upload/main)
📦 训练数据集: 尚未公开
[English version](README_en.md) | [中文版](README_zh.md)

---

## 🚀 项目特性

✅ **双阶段训练流程**：

* 阶段 1：使用对比式学习训练风格编码器，抽取可区分的代码风格向量。
* 阶段 2：冻结风格编码器，微调解码器，实现风格控制的代码生成。

✅ **显式风格向量注入**：
直接将显式编码的 34 维风格向量与代码输入融合，提升生成代码的风格可控性与稳定性。

✅ **支持多维风格特征**：
涵盖命名风格、结构布局、空白格式等多维特征，支持在不同代码生成任务中灵活迁移。

✅ **多指标评测**：
使用 BLEU、ROUGE、CSS (Code Style Similarity) 等指标，量化评估生成代码在功能与风格一致性上的表现。

---

## 🏗️ 模型架构与训练方法

Style2Code 基于 Flan-T5 语言模型，核心包括：

* **Style Encoder**：采用 BiGRU + MLP，独立学习代码风格向量（34 维 → 1024 维）。
* **StyleControlledGenerator**：将显式风格向量作为伪 token，注入至解码器。
* **对比式损失 + 样式损失**：实现风格向量的一致性和解码器的风格适配。

详细架构、损失函数和可视化结果请参考论文和 `docs/` 文件夹。

---

## ⚙️ 安装与使用方法

```bash
# 克隆项目
git clone https://github.com/your_username/Style2Code.git
cd Style2Code

# (可选) 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 🔍 推理

修改 `run.py` 中的文件路径，运行：

```bash
python run.py
```

### 🏋️‍♂️ 阶段 1 训练

```bash
python train_stage1/train_stage1.py
```

### 🏗️ 阶段 2 训练

```bash
bash train_stage2/run_ddp.sh
```

### 📊 评估

```bash
python eval/runner.py
```

---

## 🔗 预训练模型

我们已在 Hugging Face 上分享了 Style2Code 的部分预训练模型权重，欢迎下载并在新任务上微调：
👉 [模型权重下载](https://huggingface.co/DUTAOZHANG/Styele2Code_model2/upload/main)

---

## 📄 引用

如果您在研究中使用了 Style2Code，请引用以下论文：

```bibtex
@article{zhang2025style2code,
  title={Style2Code: A Style-Controllable Code Generation Framework with Dual-Modal Contrastive Representation Learning},
  author={Zhang, Dutao and Kovalchuk, Sergey and He, YuLong},
  journal={arXiv preprint arXiv:2505.19442},
  year={2025}
}
```

---

## 📬 联系方式

如有问题或合作意向，欢迎通过 issue 联系或邮件咨询：

* ✉️ [zh19980811@gmail.com](mailto:zh19980811@gmail.com)

让我们一起，让代码更有风格！ 🎨

