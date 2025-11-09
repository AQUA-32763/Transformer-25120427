# Manually-implement-transformer
A Transformer was manually implemented to complete simple Chinese-English translation tasks

# Transformer 中英文翻译模型

基于 PyTorch 实现的 Transformer 模型，用于中英文机器翻译任务。

## 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU with ≥ 8GB VRAM (推荐)
- **内存**: ≥ 16GB RAM
- **存储**: ≥ 2GB 可用空间

### 软件依赖

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```
## 快速开始

### 数据准备
确保数据文件位于 `dataset/en-cn/` 目录下：
- `train.txt`: 训练数据
- `test.txt`: 测试数据

### 运行基础模型
```bash
python Final_Transformer.py
```
随机种子: 代码中已固定为 42 (torch.manual_seed(42))

## 运行消融实验
```bash
python AblationExperiment.py
```
随机种子: 代码中已固定为 42 (torch.manual_seed(42))

## 项目文件排布说明
transformer -main/\
├── ablation_images/           # 消融实验可视化图表\
├── ablation_results/          # 消融实验详细结果\
├── images/           # 基线模型训练可视化图表\
├── results/        # 基线模型训练详细结果\
├── dataset/                 # 训练和测试数据目录\
├── save/                    # 模型保存目录（由于预训练文件较大并未上传，运行任意一个代码就能在这个文件夹得到保存好的模型）\
├── AblationExperiment.py    # 消融实验代码\
├── Final_Transformer.py     # 主模型训练代码\
├── langconv.py              # 中文简繁转换工具\
├── zh_wiki.py              # 中文简繁转换工具\
├── README.md                # 项目说明文档\
└── requirements.txt         # Python依赖包列表
## 文件夹说明
在这个小节中对项目输出文件夹进行解释。\
Results和Images两个文件夹中保存着基线模型的训练结果。results文件夹中保存着记录训练过程训练损失和测试损失的json文件和测试案例的json文件（包含详细预测结果和评估结果）。Image中保存着数据预处理阶段的可视化图像，例如位置编码热图、注意力掩码图。还包括最后的训练曲线可视化图和评估图。同理，ablation开头的两个文件夹也保存着类似的内容。\
Dataset文件夹中保存着训练集和测试集，和验证集等文件，按照文件名进行区分即可。
