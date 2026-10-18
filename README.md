# 基于千问1.5B与LoRA微调的中文实体抽取模型

基于 Qwen2.5-1.5B-Instruct，在约 8 GB 显存的环境中完成 CLUENER 中文实体抽取实验。输入中文文本，输出包含类别、原文文本和字符位置的 JSON 实体列表。

本项目提供单句推理、训练及已保存预测的离线评估入口。默认发布检查点为第三轮 LoRA；第六轮作为追加实验记录。代码由 notebook 分步整理而来。

## 环境与安装

实际使用环境：Windows、Python 3.12.12、RTX 5060 Laptop GPU（约 8 GB）、PyTorch 2.10.0+cu128。当前训练入口要求 CUDA 和 BF16；推理入口也使用 CUDA。其他硬件、Linux 和全新环境尚未验证。

在独立 Python 环境中，进入项目根目录执行：

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

所有后续命令均从项目根目录运行。模型与 tokenizer 首次使用时需要下载；基座固定为 `Qwen/Qwen2.5-1.5B-Instruct`，revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。

## 使用第三轮 adapter

在本仓库 [Release 页面](https://github.com/cxx-star/qwen1.5b-lora-chinese-ner/releases/tag/v0.1.1) 下载 `epoch3-adapter.zip`，解压到 `artifacts/epoch3/`，使权重文件位于 `artifacts/epoch3/adapter/adapter_model.safetensors`。附件说明见 [模型卡](MODEL_CARD.md)。

```powershell
python predict.py --adapter_path artifacts/epoch3/adapter --sentence "李华在北京工作"
```

脚本输出模型原始回答、校验状态和解析后的实体。一次实际运行得到了北京 `address [3,5)` 和李华 `name [0,2)`，校验通过；这只是入口演示，不代表任意文本均能正确抽取。

参数说明：

```powershell
python predict.py --help
```

提示词、模型 revision 和生成配置见 `configs/inference.json`。固定使用贪心生成，`max_new_tokens=256`、`repetition_penalty=1.0`。

## 数据与训练

从 [CLUENER2020 官方项目](https://github.com/CLUEbenchmark/CLUENER2020#数据下载地址) 获取数据，保留原文件行顺序，将 `train.json` 和 `dev.json` 放入 `data/raw/`。文件虽然以 `.json` 结尾，内容是每行一条 JSON。

- 官方训练数据：10748 条，转换后 23971 个实体。
- 项目训练/验证划分：9673 / 1075 条，使用仓库中的 `data/splits/train_val_indices.json`。
- 官方公开 dev：1343 条，3072 个实体，在本项目中作为留出评估集，并非官方隐藏测试集。

数据文件 SHA-256：

```text
train.json 60299d202779dc84f3c3bd81f4dc08f5323d0dbf762cb50cd62dd51aa9a871f8
dev.json   90036b59832e007c45d11eafe5ed80549f58a18a29aaa7fe8b1229622c69ec38
```

训练命令会新建基座及 LoRA，而非继续训练默认发布的 adapter：

```powershell
python train.py --config configs/training.json --data data/raw/train.json --split data/splits/train_val_indices.json --output_dir runs/my_training
```

使用新的输出目录，避免覆盖旧检查点。每轮保存 adapter、tokenizer 和 `optimizer.pt`。目前入口不提供自动恢复训练或自动验证；保存优化器也不等于保存了完整随机数和数据迭代状态。

默认配置：3 轮、batch size 2、AdamW lr=1e-4、weight decay=0；LoRA r=8、alpha=16、dropout=0，作用于 q_proj/v_proj，共 1,089,536 个可训练参数。训练损失只计算答案及结束部分，提示和 padding 标签为 -100。

**历史权重与新训练入口的区别：** 默认发布的第三轮权重在第二轮前因中断而重置过优化器和数据顺序。当前训练入口按用户选择简化为连续训练，不复演这次中断，也不保证重训得到完全相同的权重或分数。

## 离线评估

准备 UTF-8 JSONL，每行含 `sentence`、`gold`、`response`；其中 `response` 是模型原始回答字符串。运行：

```powershell
python evaluate.py --predictions examples/predictions.jsonl
```

仓库示例是人工构造的接口演示，预期计数 `(2, 0, 0, 0)`，不是模型实验结果。将路径替换为实际保存的预测文件即可复算；此命令不加载模型、不生成预测。当前没有批量生成预测的命令行入口。

评估规则：类别、起止位置严格匹配，按样本内实体集合计算 micro Precision/Recall/F1。位置是 Python 字符索引、左闭右开，须满足 `sentence[start:end] == text`。任何一个实体格式或位置校验失败，整条回答按空预测计分，gold 仍贡献 FN。合法空数组不会记为无效回答。

这一规则使无效回答不会贡献有效预测 FP，因此必须结合召回率和无效回答率理解精确率。评分口径不等同于官方排行榜，不能直接比较。

## 主要结果与限制

同一批 1343 条留出样本、相同提示和生成配置下：

- 基座：TP/FP/FN=10/7/3062，Precision=0.5882、Recall=0.0033、F1=0.0065，无效回答 278；1049 条合法回答为空数组。
- 第三轮 LoRA：TP/FP/FN=775/183/2297，Precision=0.8090、Recall=0.2523、F1=0.3846，无效回答 754（56.14%）；没有主动输出空数组。

LoRA 改善了抽取效果，但字符位置错误仍造成大量整条回答失效。基座合法率较高主要受到大量空回答影响，不能据此认为它的抽取能力更好。

在 1075 条项目验证样本上，第三轮 F1=0.3808，第六轮 F1=0.4670。第六轮结果来自验证集，不应当作留出集成绩。追加训练发生在已查看第三轮留出成绩之后；默认发布仍采用第三轮，未宣称第六轮获得新的独立测试结论。

另一个同 200 条验证样本的实验中，512 条样本训练 5 轮，普通损失 F1=0.1310；将位置数字 token 权重设为 5 后 F1=0.0913，本次加权未改善效果。结构化指标见 `reports/results.json`。

本项目未做多随机种子或跨领域验证，不是生产级实体抽取服务。

## 当前验证范围

- 已独立运行第三轮 adapter 的单句推理。
- 已从原始预测独立复算第三轮留出集计数和指标，与实验记录一致。
- 数据转换、固定划分、编码和 padding 已完成逐步检查；训练函数与原 notebook 的函数一致。
- 新的训练入口已完成导入和命令行帮助检查；独立端到端训练检查按项目时间安排跳过。未声称在新环境完成全流程复现。

## 代码结构

`src/data.py` 负责转换、读取、划分、编码和补齐；`src/evaluation.py` 负责严格校验与评分；`src/inference.py` 负责模型加载和生成；`src/training.py` 负责新 LoRA 初始化、单轮训练和检查点保存。根目录三个脚本分别提供推理、离线评估和训练入口。

## 来源与许可

- 基座及其许可：[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)、[固定版本 Apache-2.0 许可](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE)。
- 数据与标注规则：[CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)，论文 [Xu et al., 2020](https://arxiv.org/abs/2001.04351)。数据请从来源获取，并遵循其使用条件。
- 本项目源码及新增 LoRA adapter 采用 [Apache-2.0](LICENSE)，Copyright 2026 cxx-star。允许商用、修改和再分发，须遵守许可证中的声明保留等条件；详见 [NOTICE](NOTICE)。此授权不替代基座和数据集自身的许可。
