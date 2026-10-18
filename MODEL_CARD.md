# 第三轮中文实体抽取 LoRA

## 模型

- 项目：基于千问1.5B与LoRA微调的中文实体抽取模型。
- 基座：[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)。
- 基座 revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- 附件仅包含 LoRA adapter，不包含基座权重；加载时需要基座。
- LoRA：r=8、alpha=16、dropout=0、bias=none，q_proj/v_proj，可训练参数 1,089,536。

## 用途与输出

用于 CLUENER 风格的 10 类中文命名实体抽取教学、实验与错误分析。输出 JSON 数组，实体含 type、text、start、end；位置按 Python 字符索引计数，零起点、左闭右开。

类别为 address、book、company、game、government、movie、name、organization、position、scene。请使用仓库 `configs/inference.json` 中保存的提示词和生成配置。自行改变提示词、后处理或评分口径可能改变结果。

## 训练记录

从 CLUENER 官方训练文件的 10748 条中固定划分 9673 条训练、1075 条项目验证。batch size=2、AdamW lr=1e-4、weight decay=0，共三轮、14511 次参数更新。只对答案与结束部分计算普通 causal cross entropy。

第一轮之后发生中断；第二轮前重新创建了优化器和数据顺序生成器，第三轮继续第二轮状态。因此最后优化器累计更新为 9674 次，而权重历史累计更新为 14511 次。未完整保存各阶段随机数状态，不承诺逐位复现。

## 评估与局限

在 1343 条公开 dev（本项目留出集）上，TP=775、FP=183、FN=2297，Precision=0.808977、Recall=0.252279、F1=0.384615。754 条回答未通过校验，占 56.14%。

任何实体校验失败，整条回答按空预测处理。该严格口径与官方排行榜不等价，精确率必须结合召回率和无效回答率理解。字符位置不准确仍是主要局限。结果仅来自这一数据划分和训练经历，未做跨领域或多随机种子验证，不应作为高可靠性自动抽取系统。

第六轮模型的追加结果来自项目验证集，未包含在本附件中。本次默认发布第三轮。

## 来源与许可

基座采用 Apache-2.0，原许可随 adapter 附件保存在 `LICENSE.base-model`。本项目代码和新增 adapter 的许可见仓库发布说明。数据来自 [CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)，请参照数据提供方的说明获取和使用。
