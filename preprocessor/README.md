`preprocessor`：包含为 RiskReasoner 进行数据预处理的脚本。

* `prior.py`：处理建立基线所需的数据（例如专家系统、直接推理的大语言模型）。

* `posterior.py`：将机器学习模型的预测结果添加到提示词中供大语言模型参考（因此命名为 "posterior"）。
