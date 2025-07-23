* `kan.ipynb`：用于演示 KAN（Kolmogorov-Arnold Network）基本功能的笔记本。

* `kan_classification.ipynb` 与 `kan_classification.py`：内容相同，仅格式不同。用于在风险评估数据上**直接训练 KAN 模型**的实验，未使用任何初始化。

* `kan_classification_grid_search`：对 KAN 进行训练，并对关键超参数进行网格搜索，以寻找最佳配置。

* `lr++`：LR++ 方法，先从一个已训练的逻辑回归（LR）模型初始化 KAN，然后在训练集上进一步微调，并在测试集和 OOT（Out-of-Time）数据集上进行评估。
