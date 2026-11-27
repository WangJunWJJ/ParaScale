# 知识蒸馏验收

对应INF-04；当前`distillation.quality`返回BLOCKED。

已有[Llama蒸馏](../../gpu/example_007_llama_distillation/README.md)与[YOLO蒸馏](../../gpu/example_008_yolo_distillation/README.md)可独立运行。正式编排必须冻结教师、学生初始化和训练预算，比较KD与相同学生的纯任务损失基线，并在独立数据上评估。

学生导出后须脱离教师重载推理。模型变小造成的速度差不能当作KD自身质量收益；随机模型或训练集loss降低不构成独立质量通过。
