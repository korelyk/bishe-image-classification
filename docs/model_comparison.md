# 多模型实验对比结果

- 数据集：coco-road7-crops
- 类别：person / bicycle / car / motorcycle / bus / truck / traffic_light
- 训练/验证/测试样本量：168 / 56 / 56
- 准确率最优模型：ResNet50
- 推理速度最优模型：MobileNetV2

## 1. 主实验结果

| 模型 | 参数量(M) | 类别数 | 验证准确率 | 测试准确率 | 单张耗时(ms) | FPS | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| ResNet50 | 23.52 | 7 | 33.93% | 35.71% | 153.46 | 6.52 | 特征表达强，参数量较大 |
| MobileNetV2 | 2.23 | 7 | 46.43% | 33.93% | 39.59 | 25.26 | 轻量化明显，推理速度最快 |
| EfficientNet-B0 | 4.02 | 7 | 16.07% | 26.79% | 54.06 | 18.5 | 精度与速度较均衡 |

## 2. 复杂场景鲁棒性

- **gaussian_noise**：ResNet50 100.0% / MobileNetV2 85.71% / EfficientNet-B0 57.14%，最佳模型为 **ResNet50**。
- **low_light**：ResNet50 42.86% / MobileNetV2 28.57% / EfficientNet-B0 28.57%，最佳模型为 **ResNet50**。
- **normal**：ResNet50 28.57% / EfficientNet-B0 28.57% / MobileNetV2 14.29%，最佳模型为 **ResNet50**。
- **partial_crop**：MobileNetV2 57.14% / ResNet50 42.86% / EfficientNet-B0 28.57%，最佳模型为 **MobileNetV2**。

## 3. 说明

> 当前实验基于公开 COCO 目标检测数据裁剪出的七类道路目标分类集，已能用于展示真实公开数据来源、训练流程、模型对比和论文实验方法。若后续拿到更贴近学校题目的专用七类数据，可直接复用现有脚本重跑。
