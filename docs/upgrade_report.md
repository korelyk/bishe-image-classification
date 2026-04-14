# 系统升级回奏（公开数据强化版）

## 一、已完成改造项

### 1. 多模型对比
- 已统一接入 `ResNet50 / MobileNetV2 / EfficientNet-B0`；
- 已完成公开数据上的训练、验证、测试与速度评测；
- 前端可直接读取并展示模型对比表。

### 2. 数据增强
- 已加入旋转、翻转、裁剪、亮度/对比度扰动、高斯噪声、低照度模拟；
- 同时用于训练阶段和鲁棒性测试阶段。

### 3. Grad-CAM 可视化
- 单图识别支持返回热力图；
- 已保留 Grad-CAM 生成脚本，可继续为答辩补图。

### 4. 复杂测试场景
- 已加入低光照、噪声、局部裁剪鲁棒性测试；
- 前端已支持实时拍照识别与批量识别。

### 5. 公开数据补强
- 已不再停留于仓库演示图扩增；
- 新增基于 **COCO 2017 val 标注裁剪** 的七类道路目标分类数据构建脚本；
- 数据来源、图片文件、标注框、原始链接均可在 `data/experiments/dataset_manifest.json` 追溯。

## 二、关键产物

### 代码文件
- `app/main.py`
- `app/model_engine.py`
- `app/vision_models.py`
- `app/templates/index.html`
- `app/static/app.js`
- `app/static/style.css`
- `scripts/build_demo_dataset.py`
- `scripts/train_models.py`
- `scripts/benchmark_models.py`
- `scripts/generate_gradcam_gallery.py`
- `scripts/run_all_experiments.sh`

### 实验文件
- `data/experiments/dataset_manifest.json`
- `data/reports/model_comparison.json`
- `data/reports/robustness_report.json`
- `data/reports/training_summary.json`
- `data/experiments/checkpoints/*.pt`

### 文档文件
- `docs/model_comparison.md`
- `docs/experiment_results.md`
- `README.md`

## 三、当前论文可引用数据

### 数据集
- 名称：`coco-road7-crops`
- 类别：`person / bicycle / car / motorcycle / bus / truck / traffic_light`
- 规模：训练 168、验证 56、测试 56
- 来源：COCO 2017 val 标注裁剪

### 模型对比结果

| 模型 | 参数量(M) | 测试准确率 | 平均耗时(ms) | FPS |
|---|---:|---:|---:|---:|
| ResNet50 | 23.52 | 35.71% | 153.46 | 6.52 |
| MobileNetV2 | 2.23 | 33.93% | 39.59 | 25.26 |
| EfficientNet-B0 | 4.02 | 26.79% | 54.06 | 18.50 |

建议写法：
- 若突出**当前公开数据集上的最佳精度**，写 `ResNet50`；
- 若突出**轻量化实时性**，写 `MobileNetV2`；
- 若强调**多模型对比实验完整性**，三者共同保留。

## 四、运行方式

### 启动系统
```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 19001
```

### 重新生成全部实验结果
```bash
bash scripts/run_all_experiments.sh
```

## 五、现阶段结论与剩余建议

1. 已补上“公开网站找真实数据”的关键短板；
2. 已形成可复现的数据构建、训练、评测、对比、论文落盘链路；
3. 当前结果说明系统已具备真实实验过程，但公开裁剪数据规模仍偏小，精度不算漂亮；
4. 若要继续冲更高把握，下一步仍建议追加更多公开道路场景样本，或引入更贴题的专用七类数据集继续重训。
