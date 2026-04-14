# 基于深度学习的图片分类系统（升级版）

这是一个基于深度学习的**道路目标图片分类系统升级版**。本次升级重点补充了系统功能、实验设计与结果分析中较为关键的四类内容：

1. **多模型对比**：ResNet50 / MobileNetV2 / EfficientNet-B0
2. **数据增强训练**：旋转、翻转、裁剪、噪声、低照度扰动
3. **Grad-CAM 可视化**：展示模型关注区域
4. **复杂测试场景**：低光照、噪声、局部裁剪、实时拍照、批量识别

并且进一步补上了最关键的一点：**公开数据来源与可复现训练链路**。

---

## 一、当前系统能力

### 1. 在线功能

访问 `/` 页面后可直接使用：

- 单图上传识别
- 模型切换（ResNet50 / MobileNetV2 / EfficientNet-B0）
- Grad-CAM 热力图生成
- 浏览器实时拍照识别
- 多图片批量识别
- 最近识别历史展示
- 离线实验报告与模型对比表展示

### 2. 后台能力

访问 `/admin` 页面可查看：

- 总识别次数
- 类别分布
- 最近识别记录
- 数据删除管理

### 3. 论文与实验支撑材料

已在仓库内生成：

- `docs/model_comparison.md`：模型对比表
- `docs/experiment_results.md`：实验结果摘要
- `docs/upgrade_report.md`：本次升级说明
- `data/reports/model_comparison.json`：前端读取的对比结果
- `data/reports/robustness_report.json`：复杂场景鲁棒性结果
- `data/reports/training_summary.json`：训练摘要
- `data/experiments/dataset_manifest.json`：公开数据来源与标注追溯清单
- `data/reports/gradcam_gallery.json`：Grad-CAM 样例清单

---

## 二、技术路线

### 在线识别系统

- 后端：FastAPI
- 前端：Jinja2 + 原生 JS
- 模型：Torchvision 预训练模型
- 数据库：SQLite
- 图像处理：Pillow
- 深度学习框架：PyTorch / Torchvision

### 训练与实验部分

- 公开数据集：基于 **COCO 2017 val** 标注裁剪构建的 `coco-road7-crops`
- 实验类别：`person / bicycle / car / motorcycle / bus / truck / traffic_light`
- 实验模型：ResNet50、MobileNetV2、EfficientNet-B0
- 训练策略：迁移学习 + 冻结骨干网络 + 微调分类头
- 增强方式：随机裁剪、翻转、旋转、亮度对比度变化、高斯噪声、低照度模拟

> 说明：当前这版已经不再只是仓库样例图扩增，而是接入了公开可追溯的数据来源。若后续拿到更贴近学校题目的专用七类数据集，可直接沿用同一套脚本继续训练。

---

## 三、已生成的实验结果

### 1. 主实验结果（CPU 环境）

| 模型 | 参数量(M) | 类别数 | 验证准确率 | 测试准确率 | 单张耗时(ms) | FPS | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| ResNet50 | 23.52 | 7 | 33.93% | 35.71% | 153.46 | 6.52 | 当前精度最好，但速度最慢 |
| MobileNetV2 | 2.23 | 7 | 46.43% | 33.93% | 39.59 | 25.26 | 轻量化明显，速度最快 |
| EfficientNet-B0 | 4.02 | 7 | 16.07% | 26.79% | 54.06 | 18.50 | 当前在该公开小样本集上表现一般 |

### 2. 复杂场景鲁棒性

- `gaussian_noise`：ResNet50 最稳
- `low_light`：ResNet50 相对更稳
- `partial_crop`：MobileNetV2 更稳

详细数据见：`docs/model_comparison.md`

---

## 四、项目结构

```text
bishe-image-classification/
├── app/
│   ├── db.py
│   ├── main.py
│   ├── model_engine.py
│   ├── vision_models.py
│   ├── static/
│   └── templates/
├── data/
│   ├── annotated/
│   ├── demo/
│   ├── experiments/
│   ├── external/
│   ├── reports/
│   ├── uploads/
│   └── app.db
├── docs/
│   ├── experiment_results.md
│   ├── model_comparison.md
│   └── upgrade_report.md
├── scripts/
│   ├── benchmark_models.py
│   ├── build_demo_dataset.py
│   ├── generate_gradcam_gallery.py
│   ├── run_all_experiments.sh
│   └── train_models.py
├── requirements.txt
└── README.md
```

---

## 五、本地运行

### 1. 安装依赖

```bash
cd bishe-image-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2. 启动系统

```bash
uvicorn app.main:app --host 0.0.0.0 --port 19001
```

访问地址：

- 用户端：`http://127.0.0.1:19001`
- 管理端：`http://127.0.0.1:19001/admin`
- 健康检查：`http://127.0.0.1:19001/api/health`

---

## 六、离线实验运行方式

### 方案一：一步跑完

```bash
bash scripts/run_all_experiments.sh
```

### 方案二：分步执行

```bash
source .venv/bin/activate
python scripts/build_demo_dataset.py
python scripts/train_models.py --epochs 8 --batch-size 8
python scripts/benchmark_models.py
python scripts/generate_gradcam_gallery.py
```

生成产物：

- `data/experiments/demo_dataset/`：训练/验证/测试集
- `data/experiments/checkpoints/*.pt`：模型权重
- `data/reports/*.json`：实验报告
- `data/annotated/*gradcam.jpg`：Grad-CAM 热力图

---

## 七、接口说明

### 1. 健康检查

```http
GET /api/health
```

### 2. 单图识别

```http
POST /api/classify
Content-Type: multipart/form-data
字段：
- file
- model_name=resnet50|mobilenet_v2|efficientnet_b0
- with_gradcam=true|false
```

### 3. 批量识别

```http
POST /api/classify/batch
Content-Type: multipart/form-data
字段：
- files（多文件）
- model_name
- with_gradcam
```

### 4. 历史记录

```http
GET /api/history?limit=12
```

### 5. 对比实验报告

```http
GET /api/reports/model-comparison
GET /api/reports/robustness
```
