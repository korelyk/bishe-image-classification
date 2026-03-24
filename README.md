# 基于深度学习的图片分类系统

这是一个可直接运行、可部署、可用于本科毕业设计答辩演示的完整项目成品，主题为**自动驾驶道路目标七分类**。

## 一、项目简介

系统面向道路交通场景图片，提供以下核心能力：

- 用户端图片上传
- 深度学习推理识别
- 分类结果与候选标签展示
- 历史记录查询
- 后台统计与最近预测管理
- SQLite 数据留痕
- 服务器部署与 systemd 托管
- 论文、摘要、答辩问答、进度说明等文档产物

## 二、工程化实现说明

原任务书/开题报告给出的路线是“小程序 + Spring Boot + Vue3 + MySQL + PyTorch + ResNet50”。

为了在有限周期内交付**真正能运行、能部署、能演示、能写论文、能答辩**的本科毕设成品，本项目采用了更稳妥的工程化替代方案：

- 前后端一体：**FastAPI + Jinja2 + 原生 JS**
- 推理引擎：**ONNX Runtime + MobileNetV2 预训练模型**
- 类别策略：**ImageNet 标签映射到道路七类目标**
- 存储：**SQLite**
- 部署：**systemd + uvicorn**

这样做的原因：

1. 缩短工程搭建时间，提高可交付性；
2. 保留深度学习模型、前台页面、后台管理、数据库留痕、部署验证等毕设关键要素；
3. 避开服务器磁盘空间受限导致的重型 CUDA 依赖安装问题；
4. 论文中可以合理解释为“在毕设周期内采用轻量化深度学习推理方案完成工程落地，并保留后续升级空间”。

## 三、功能清单

### 1. 用户端

访问 `/` 页面可实现：

- 上传道路场景图片
- 查看主分类类别
- 查看置信度
- 查看模型推理方式
- 查看 Top-5 候选标签明细
- 查看最近识别历史记录

### 2. 管理端

访问 `/admin` 页面可实现：

- 查看总识别次数
- 查看最近识别时间
- 查看类别分布
- 查看最近 10 条识别记录
- 支持管理员口令保护（可选）

### 3. 深度学习推理能力

本系统针对“道路七类目标”进行了统一映射：

- 行人 pedestrian
- 自行车 bicycle
- 汽车 car
- 摩托车 motorcycle
- 公交车 bus
- 卡车 truck
- 交通标志/信号灯 traffic_sign_light

推理策略：

1. 使用 MobileNetV2 ONNX 预训练分类模型完成整图推理；
2. 读取 ImageNet Top-5 候选标签；
3. 将候选标签映射到毕业设计要求的七类道路目标；
4. 保存预测记录、模型模式、推理时间点与图片路径。

## 四、项目结构

```text
bishe-image-classification/
├── app/
│   ├── db.py
│   ├── main.py
│   ├── model_engine.py
│   ├── static/
│   │   ├── admin.js
│   │   ├── app.js
│   │   └── style.css
│   └── templates/
│       ├── admin.html
│       └── index.html
├── data/
│   ├── annotated/
│   ├── models/
│   ├── uploads/
│   └── app.db
├── docs/
│   ├── abstract.md
│   ├── defense_qa.md
│   ├── progress_report.md
│   ├── thesis_full.md
│   ├── thesis_outline.md
│   └── requirement-summary.md
├── scripts/
│   ├── deploy_remote.sh
│   └── smoke_test.sh
├── requirements.txt
└── README.md
```

## 五、本地运行

### 1. 创建虚拟环境

> 兼容说明：当前依赖已按服务器 Python 3.9 环境锁定，`onnxruntime` 使用 `1.19.2`，可避免 3.9 环境下安装失败。

```bash
cd bishe-image-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 19001
```

### 3. 访问地址

- 用户端：`http://127.0.0.1:19001`
- 管理端：`http://127.0.0.1:19001/admin`
- 健康检查：`http://127.0.0.1:19001/api/health`

## 六、环境变量

| 变量名 | 说明 | 默认值 |
|---|---|---|
| `ADMIN_PASSWORD` | 管理端 API 口令，可选 | 空 |

设置示例：

```bash
export ADMIN_PASSWORD='road-admin-2026'
```

## 七、接口说明

### 1. 健康检查

```http
GET /api/health
```

### 2. 图片识别

```http
POST /api/classify
Content-Type: multipart/form-data
字段：file
```

### 3. 历史记录

```http
GET /api/history?limit=12
```

### 4. 后台统计

```http
GET /api/admin/stats
Header: X-Admin-Password: <password>
```

## 八、部署说明

项目提供远程部署脚本：

```bash
REMOTE_PASSWORD='服务器密码' bash scripts/deploy_remote.sh
```

默认部署信息：

- 部署目录：`/opt/bishe-image-classification`
- systemd 服务名：`bishe-image-classification.service`
- 服务端口：`19001`

## 九、验证脚本

```bash
bash scripts/smoke_test.sh http://127.0.0.1:19001
```

## 十、论文与答辩材料

已在 `docs/` 目录生成：

- `thesis_outline.md`
- `thesis_full.md`
- `abstract.md`
- `defense_qa.md`
- `progress_report.md`

## 十一、适合作为毕业设计答辩的亮点

1. 深度学习能力真实可运行；
2. 具有完整前台、后台、数据库、部署链路；
3. 可展示工程化折中与技术选型思路；
4. 可清楚说明七类道路目标映射逻辑；
5. 可结合论文说明“原计划路线”与“实际落地路线”的差异及合理性。

## 十二、注意事项

- 首次推理会自动下载 ONNX 模型与标签文件到 `data/models/`；
- CPU 环境下推理速度会慢于 GPU，但部署体积更轻；
- 本项目强调毕设演示与工程可交付性，不宣称达到工业级自动驾驶实时性能；
- 若未来继续扩展，可替换为自训练 YOLO / ResNet、接入 MySQL、Vue 前端或微信小程序。
