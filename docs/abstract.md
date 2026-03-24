# 摘要

本文围绕本科毕业设计题目《基于深度学习的图片分类系统》，设计并实现了一套面向自动驾驶道路场景的七类目标图片分类系统。系统以道路交通图片为输入，围绕行人、自行车、汽车、摩托车、公交车、卡车以及交通标志/信号灯七类目标构建完整业务流程。系统采用 FastAPI 作为后端服务框架，结合 Jinja2 模板引擎与原生 JavaScript 实现用户端页面和管理端页面；在模型层，采用 ONNX Runtime 加载 MobileNetV2 预训练分类模型，并通过 ImageNet 标签映射策略将通用分类结果转换为符合课题要求的七类道路目标。系统还使用 SQLite 对识别记录进行持久化存储，实现图片路径、预测类别、置信度、模型模式、创建时间等信息的完整留痕，并提供后台统计、历史查询和健康检查等功能。

在工程实现过程中，考虑到部署服务器磁盘空间有限、毕设周期紧张以及项目必须真正上线运行等现实约束，本文没有继续采用重量级训练框架，而是将深度学习推理方案优化为轻量化 ONNX 部署形态。这一方案既保留了深度学习模型推理的核心特征，又显著降低了依赖体积和部署复杂度，使系统能够以较低资源消耗完成真实落地。实践结果表明，该系统能够稳定完成图片上传、分类推理、结果展示、历史记录管理、后台统计和服务器部署等关键功能，满足毕业设计演示与答辩的基本要求。

**关键词：** 深度学习；图片分类；自动驾驶；ONNX Runtime；FastAPI；系统部署

---

# Abstract

This thesis designs and implements a deep-learning-based image classification system for road scenes in autonomous driving. The system accepts traffic-related images as input and maps the inference results into seven target classes: pedestrian, bicycle, car, motorcycle, bus, truck, and traffic sign/light. FastAPI is adopted as the backend framework, while Jinja2 and native JavaScript are used to build the user and admin pages. In the model layer, a pretrained MobileNetV2 model is deployed through ONNX Runtime, and its ImageNet top predictions are mapped to the seven road-related categories required by the graduation project. SQLite is used to store prediction records, including filenames, image paths, predicted labels, confidence scores, model modes, and timestamps.

To ensure real deployment and delivery within the undergraduate project schedule, the implementation adopts a lightweight deployment-oriented deep learning solution instead of a heavyweight training stack. This reduces resource consumption and dependency size while retaining the essential characteristics of deep learning inference. Experimental deployment and functional verification show that the system can stably support image upload, classification inference, history query, admin statistics, and remote server deployment.

**Key Words:** deep learning; image classification; autonomous driving; ONNX Runtime; FastAPI; deployment
