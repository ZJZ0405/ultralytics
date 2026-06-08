# ArmorPose 自研训练框架 — 任务列表

## 目标

替代 ultralytics，实现干净可控的装甲板姿态估计训练流程：
- 自定义 head (color_head + type_head 双路分类 + cv4 关键点)
- 自定义 loss (关键点回归 + color 分类 + type 分类)
- 标准 YOLO pose 格式数据 (17 列，class_id 编码 type+color)
- 训练 / 验证 / 推理全流程

---

## Phase 1 — 数据管线

- [ ] **1.1 YOLO pose 数据集解析**
  - 读取 data.yaml (path, train, val, names, kpt_shape)
  - 解析 .txt 标签 → (class_id, bbox, keypoints)

- [ ] **1.2 数据增强 pipeline**
  - Mosaic (前 90% epoch)
  - 几何增强 (旋转 ±90°、缩放 ±90%、平移 ±20%、透视、水平翻转)
  - HSV 关闭 (保留原始颜色)

- [ ] **1.3 DataLoader**
  - 多进程加载 + collate_fn
  - 矩形训练 (rect=True) 可选

---

## Phase 2 — 模型

- [ ] **2.1 Backbone + Neck**
  - 从 ultralytics 提取 YOLOv8 backbone/neck 权重
  - 或者手写轻量 CSPDarknet + PAN-FPN

- [ ] **2.2 ArmorPose Head**
  - cv2: box regression (4×reg_max)
  - cv4: keypoint prediction (nk = kpt_shape[0]×kpt_shape[1])
  - color_head: 4 类颜色分类
  - type_head: 9 类装甲类型分类
  - 36-way scores = color_logit ⊗ type_logit (outer sum)

- [ ] **2.3 权重加载**
  - 从 ultralytics .pt checkpoint 提取 backbone/neck 权重
  - head 随机初始化

---

## Phase 3 — Loss

- [ ] **3.1 Task-Aligned Assigner (TAL)**
  - 基于 alignment metric 分配 anchor → GT
  - 返回 fg_mask, target_gt_idx, target_bboxes, target_scores

- [ ] **3.2 Detection Loss**
  - box_loss (CIoU)
  - dfl_loss (Distribution Focal Loss)

- [ ] **3.3 Keypoint Loss**
  - kpt_loc_loss (OKS-based KeypointLoss)
  - kpt_vis_loss (BCEWithLogits)

- [ ] **3.4 Classification Loss**
  - cls_loss = 0 (被 color + type 替代)
  - color_loss (CrossEntropy, 4 类)
  - type_loss (CrossEntropy, 9 类)
  - target 从 batch_cls[target_gt_idx] 推导: color = cls % 4, type = cls // 4

- [ ] **3.5 增益控制**
  - box=7.5, pose=30.0, kobj=3.0, color=1.0, type=1.0, dfl=1.5
  - 支持命令行 / yaml 覆盖

---

## Phase 4 — 训练循环

- [ ] **4.1 训练引擎**
  - AMP 混合精度
  - EMA 指数滑动平均
  - Cosine / linear warmup lr schedule
  - Gradient clipping

- [ ] **4.2 Checkpoint**
  - 保存 last.pt (model + optimizer + epoch)
  - 保存 best.pt (按 val metric)
  - 支持断点续训 (--resume)

- [ ] **4.3 日志**
  - tqdm 进度条 (loss 实时)
  - TensorBoard / wandb (可选)
  - 每 N epoch 保存训练样本可视化

---

## Phase 5 — 验证 & 推理

- [ ] **5.1 验证循环**
  - mAP 计算 (Box + Pose)
  - 按 class 分项统计
  - COCO JSON 导出 (用于外部评估)

- [ ] **5.2 推理**
  - ONNX / TensorRT 导出
  - 单图 / 视频推理
  - 可视化 (bbox + keypoints + color/type 标签)

---

## Phase 6 — 工程

- [ ] **6.1 配置管理**
  - YAML 配置文件
  - 命令行 override

- [ ] **6.2 测试**
  - 数据加载单测
  - loss 数值校验
  - 小数据集过拟合测试

---

## 预估工作量

| Phase | 内容 | 预估行数 | 耗时 |
|-------|------|---------|------|
| 1 | 数据管线 | ~300 行 | 0.5d |
| 2 | 模型 | ~200 行 | 0.5d |
| 3 | Loss | ~400 行 | 1d |
| 4 | 训练循环 | ~300 行 | 0.5d |
| 5 | 验证推理 | ~200 行 | 0.5d |
| 6 | 工程 | ~100 行 | 0.5d |
| **总计** | | **~1500 行** | **3.5d** |

对比：现在跟 ultralytics 斗智斗勇已经花了 ~2 天，还留下一堆 hack。
