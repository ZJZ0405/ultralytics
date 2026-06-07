#!/usr/bin/env python3
"""ArmorPose 训练脚本 — 基于标准 ultralytics pose 任务，最小侵入。

用法:
    python train.py              # 从头训练
    python train.py --resume     # 自动找 last.pt 续训
    python train.py --resume --checkpoint path/to/last.pt  # 指定 checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

# ── 配置 ────────────────────────────────────────────────────────────────
DATA_YAML = Path(__file__).resolve().parent / "datasets" / "armor" / "armor.yaml"
MODEL_CFG = "ultralytics/cfg/models/armor/armor-pose.yaml"

EPOCHS  = 100
IMGSZ   = 640
BATCH   = 32
DEVICE  = 0
WORKERS = 4

# ── 主流程 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArmorPose 训练")
    parser.add_argument("--resume", action="store_true", help="断点续训")
    parser.add_argument("--checkpoint", type=str, default=None, help="续训 checkpoint 路径")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        print("数据集不存在，先运行 python prepare_data.py")
        sys.exit(1)

    # 续训：加载 checkpoint；否则从 YAML 构建
    checkpoint = args.checkpoint
    if args.resume and checkpoint is None:
        from ultralytics.utils.checks import check_file
        from ultralytics.engine.trainer import get_latest_run
        try:
            checkpoint = str(get_latest_run())
            print(f"找到 checkpoint: {checkpoint}")
        except Exception:
            print("找不到可续训的 checkpoint，将从头训练")
            args.resume = False

    if args.resume and checkpoint:
        model = YOLO(checkpoint)
    else:
        model = YOLO(MODEL_CFG)

    # 训练启动后替换为 ArmorPoseModel + ArmorPoseLoss，并注入 loss 名
    def _setup_armor(trainer):
        from ultralytics.nn.tasks import ArmorPoseModel
        from ultralytics.utils.torch_utils import unwrap_model
        m = unwrap_model(trainer.model)
        if not isinstance(m, ArmorPoseModel):
            m.__class__ = ArmorPoseModel
        m.set_armor_loss()
        if "color_loss" not in trainer.loss_names:
            trainer.loss_names += ("color_loss", "type_loss")

    model.add_callback("on_pretrain_routine_end", _setup_armor)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project="runs/armor-pose",
        name="train",
        patience=30,
        save=True,
        save_period=10,
        val=True,
        # 数据增强：关闭 HSV（保留颜色），几何增强开满
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=90.0,
        shear=10.0,
        scale=0.9,
        translate=0.2,
        perspective=0.001,
        fliplr=0.5,
        # loss 增益
        pose=30.0,  # 关键点回归（默认12.0，提高以强化pose精度）
        kobj=3.0,   # 关键点可见性（默认1.0）
        color=1.0,  # 颜色分类（默认0.5）
        type=1.0,   # 装甲类型分类（默认0.5）
    )
