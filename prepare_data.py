#!/usr/bin/env python3
"""准备 ArmorPose 训练数据集。

将原始 data_dir 下的图片和标签按比例切分为 train/val，输出到 out_dir。
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

# ── 配置 ────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).resolve().parent / "full"
OUT_DIR     = Path(__file__).resolve().parent / "datasets" / "armor"
TRAIN_RATIO = 0.8
SEED        = 42

# ── 主流程 ──────────────────────────────────────────────────────────────
def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    # 收集 (图片, 标签) 对
    pairs = []
    for img in sorted(DATA_DIR.rglob("images/**/*.jpg")):
        rel = img.relative_to(DATA_DIR / "images")
        lbl = DATA_DIR / "labels" / rel.with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))

    random.seed(SEED)
    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]

    print(f"总样本: {len(pairs)}  →  train: {len(train_pairs)}  val: {len(val_pairs)}")

    # 复制图片（软链接），标签预处理为标准 17 列格式（剥离末尾 color/type 字符串）
    def link(pairs_list, split):
        for img, lbl in pairs_list:
            d_i = OUT_DIR / "images" / split / img.name
            d_l = OUT_DIR / "labels" / split / lbl.name
            d_i.parent.mkdir(parents=True, exist_ok=True)
            d_l.parent.mkdir(parents=True, exist_ok=True)
            d_i.symlink_to(img.resolve())
            # 读取原始标签，只保留前 17 列数值
            lines = []
            for line in lbl.read_text().strip().splitlines():
                tokens = line.strip().split()
                if len(tokens) >= 17:
                    lines.append(" ".join(tokens[:17]))
            d_l.write_text("\n".join(lines) + "\n")

    link(train_pairs, "train")
    link(val_pairs,   "val")

    # 写 data.yaml
    yaml = OUT_DIR / "armor.yaml"
    yaml.write_text(f"""# Armor keypoint detection dataset (auto-generated)
path: {OUT_DIR}
train: images/train
val: images/val
kpt_shape: [4, 3]
names:
  0: Hero_Red
  1: Hero_Blue
  2: Hero_None
  3: Hero_Purple
  4: Engineer_Red
  5: Engineer_Blue
  6: Engineer_None
  7: Engineer_Purple
  8: Three_Red
  9: Three_Blue
  10: Three_None
  11: Three_Purple
  12: Four_Red
  13: Four_Blue
  14: Four_None
  15: Four_Purple
  16: Five_Red
  17: Five_Blue
  18: Five_None
  19: Five_Purple
  20: Outpost_Red
  21: Outpost_Blue
  22: Outpost_None
  23: Outpost_Purple
  24: Sentry_Red
  25: Sentry_Blue
  26: Sentry_None
  27: Sentry_Purple
  28: SmallBase_Red
  29: SmallBase_Blue
  30: SmallBase_None
  31: SmallBase_Purple
  32: BigBase_Red
  33: BigBase_Blue
  34: BigBase_None
  35: BigBase_Purple
""")
    print(f"数据集已生成: {OUT_DIR}")


if __name__ == "__main__":
    main()
