#!/usr/bin/env python3
"""
Convert a COCO-format armor dataset to YOLO pose format for ArmorPose model training.

The ArmorPose model requires each label line to have 19 columns:
    class_id x_center y_center width height \
    kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis kp3_x kp3_y kp3_vis kp4_x kp4_y kp4_vis \
    color_id label_id

where color_id and label_id come from annotation["attributes"]["color"] and ["label"]
in the COCO JSON.

Usage:
    python scripts/coco2yolo_armorpose.py \
        --json /path/to/coco_annotations/ \
        --save-dir /path/to/output/ \
        --images-dir /path/to/images/ \
        --color-names red blue green yellow white black gray other \
        --label-names 1 2 3 4 5 6 7 8 9 10

    # Or process a single JSON file:
    python scripts/coco2yolo_armorpose.py \
        --json /path/to/annotations.json \
        --save-dir ./armorpose_dataset/
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


def extract_color_label(ann: dict) -> "tuple[int | None, int | None]":
    """Extract color and label class indices from a COCO annotation's attributes.

    Args:
        ann: A COCO annotation dict.

    Returns:
        (color_id, label_id) tuple. Each is None if not found.
    """
    attrs = ann.get("attributes", {})
    color = attrs.get("color")
    label = attrs.get("label")

    # Validate types
    if color is not None:
        try:
            color = int(color)
        except (ValueError, TypeError):
            print(f"  [WARN] Invalid color value '{color}' in annotation id={ann.get('id')}, ignoring.")
            color = None
    if label is not None:
        try:
            label = int(label)
        except (ValueError, TypeError):
            print(f"  [WARN] Invalid label value '{label}' in annotation id={ann.get('id')}, ignoring.")
            label = None

    return color, label


def convert_coco_armorpose(
    json_path: str,
    save_dir: str,
    images_dir: str = None,
    color_names: list = None,
    label_names: list = None,
    nc_color: int = 8,
    nc_label: int = 10,
    kpt_shape: tuple = (4, 3),
    copy_images: bool = False,
    symlink_images: bool = True,
) -> str:
    """Convert COCO-format armor dataset to YOLO ArmorPose format.

    Args:
        json_path: Path to a COCO JSON file or a directory of JSON files.
        save_dir: Output directory root.
        images_dir: Path to the directory containing source images.
                    If None, images are expected at {save_dir}/images/{split}/.
        color_names: List of color class names (e.g. ['red', 'blue', ...]).
        label_names: List of label class names (e.g. ['1', '2', ...]).
        nc_color: Number of color classes (used if color_names not provided).
        nc_label: Number of label classes (used if label_names not provided).
        kpt_shape: Keypoint shape as (num_keypoints, dims).
        copy_images: If True, copy images to output dir.
        symlink_images: If True, create symlinks to images in output dir.

    Returns:
        Path to the generated data.yaml file.
    """
    json_path = Path(json_path)
    save_dir = Path(save_dir)

    # --- Collect JSON files ---
    if json_path.is_dir():
        json_files = sorted(json_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in {json_path}")
    else:
        json_files = [json_path]

    # --- Resolve color/label names ---
    if color_names is None:
        color_names = [f"color_{i}" for i in range(nc_color)]
    else:
        nc_color = len(color_names)
    if label_names is None:
        label_names = [f"label_{i}" for i in range(nc_label)]
    else:
        nc_label = len(label_names)

    nk = kpt_shape[0] * kpt_shape[1]  # total keypoint values (e.g., 4*3=12)

    # --- Collect all category names from all JSONs ---
    all_category_names: dict[int, str] = {}
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            cat_id = int(cat["id"])
            if cat_id not in all_category_names:
                all_category_names[cat_id] = cat.get("name", f"class_{cat_id}")

    # Build a mapping from COCO category_id to 0-based class index
    sorted_cat_ids = sorted(all_category_names.keys())
    coco_id_to_cls: dict[int, int] = {cid: i for i, cid in enumerate(sorted_cat_ids)}
    class_names: dict[int, str] = {i: all_category_names[cid] for i, cid in enumerate(sorted_cat_ids)}

    # --- Process each JSON file ---
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        # Determine split name from filename
        stem_lower = jf.stem.lower()
        if "train" in stem_lower:
            split = "train"
        elif "val" in stem_lower or "valid" in stem_lower:
            split = "val"
        elif "test" in stem_lower:
            split = "test"
        else:
            split = "train"  # default

        # Create output directories
        label_out = save_dir / "labels" / split
        image_out = save_dir / "images" / split
        label_out.mkdir(parents=True, exist_ok=True)
        image_out.mkdir(parents=True, exist_ok=True)

        # Build image lookup
        images = {img["id"]: img for img in data["images"]}

        # Build image_id -> annotations mapping
        img_anns = defaultdict(list)
        for ann in data["annotations"]:
            img_anns[ann["image_id"]].append(ann)

        # Statistics
        total_anns = 0
        skipped_no_kpt = 0
        skipped_no_color = 0
        skipped_no_label = 0
        skipped_bad_box = 0

        print(f"\nProcessing: {jf.name}  ({len(img_anns)} images) -> {split} split")

        for img_id, anns in tqdm(img_anns.items(), desc=f"  {split}", unit="img"):
            img = images.get(img_id)
            if img is None:
                continue
            h, w = img["height"], img["width"]
            img_filename = img["file_name"]

            # Handle image
            if images_dir:
                src_img = Path(images_dir) / img_filename
            else:
                src_img = Path(img_filename)  # relative, may not exist

            dst_img = image_out / Path(img_filename).name

            if not dst_img.exists():
                if symlink_images and src_img.exists():
                    try:
                        dst_img.symlink_to(src_img.resolve())
                    except OSError:
                        shutil.copy2(src_img, dst_img)
                elif copy_images and src_img.exists():
                    shutil.copy2(src_img, dst_img)

            # Build label lines
            lines = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue

                # --- Bbox: COCO [x_tl, y_tl, w, h] -> YOLO [x_c, y_c, w, h] normalized ---
                box = np.array(ann["bbox"], dtype=np.float64)
                box[0] += box[2] / 2.0  # x_tl -> x_center
                box[1] += box[3] / 2.0  # y_tl -> y_center
                box[0] /= w
                box[2] /= w
                box[1] /= h
                box[3] /= h

                if box[2] <= 0 or box[3] <= 0:
                    skipped_bad_box += 1
                    continue

                # --- Class ---
                coco_cat_id = ann["category_id"]
                cls = coco_id_to_cls.get(coco_cat_id, 0)

                # --- Keypoints ---
                kpts_raw = ann.get("keypoints")
                if kpts_raw is None or len(kpts_raw) == 0:
                    skipped_no_kpt += 1
                    continue

                kpts_arr = np.array(kpts_raw, dtype=np.float64).reshape(-1, 3)
                # Normalize x/w, y/h, keep visibility as-is
                kpts_arr[:, 0] /= w
                kpts_arr[:, 1] /= h
                kpts_flat = kpts_arr.reshape(-1).tolist()

                # --- Color & Label ---
                color_id, label_id = extract_color_label(ann)

                if color_id is None:
                    skipped_no_color += 1
                    color_id = -1  # placeholder
                if label_id is None:
                    skipped_no_label += 1
                    label_id = -1  # placeholder

                # --- Assemble line ---
                line = [cls, *box.tolist(), *kpts_flat, color_id, label_id]
                lines.append(line)
                total_anns += 1

            # Write label file
            if lines:
                txt_path = label_out / f"{Path(img_filename).stem}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    for line_vals in lines:
                        # Format: use "%.6g" for floats, "%d" for integers
                        formatted = []
                        for i, val in enumerate(line_vals):
                            if i == 0 or i == len(line_vals) - 2 or i == len(line_vals) - 1:
                                # class_id (0), color_id (17), label_id (18) -> int
                                formatted.append(str(int(val)))
                            else:
                                formatted.append(f"{float(val):.6g}")
                        f.write(" ".join(formatted) + "\n")

        # --- Summary ---
        print(f"  ✓ {total_anns} labels written to {label_out}")
        if skipped_no_kpt:
            print(f"  ⚠ {skipped_no_kpt} annotations skipped (missing keypoints)")
        if skipped_bad_box:
            print(f"  ⚠ {skipped_bad_box} annotations skipped (invalid bbox)")
        if skipped_no_color:
            print(f"  ⚠ {skipped_no_color} annotations missing color attribute (set to -1)")
        if skipped_no_label:
            print(f"  ⚠ {skipped_no_label} annotations missing label attribute (set to -1)")

    # --- Generate data.yaml ---
    yaml_path = save_dir / "data.yaml"
    data_config = {
        "path": str(save_dir.resolve()),
        "train": "images/train" if (save_dir / "images" / "train").exists() else None,
        "val": "images/val" if (save_dir / "images" / "val").exists() else None,
        "test": "images/test" if (save_dir / "images" / "test").exists() else None,
        "kpt_shape": list(kpt_shape),
        "names": class_names,
        "nc_color": nc_color,
        "color_names": {i: name for i, name in enumerate(color_names)},
        "nc_label": nc_label,
        "label_names": {i: name for i, name in enumerate(label_names)},
    }

    # Remove None values
    data_config = {k: v for k, v in data_config.items() if v is not None}

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n✅ Dataset YAML saved to: {yaml_path.resolve()}")
    print(f"   Detection classes: {len(class_names)}")
    print(f"   Color classes:     {nc_color}")
    print(f"   Label classes:     {nc_label}")
    print(f"   Keypoint shape:    {list(kpt_shape)}")

    return str(yaml_path.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO armor dataset to YOLO ArmorPose format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to a COCO JSON file or a directory containing JSON files.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Output directory root (will create images/ and labels/ subdirs).",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Path to source images directory. If not set, images are NOT copied.",
    )
    parser.add_argument(
        "--color-names",
        nargs="+",
        default=None,
        help="Color class names, space-separated. Default: color_0 color_1 ...",
    )
    parser.add_argument(
        "--label-names",
        nargs="+",
        default=None,
        help="Label class names, space-separated. Default: label_0 label_1 ...",
    )
    parser.add_argument(
        "--nc-color",
        type=int,
        default=8,
        help="Number of color classes (default: 8). Ignored if --color-names is set.",
    )
    parser.add_argument(
        "--nc-label",
        type=int,
        default=10,
        help="Number of label classes (default: 10). Ignored if --label-names is set.",
    )
    parser.add_argument(
        "--kpt-shape",
        type=int,
        nargs=2,
        default=[4, 3],
        metavar=("N_KPT", "DIM"),
        help="Keypoint shape as 'n_kpts dims' (default: 4 3).",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images to the output directory.",
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Disable symlinking images (only relevant with --images-dir).",
    )

    args = parser.parse_args()

    convert_coco_armorpose(
        json_path=args.json,
        save_dir=args.save_dir,
        images_dir=args.images_dir,
        color_names=args.color_names,
        label_names=args.label_names,
        nc_color=args.nc_color,
        nc_label=args.nc_label,
        kpt_shape=tuple(args.kpt_shape),
        copy_images=args.copy_images,
        symlink_images=not args.no_symlink,
    )


if __name__ == "__main__":
    main()
