# YOLOv8n-Pose Training and Validation Flow in Ultralytics

This note is a code-grounded handoff for other agents. It maps what happens in this Ultralytics repo when selecting `yolov8n-pose` for training, including data flow, augmentation, trainer/optimizer configuration, model, loss, and validation settings.

Repository root used for this analysis: `/home/violet/workspace/ultralytics`.

## 1. Entrypoint and Task Routing

Typical usage:

```bash
yolo pose train model=yolov8n-pose.pt data=coco-pose.yaml imgsz=640
```

or Python:

```python
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
model.train(data="coco-pose.yaml")
```

Pose training uses `PoseTrainer` in `ultralytics/models/yolo/pose/train.py`. `PoseTrainer` inherits `DetectionTrainer` and forces:

```python
overrides["task"] = "pose"
```

Main class chain:

1. `BaseTrainer` in `ultralytics/engine/trainer.py`: generic train loop, optimizer, scheduler, AMP, EMA, checkpointing, validation dispatch.
2. `DetectionTrainer` in `ultralytics/models/yolo/detect/train.py`: YOLO dataset/dataloader/preprocess/model conventions.
3. `PoseTrainer` in `ultralytics/models/yolo/pose/train.py`: pose-specific model, validator, and loss names.

Important methods:

- `PoseTrainer.__init__`: forces `task="pose"`.
- `PoseTrainer.get_model`: builds `PoseModel`.
- `PoseTrainer.set_model_attributes`: sets `kpt_shape` and `kpt_names` on the model.
- `PoseTrainer.get_validator`: returns `PoseValidator` and sets loss names.
- `PoseTrainer.get_dataset`: requires `kpt_shape` in dataset YAML.

## 2. Dataset Configuration: `coco-pose.yaml`

File: `ultralytics/cfg/datasets/coco-pose.yaml`

Key fields:

```yaml
path: coco-pose
train: train2017.txt
val: val2017.txt
test: test-dev2017.txt

kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person
```

For COCO pose:

- `nc=1`
- class name is `person`
- 17 keypoints per object
- each keypoint has 3 dimensions: `x, y, visible`
- `flip_idx` swaps left/right keypoints during flip augmentation

`PoseTrainer.get_dataset()` raises if `kpt_shape` is absent.

## 3. Training Data Flow

High-level call chain:

```text
BaseTrainer._setup_train()
  -> BaseTrainer._build_train_pipeline()
    -> DetectionTrainer.get_dataloader()
      -> DetectionTrainer.build_dataset()
        -> build_yolo_dataset()
          -> YOLODataset(...)
            -> BaseDataset.get_img_files()
            -> YOLODataset.get_labels()
            -> YOLODataset.build_transforms()
```

### 3.1 Dataset Construction

`DetectionTrainer.build_dataset()` computes grid stride and calls:

```python
build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
```

`build_yolo_dataset()` in `ultralytics/data/build.py` creates a `YOLODataset` with:

```python
YOLODataset(
    img_path=img_path,
    imgsz=cfg.imgsz,
    batch_size=batch,
    augment=mode == "train",
    hyp=cfg,
    rect=cfg.rect or rect,
    cache=cfg.cache or None,
    single_cls=cfg.single_cls or False,
    stride=stride,
    pad=0.0 if mode == "train" else 0.5,
    prefix=f"{mode}: ",
    task=cfg.task,
    classes=cfg.classes,
    data=data,
    fraction=cfg.fraction if mode == "train" else 1.0,
)
```

For pose, `task="pose"`, so `YOLODataset.__init__` sets:

```python
self.use_keypoints = task == "pose"
```

### 3.2 Label Reading and Validation

`YOLODataset.cache_labels()` in `ultralytics/data/dataset.py`:

- reads image files and YOLO labels
- checks `kpt_shape` when `use_keypoints=True`
- calls `verify_image_label(...)`
- stores each label as:

```python
{
    "im_file": im_file,
    "shape": shape,
    "cls": lb[:, 0:1],
    "bboxes": lb[:, 1:],
    "segments": segments,
    "keypoints": keypoint,
    "normalized": True,
    "bbox_format": "xywh",
}
```

`YOLODataset.update_labels_info()` wraps boxes, segments, and keypoints into an `Instances` object.

### 3.3 Image Loading and Resize

`BaseDataset.load_image()` in `ultralytics/data/base.py`:

- reads images with OpenCV, BGR order
- resizes the long side to `imgsz` while preserving aspect ratio by default
- supports RAM/disk caching
- maintains an image buffer during augmentation for mosaic-style transforms

## 4. Data Augmentation Flow

`YOLODataset.build_transforms()` in `ultralytics/data/dataset.py` decides the transform pipeline.

For training (`augment=True`):

```python
hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
transforms = v8_transforms(self, self.imgsz, hyp)
```

For validation (`augment=False`):

```python
Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
```

Both train and val append `Format(...)` with:

```python
return_keypoint=self.use_keypoints
```

### 4.1 YOLOv8 Training Transform Order

`v8_transforms()` in `ultralytics/data/augment.py` builds:

1. `pre_transform`
   - `Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)`
   - optional `CopyPaste(...)`
   - `RandomPerspective(...)`
2. `MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup)`
3. `CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix)`
4. `Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None))`
5. `RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v)`
6. `RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx)`
7. `RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx)`

Pose-specific flip handling:

```python
flip_idx = dataset.data.get("flip_idx", [])
if getattr(dataset, "use_keypoints", False):
    kpt_shape = dataset.data.get("kpt_shape", None)
    if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
        hyp.fliplr = hyp.flipud = 0.0
    elif flip_idx and (len(flip_idx) != kpt_shape[0]):
        raise ValueError(...)
```

So pose flip requires a valid `flip_idx`; otherwise flip augmentations are disabled.

### 4.2 Default Augmentation Hyperparameters

From `ultralytics/cfg/default.yaml`:

```yaml
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
```

Default YOLOv8n-pose behavior:

- Mosaic enabled with probability 1.0.
- MixUp disabled.
- CutMix disabled.
- CopyPaste disabled.
- HSV augmentation enabled.
- Horizontal flip probability 0.5, using `flip_idx` for keypoint remapping.
- Vertical flip disabled.
- Rotation/shear/perspective disabled.
- Translation 0.1 and scale 0.5 enabled.

### 4.3 Format Transform

`Format` in `ultralytics/data/augment.py`:

- converts image from HWC NumPy to CHW tensor
- handles BGR/RGB channel order
- converts boxes to `xywh`
- normalizes bboxes
- for pose, returns normalized `keypoints`
- adds `batch_idx`

For keypoints:

```python
labels["keypoints"] = torch.from_numpy(instances.keypoints)
if self.normalize:
    labels["keypoints"][..., 0] /= w
    labels["keypoints"][..., 1] /= h
```

## 5. Dataloader Configuration

`DetectionTrainer.get_dataloader()` in `ultralytics/models/yolo/detect/train.py`:

- train: `shuffle=True`, workers = `args.workers`
- val: `shuffle=False`, workers = `args.workers * 2`

`build_dataloader()` in `ultralytics/data/build.py` returns `InfiniteDataLoader` with:

- `batch = min(batch, len(dataset))`
- worker count bounded by CPU/GPU count
- DDP sampler when needed
- `prefetch_factor=4` when workers > 0
- CUDA `pin_memory=True`
- dataset `collate_fn`
- seeded generator and worker init

Pose batch fields commonly include:

```text
img
cls
bboxes
keypoints
batch_idx
im_file
ori_shape
resized_shape
ratio_pad
```

`YOLODataset.collate_fn()`:

- stacks `img` into `[B, C, H, W]`
- concatenates `keypoints`, `bboxes`, and `cls`
- offsets and concatenates `batch_idx`

## 6. Training Batch Preprocess

`DetectionTrainer.preprocess_batch()`:

```python
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
batch["img"] = batch["img"].float() / 255
```

If `multi_scale > 0.0`, it randomly resizes images to a stride-aligned size. Default `multi_scale: 0.0`, so this is off by default.

## 7. Model: YOLOv8n-Pose

Model YAML: `ultralytics/cfg/models/v8/yolov8-pose.yaml`

Key fields:

```yaml
nc: 1
kpt_shape: [17, 3]
scales:
  n: [0.33, 0.25, 1024]
```

For `yolov8n-pose`, scale `n` means:

- depth multiple: `0.33`
- width multiple: `0.25`
- max channels: `1024`

Architecture summary:

- Backbone: `Conv`, `C2f`, `SPPF`
- Neck/head: upsample, concat, `C2f`
- Multi-scale outputs: P3, P4, P5
- Final layer:

```yaml
[[15, 18, 21], 1, Pose, [nc, kpt_shape]]
```

### 7.1 PoseModel Construction

`PoseTrainer.get_model()` builds:

```python
model = PoseModel(
    cfg,
    nc=self.data["nc"],
    ch=self.data["channels"],
    data_kpt_shape=self.data["kpt_shape"],
)
if weights:
    model.load(weights)
```

`PoseModel` inherits `DetectionModel`. If dataset `kpt_shape` differs from the model YAML, `PoseModel.__init__` overrides the model YAML shape:

```python
if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
    cfg["kpt_shape"] = data_kpt_shape
```

`PoseTrainer.set_model_attributes()` also sets:

```python
self.model.kpt_shape = self.data["kpt_shape"]
self.model.kpt_names = kpt_names
```

## 8. Loss: YOLOv8 Pose Loss

`PoseModel.init_criterion()`:

```python
return E2ELoss(self, PoseLoss26) if getattr(self, "end2end", False) else v8PoseLoss(self)
```

YOLOv8n-pose is not end-to-end by default, so it uses `v8PoseLoss` in `ultralytics/utils/loss.py`.

### 8.1 Loss Items

`PoseTrainer.get_validator()` sets display names:

```python
("box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss")
```

`v8PoseLoss.loss()` uses:

```python
loss = torch.zeros(5)  # box, kpt_location, kpt_visibility, cls, dfl
```

Meaning:

1. `box_loss`: bbox CIoU loss from detection loss.
2. `pose_loss`: keypoint coordinate loss.
3. `kobj_loss`: keypoint visibility/objectness BCE.
4. `cls_loss`: class BCE.
5. `dfl_loss`: Distribution Focal Loss.

Default gains from `default.yaml`:

```yaml
box: 7.5
cls: 0.5
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
```

YOLOv8 pose normally uses the first five. `rle` is for the YOLO26/flow-model pose path.

### 8.2 Detection Part of Pose Loss

`v8DetectionLoss` handles box/cls/dfl and is inherited by `v8PoseLoss`.

Important defaults:

- `TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)`
- classification: `BCEWithLogitsLoss(reduction="none")`
- box: CIoU via `BboxLoss`
- DFL via `DFLoss`

### 8.3 Keypoint Loss

`v8PoseLoss.__init__`:

```python
self.kpt_shape = model.model[-1].kpt_shape
self.bce_pose = nn.BCEWithLogitsLoss()
is_pose = self.kpt_shape == [17, 3]
sigmas = OKS_SIGMA if is_pose else torch.ones(nkpt) / nkpt
self.keypoint_loss = KeypointLoss(sigmas=sigmas)
```

For COCO 17-point pose, it uses COCO OKS sigma values.

`KeypointLoss.forward()`:

```python
d = (pred_x - gt_x)^2 + (pred_y - gt_y)^2
e = d / ((2 * sigmas)^2 * area * 2)
loss = (1 - exp(-e)) * visible_mask
```

`v8PoseLoss.calculate_keypoints_loss()`:

- selects GT keypoints for each positive anchor using `target_gt_idx`
- divides selected keypoint coordinates by stride
- uses `gt_kpt[..., 2] != 0` as visibility mask when keypoints have 3 dims
- computes coordinate loss with `KeypointLoss`
- computes visibility/objectness loss with BCE:

```python
kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
```

### 8.4 Keypoint Decode

`v8PoseLoss.kpts_decode()`:

```python
y = pred_kpts.clone()
y[..., :2] *= 2.0
y[..., 0] += anchor_points[:, [0]] - 0.5
y[..., 1] += anchor_points[:, [1]] - 0.5
return y
```

Keypoint predictions are decoded relative to anchor/grid points before loss calculation.

## 9. Optimizer, Scheduler, and Training Optimization

Defaults from `ultralytics/cfg/default.yaml`:

```yaml
epochs: 100
batch: 16
imgsz: 640
workers: 8
pretrained: True
optimizer: auto
seed: 0
deterministic: True
rect: False
cos_lr: False
close_mosaic: 10
amp: True
multi_scale: 0.0
compile: False
```

### 9.1 Optimizer Construction

`BaseTrainer._build_train_pipeline()`:

```python
self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
iterations = ceil(len(train_dataset) / max(batch_size, nbs)) * epochs
self.optimizer = self.build_optimizer(...)
```

Default:

```yaml
nbs: 64
weight_decay: 0.0005
```

`BaseTrainer.build_optimizer()` supports:

```text
SGD, MuSGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto
```

For `optimizer=auto`:

```python
if iterations > 10000:
    name, lr, momentum = ("MuSGD", 0.01, 0.9)
else:
    name, lr, momentum = ("AdamW", lr_fit, 0.9)

lr_fit = round(0.002 * 5 / (4 + nc), 6)
```

For COCO pose `nc=1`, `lr_fit = 0.002` in the AdamW branch. If `optimizer=auto`, the code explicitly ignores configured `lr0` and `momentum`.

### 9.2 Parameter Groups

`build_optimizer()` groups parameters into:

1. normal weights with weight decay
2. normalization weights without weight decay
3. bias parameters without weight decay
4. MuSGD/muon group when applicable

### 9.3 Scheduler

`BaseTrainer._setup_scheduler()`:

- `cos_lr=True`: one-cycle cosine schedule
- default `cos_lr=False`: linear decay

Default:

```yaml
lrf: 0.01
```

Linear schedule:

```python
lr_lambda = max(1 - epoch / epochs, 0) * (1.0 - lrf) + lrf
```

Final LR is `lr0 * lrf`.

### 9.4 Warmup

Defaults:

```yaml
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
```

Actual warmup iterations:

```python
nw = max(round(warmup_epochs * nb), 100)
```

During warmup:

- bias LR interpolates from `warmup_bias_lr` to target LR
- non-bias LR interpolates from `0.0` to target LR
- momentum interpolates from `warmup_momentum` to target momentum

If `optimizer=auto` selects AdamW, `warmup_bias_lr` is set to `0.0`.

### 9.5 AMP, Gradient Clipping, EMA

Training uses AMP by default when supported.

`BaseTrainer.optimizer_step()`:

```python
self.scaler.unscale_(self.optimizer)
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
self.scaler.step(self.optimizer)
self.scaler.update()
self.optimizer.zero_grad()
self.ema.update(self.model)
```

So default training includes:

- AMP
- gradient clipping max norm 10.0
- EMA model updates

### 9.6 Close Mosaic

Default:

```yaml
close_mosaic: 10
```

In the final 10 epochs, `BaseTrainer._close_dataloader_mosaic()` calls `YOLODataset.close_mosaic()`:

```python
hyp.mosaic = 0.0
hyp.copy_paste = 0.0
hyp.mixup = 0.0
hyp.cutmix = 0.0
self.transforms = self.build_transforms(hyp)
```

## 10. Validation Flow and Configuration

### 10.1 Validation During Training

In `BaseTrainer._do_train()`:

```python
if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
    self.metrics, self.fitness = self.validate()
```

Default validation config:

```yaml
val: True
split: val
save_json: False
conf: null
iou: 0.7
max_det: 300
half: False
dnn: False
plots: True
end2end: null
```

During training validation, `BaseValidator.__call__()` uses:

```python
self.args.half = self.device.type != "cpu" and trainer.amp
model = trainer.ema.ema or trainer.model
```

So training-time validation uses the EMA model and FP16 on GPU when AMP is enabled.

### 10.2 PoseValidator

`PoseTrainer.get_validator()` returns `PoseValidator` in `ultralytics/models/yolo/pose/val.py`.

`PoseValidator.__init__()`:

```python
self.args.task = "pose"
self.metrics = PoseMetrics()
```

### 10.3 Validation Dataloader

`BaseTrainer._build_train_pipeline()` creates validation loader with:

```python
batch_size=batch_size * 2
mode="val"
```

For pose, validation batch size is twice train batch size by default.

Validation dataset uses:

- `augment=False`
- `pad=0.5`
- `rect=True`
- transforms: `LetterBox(scaleup=False)` + `Format(return_keypoint=True)`

### 10.4 Validation Preprocess

`PoseValidator.preprocess()`:

```python
batch = super().preprocess(batch)
batch["keypoints"] = batch["keypoints"].float()
```

Parent `DetectionValidator.preprocess()`:

```python
batch[k] = batch[k].to(device)
batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
```

### 10.5 Validation NMS

`DetectionValidator.postprocess()` calls:

```python
nms.non_max_suppression(
    preds,
    self.args.conf,
    self.args.iou,
    nc=0 if self.args.task == "detect" else self.nc,
    multi_label=True,
    agnostic=self.args.single_cls or self.args.agnostic_nms,
    max_det=self.args.max_det,
    end2end=self.end2end,
    rotated=self.args.task == "obb",
)
```

Pose defaults:

- `conf`: if unspecified, `BaseValidator.__init__()` sets val conf to `0.001`
- `iou`: `0.7`
- `max_det`: `300`
- `multi_label`: `True`
- `agnostic_nms`: `False`
- `single_cls`: `False`
- `nc`: model class count, usually 1 for COCO pose

### 10.6 Keypoint Postprocess

`PoseValidator.postprocess()`:

```python
preds = super().postprocess(preds)
for pred in preds:
    pred["keypoints"] = pred.pop("extra").view(-1, *self.kpt_shape)
```

NMS `extra` columns become `[N, 17, 3]` keypoints for COCO pose.

### 10.7 Metrics

`PoseValidator.get_desc()` prints:

```text
Class
Images
Instances
Box(P
R
mAP50
mAP50-95)
Pose(P
R
mAP50
mAP50-95)
```

`PoseValidator.init_metrics()`:

```python
self.kpt_shape = self.data["kpt_shape"]
is_pose = self.kpt_shape == [17, 3]
self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
```

For COCO pose, validation uses COCO OKS sigma.

`PoseValidator._process_batch()`:

1. Parent detection validator computes bbox true positives over IoU thresholds `0.50:0.05:0.95`.
2. Pose validator computes keypoint OKS IoU:

```python
area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
```

The final metrics include both box and pose AP.

### 10.8 COCO JSON Evaluation

Default `save_json: False`. In standalone val, detection validator may auto-enable JSON saving for COCO/LVIS when not training.

`PoseValidator.pred_to_json()` adds keypoints to COCO-style prediction JSON.

`PoseValidator.eval_json()` evaluates:

```python
anno_json = data["path"] / "annotations/person_keypoints_val2017.json"
pred_json = save_dir / "predictions.json"
coco_evaluate(..., ["bbox", "keypoints"], suffix=["Box", "Pose"])
```

## 11. Training Loop Summary

```text
for epoch:
  scheduler.step()
  model.train()

  if epoch == epochs - close_mosaic:
      disable mosaic/mixup/cutmix/copy_paste

  for batch in train_loader:
      batch = preprocess_batch(batch)
          tensors -> device
          img -> float / 255
          optional multi_scale

      loss, loss_items = model(batch)
          BaseModel.forward(dict)
            -> model.loss(batch)
              -> PoseModel.init_criterion()
              -> v8PoseLoss(preds, batch)

      scaler.scale(loss).backward()
      if accumulated:
          unscale
          clip_grad_norm 10
          optimizer.step
          EMA update

  if val:
      validator(trainer)
          model = EMA
          img -> half/float / 255
          inference
          training val loss
          NMS conf=0.001, iou=0.7, max_det=300
          box metrics + pose OKS metrics

  save metrics/checkpoints
```

## 12. Default Config Summary

From `ultralytics/cfg/default.yaml`.

### Train

```yaml
epochs: 100
batch: 16
imgsz: 640
save: True
save_period: -1
cache: False
workers: 8
pretrained: True
optimizer: auto
seed: 0
deterministic: True
single_cls: False
rect: False
cos_lr: False
close_mosaic: 10
resume: False
amp: True
fraction: 1.0
freeze: null
multi_scale: 0.0
compile: False
```

### Optimizer / LR

```yaml
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
nbs: 64
```

With `optimizer=auto`, actual optimizer is determined by iterations:

- `iterations > 10000`: `MuSGD(lr=0.01, momentum=0.9)`
- otherwise: `AdamW(lr=0.002 * 5 / (4 + nc), beta1=0.9)`

### Pose Loss Gains

```yaml
box: 7.5
cls: 0.5
dfl: 1.5
pose: 12.0
kobj: 1.0
```

### Augmentation

```yaml
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
```

### Validation

```yaml
val: True
split: val
save_json: False
conf: null       # becomes 0.001 in BaseValidator for val, except OBB uses 0.01
iou: 0.7
max_det: 300
half: False      # training val overrides based on trainer AMP/GPU
dnn: False
plots: True
end2end: null
augment: False
agnostic_nms: False
classes: null
save_txt: False
save_conf: False
```

## 13. Key Takeaway

`yolov8n-pose` training is the standard YOLO detection training pipeline plus pose-specific dataset keypoints, `Pose` head, `v8PoseLoss`, and `PoseValidator`. Augmentation uses the YOLOv8 transform stack, with pose-specific `flip_idx` handling and `Format(return_keypoint=True)`. Loss adds keypoint coordinate OKS-style loss and keypoint visibility BCE on top of box/cls/dfl. Validation reports both bbox AP and pose OKS AP.
