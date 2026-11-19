# Package detector

The package detector locates parcels in logistics and warehouse images. It was fine-tuned from Ultralytics YOLOv8n and has one class named `package`. The original COCO checkpoint has no package category. The trained PyTorch checkpoint is `models/package-yolov8n.pt`. The browser export is `frontend/public/models/package.onnx`, with a 320 × 320 RGB input, a single output of shape [1, 5, 2100], and ONNX opset 17.

## Data and evaluation

The training run used the factorypackage group's **factory_package** dataset (2024), distributed by Ultralytics as Package-Seg. Each polygon was converted into its enclosing bounding box. The run sampled one variant per source filename within each official split, then selected 256 training images and 72 validation images with random seed 17. The 89 test images contain 325 packages. No test source filename appears in either development split. One source filename occurs in both training and validation. The test results describe logistics imagery. Household photo performance is unmeasured. [Original dataset](https://universe.roboflow.com/factorypackage/factory_package) · [Ultralytics dataset guide](https://docs.ultralytics.com/datasets/segment/package-seg/)

| Held-out metric at 320 × 320 | Result |
| --- | ---: |
| AP at IoU 0.50 | 0.746 |
| AP averaged over IoU 0.50–0.95 | 0.547 |
| Precision at confidence 0.25 | 0.839 |
| Recall at confidence 0.25 | 0.609 |

At confidence 0.25 and NMS IoU 0.45, the PyTorch model matched 198 of 325 objects, missed 127, and returned 38 unmatched detections. Matches used one prediction per ground-truth box at IoU ≥ 0.50.

## Reproduce the weights

Use a clean working directory with the Package-Seg archive extracted into `images/` and `labels/` and the original `yolov8n.pt` checkpoint alongside them. Run `python /path/to/repo/scripts/train_package_detector.py` there. The run used Python 3.12, Ultralytics 8.3.237, CPU PyTorch 2.5.1, torchvision 0.20.1, and ONNX 1.17.0. It trained five epochs, froze the first ten backbone modules, selected the checkpoint using validation metrics, and exported the weights into `fit/package/weights/`. [Dataset archive](https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip) · [Starting checkpoint](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt)

| Asset | SHA-256 |
| --- | --- |
| Package-Seg archive | `62ff29db4b04c2922ebee6879fc5e05de4bd8c8b35b61088337e0b64a68324f6` |
| Upstream YOLOv8n | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| Fine-tuned PyTorch model | `b71faa9f4fd8972d4c0dde783fea41d34a72c4c661bf5c0236ff8bd10b2114f4` |
| Exported ONNX model | `b52d06ee1628af930bf6485f7a3b8b062b63fd9b387d310d61d631aad26af743` |

The fine-tuned weights derive from Ultralytics YOLOv8n and are distributed under AGPL-3.0. [License](PACKAGE-LICENSE.txt) · [Ultralytics licensing](https://ultralytics.com/license)
