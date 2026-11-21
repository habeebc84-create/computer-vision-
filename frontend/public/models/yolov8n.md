# YOLOv8n COCO ONNX model

The bundled model is the YOLOv8n ONNX export from
[yoobright/yolo-onnx](https://github.com/yoobright/yolo-onnx/blob/bbaf44a18a69869452db0b7e53af017557f46f49/yolov8n.onnx),
used for the browser's 80-class detector. YOLOv8 is distributed by Ultralytics
under the AGPL-3.0 license (see AGPL-3.0.txt). Consult Ultralytics for other
licensing options.

The model is included so that detection does not depend on another host being
reachable at runtime. It expects a 640×640 image and produces [1, 84, 8400].
