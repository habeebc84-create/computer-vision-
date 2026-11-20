# Everyday objects model

`objects365.onnx` is the FP32 YOLOE-11s ONNX export published by wide-video. The export embeds the 365 Objects365 category prompts. Its 640 × 640 RGB input produces a detection tensor of shape [1, 369, 8400]. The application takes the best class score per candidate and applies class-aware non-maximum suppression. Where two labels have boxes with intersection-over-union above 0.9, it retains the higher-scoring label. The ordered labels in `src/lib/objects365.ts` come from the ONNX metadata.

- [Original ONNX file and export recipe at revision 93d84d5](https://huggingface.co/wide-video/yoloe-11-Objects365-v1.0.0/tree/93d84d557920494a1070fe0dbd71a87be339d89a)
- [Ultralytics YOLOE documentation and model credits](https://docs.ultralytics.com/models/yoloe/)
- [Objects365 dataset](https://www.objects365.org/)
- [Model license, GNU AGPL version 3](AGPL-3.0.txt)

`npm run dev` and `npm run build` download the pinned model into this directory on first use and verify its checksum. Vite serves the file alongside the app, including on a static deployment. SHA-256 of the ONNX file is `4bd6925933413ac257ddb8dbd2c9ae7091150baffe6f7b41dcac580e889edb19`.
