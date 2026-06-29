# GPU Real YOLO-World Inference

Runs ParaScale inference with a real YOLO-World checkpoint on CUDA.

```bash
bash examples/gpu/example_004_yolo_world_real_inference/run.sh
```

The default path uses ParaScale device-side detection postprocess instead of
Ultralytics `predict()` postprocess.

Expected model mount:

```text
/models/yolo/yolov8s-worldv2.pt
```
