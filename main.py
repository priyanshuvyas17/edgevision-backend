import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Export to ONNX if not exists, then load ONNX model (uses ONNX Runtime for inference)
MODEL_ONNX = "yolov8n.onnx"
if not os.path.exists(MODEL_ONNX):
    pt_model = YOLO("yolov8n.pt")
    pt_model.export(format="onnx", imgsz=640, simplify=True)
model = YOLO(MODEL_ONNX)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    results = model(file_path)
    detections = []

    for r in results:
        img_height, img_width = r.orig_shape
        if r.boxes is not None:
            for det_id, box in enumerate(r.boxes, start=1):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                bbox_norm = [
                    round(x1 / img_width, 6),
                    round(y1 / img_height, 6),
                    round(x2 / img_width, 6),
                    round(y2 / img_height, 6),
                ]
                detections.append({
                    "id": det_id,
                    "bbox": bbox_norm,
                    "label": label,
                    "confidence": conf,
                    "mask": {"type": "box", "bbox": bbox_norm},
                })

    return {"bbox_format": "xyxy", "detections": detections}


@app.post("/track")
async def track_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save video
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Run YOLO tracking (ONNX inference)
    results = model.track(
        source=file_path,
        stream=True,
        persist=True,
        imgsz=320,
        conf=0.25,
        verbose=False,
    )

    frames_output = []
    frame_id = 0

    for r in results:
        frame_id += 1

        # 🔴 LIMIT FRAMES (VERY IMPORTANT)
        if frame_id > 15:
            break

        # ⚡ SKIP FRAMES (FASTER)
        if frame_id % 2 != 0:
            continue

        detections = []

        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                track_id = int(box.id[0]) if box.id is not None else -1

                # Normalize coordinates
                h, w = r.orig_shape
                x1 /= w
                x2 /= w
                y1 /= h
                y2 /= h

                detections.append({
                    "track_id": track_id,
                    "bbox": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                    "label": label,
                    "confidence": round(conf, 3)
                })

        frames_output.append({
            "frame_id": frame_id,
            "detections": detections
        })

    return {
        "frames": frames_output
    }