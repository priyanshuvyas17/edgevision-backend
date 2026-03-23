"""
Export YOLOv8 to ONNX format for faster inference.
Run once before starting the server, or the server will export on first start.
"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=640, simplify=True)
    print("Exported yolov8n.onnx successfully.")
