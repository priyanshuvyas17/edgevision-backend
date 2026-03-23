# EdgeVision Backend

FastAPI backend with YOLOv8 object detection and tracking (ONNX Runtime).

## Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Pre-export the model to ONNX for faster startup:

   ```bash
   python export_onnx.py
   ```

   If skipped, the server will export on first run.

## Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- **API docs**: http://localhost:8000/docs
- **POST /upload** – Image detection (returns normalized bboxes)
- **POST /track** – Video tracking (returns per-frame detections with track IDs)
