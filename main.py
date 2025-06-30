import os
from ultralytics import YOLO

# Always load from absolute path to avoid "file not found"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best.pt")
video_path = os.path.join(BASE_DIR, "input", "match_clip.mp4")

# Load model
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# Run prediction
print(f"Running prediction on: {video_path}")
results = model.predict(source=video_path, save=True, conf=0.25)

print("✅ Prediction complete! Output saved to 'runs/detect/' folder.")
