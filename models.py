# src/models.py

import torch
from torchvision.transforms import ToTensor
import cv2
import numpy as np
from ultralytics import YOLO
from restormer import Restormer

# Pilih device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) Inisialisasi Restormer
rest_model = Restormer().to(DEVICE)
rest_model.load_state_dict(torch.load(
    'weights/restormer.pth', map_location=DEVICE))
rest_model.eval()

def enhance(img_bgr: np.ndarray) -> np.ndarray:
    """Enhance image using Restormer."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = ToTensor()(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = rest_model(tensor)
    out_np = out.squeeze(0).permute(1,2,0).cpu().numpy()
    img_enh = (np.clip(out_np, 0, 1)*255).astype(np.uint8)
    return cv2.cvtColor(img_enh, cv2.COLOR_RGB2BGR)

# 2) Inisialisasi YOLOv8
yolo_model = YOLO('weights/best.pt')

def detect(img_bgr: np.ndarray, conf=0.25, iou=0.45):
    """Detect plates using YOLOv8 and return annotated image + box list."""
    results   = yolo_model(img_bgr, conf=conf, iou=iou)[0]
    annotated = results.plot()
    boxes     = []
    for box, score in zip(results.boxes.xyxy.cpu().numpy(),
                          results.boxes.conf.cpu().numpy()):
        x1,y1,x2,y2 = box.round().astype(int)
        boxes.append((x1, y1, x2, y2, float(score)))
    return annotated, boxes
