import cv2
import numpy as np

class SemanticFilter:
    def __init__(self, model_name="yolov8n.pt", conf_thresh=0.3):
        self.enabled = False
        self.conf_thresh = conf_thresh
        try:
            from ultralytics import YOLO
            # Load YOLOv8 model
            self.model = YOLO(model_name)
            self.enabled = True
            print(f"==> YOLOv8 Loaded for Semantic SLAM ({model_name})")
        except ImportError:
            print("==> [Warning] ultralytics not installed. Semantic SLAM disabled.")
            print("==> To enable: pip install ultralytics")
        
        # COCO class IDs for dynamic objects we want to mask
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.dynamic_classes = [0, 1, 2, 3, 5, 7]

    def get_dynamic_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask of the same shape as frame (H, W).
        True means the pixel belongs to a dynamic object.
        False means it's static environment.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        if not self.enabled:
            return mask

        # Run inference
        results = self.model(frame, verbose=False, conf=self.conf_thresh)
        
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in self.dynamic_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Ensure within bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    mask[y1:y2, x1:x2] = True
                    
        return mask
