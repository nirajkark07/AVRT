import sys
import os
import cv2
import math
from ultralytics import YOLOv10
import numpy as np

class YOLOv10Detector:
    def __init__(self, model_path="best.pt", class_names=None, img_width = 640, img_height = 480):
        self.model = YOLOv10(model_path)
        self.class_names = class_names if class_names else ["Gearbox"]
        self.width = img_width
        self.height = img_height

    def get_detection_mask(self, frame, conf_threshold=0.25):
        """
        Takes an RGB frame and returns the mask of the detected object.
        :param frame: The RGB image frame.
        :param conf_threshold: The confidence threshold for detection.
        :return: A mask where the object is detected.
        """
        results = self.model.predict(frame, conf=conf_threshold)
        mask = None
        detected_class = False

        # Create a blank mask with the same size as the frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Draw rectangle on mask for each detected object
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # 255 means the object region
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = self.class_names[cls]
                # print("THIS IS THE CLASS_NAME")
                # print(class_name)
                # if class_name == 'GearBox':
                print(f"Detected: {class_name} with confidence {conf}")
                mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool).astype(np.uint8)
                detected_class = True
                
        
        return mask, detected_class 

        
