import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class LandmarkExtractor:
    def __init__(self, model_path='models/hand_landmarker.task'):
        self.model_path = model_path
        
        # Create model folder if not exists (though we expect model to be there)
        if not os.path.exists(self.model_path):
            print(f"Warning: Model not found at {self.model_path}. Please ensure it is downloaded.")
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
            
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.background = None # logic removed, kept for compatibility if accessed

    def capture_background(self, image_rgb):
        # specific request: "remove background capture"
        # We make this a no-op or just print instructions are not needed
        print("Background capture not needed with this model.")
        pass

    def get_landmarks(self, image_rgb):
        """
        Uses MediaPipe HandLandmarker.
        Returns: flattened_landmarks (list), hand_landmarks (list of objects), None (mask)
        """
        # Create MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect
        detection_result = self.landmarker.detect(mp_image)
        
        if detection_result.hand_landmarks:
            # Get first hand
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Use raw unflattened landmarks for internal use if needed, 
            # but return flattened list for CSV compatibility: [x0, y0, z0, x1, y1, z1...]
            flattened = []
            
            for lm in hand_landmarks:
                flattened.extend([lm.x, lm.y, lm.z])
            
            # We return:
            # 1. Flattened list (for model)
            # 2. hand_landmarks objects (for drawing) - hacked into "contour" slot
            # 3. None (mask is irrelevant)
            return flattened, hand_landmarks, None
            
        return [], None, None

    def draw_landmarks(self, image, landmarks):
        """
        Draws the landmarks and connections on the image.
        Args:
            image: BGR image to draw on.
            landmarks: list of NormalizedLandmark objects (returned as second arg from get_landmarks)
        """
        if not landmarks:
            return

        h, w, _ = image.shape
        
        # Connections for a hand
        CONNECTIONS = frozenset([
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ])

        # Convert to pixel coords
        points = {}
        for idx, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[idx] = (cx, cy)
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
            
        # Draw lines
        for start_idx, end_idx in CONNECTIONS:
            if start_idx in points and end_idx in points:
                cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)
