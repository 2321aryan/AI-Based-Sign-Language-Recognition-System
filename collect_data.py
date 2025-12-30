import cv2
import pandas as pd
import os
import sys
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.landmark_extraction import LandmarkExtractor

# Config
DATASET_PATH = 'dataset/signs_dataset.csv'
COLUMNS = []
for i in range(21):
    COLUMNS.extend([f'x{i}', f'y{i}', f'z{i}'])
COLUMNS.append('label')

def collect_data():
    cap = cv2.VideoCapture(0)
    extractor = LandmarkExtractor()
    
    data = []
    
    # Check if dataset exists
    if os.path.exists(DATASET_PATH):
        print(f"Dataset found at {DATASET_PATH}. New data will be appended.")
        print("NOTE: If you changed the feature extractor (e.g. CV to MediaPipe), delete the old dataset first!")
    else:
        print("Creating new dataset.")

    current_label = input("Enter the gesture label to collect (e.g., 'A', 'Hello'): ").strip()
    if not current_label:
        print("Label cannot be empty.")
        return

    print(f"Collecting data for label: '{current_label}'")
    print("Press 's' to save a frame (hold for continuous), 'q' to quit.")

    params = {'samples_collected': 0}
    last_save_time = 0
    save_cooldown = 0.1 # Faster collection
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Define ROI (Same 300x300 box)
        roi_x1, roi_y1 = w - 350, 50 
        roi_x2, roi_y2 = w - 50, 350
        
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        # MediaPipe needs RGB
        rgb_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        
        # landmarks is flattened list [x,y,z...], hand_landmarks is object list (for drawing)
        landmarks, hand_landmarks, _ = extractor.get_landmarks(rgb_roi)
        
        # Draw ROI Box
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        cv2.putText(frame, "PLACE HAND HERE", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if landmarks:
             # Draw Skeleton on the ROI
             extractor.draw_landmarks(roi_frame, hand_landmarks)
             # roi_frame is a view, so it updates 'frame' automatically
        
        # UI Overlay
        cv2.putText(frame, f"Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {params['samples_collected']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if not landmarks:
             cv2.putText(frame, "NO HAND DETECTED", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Feedback for recent save
        if time.time() - last_save_time < 0.5:
             cv2.putText(frame, "SAVED!", (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, "Press 'S' to Save, 'Q' to Quit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Data Collection", frame)
        
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("Quit pressed. Exiting...")
            break
        elif key == ord('s') or key == ord('S'):
            current_time = time.time()
            if current_time - last_save_time > save_cooldown:
                if landmarks:
                    row = landmarks + [current_label]
                    data.append(row)
                    params['samples_collected'] += 1
                    print(f"Saved sample {params['samples_collected']}")
                    last_save_time = current_time
                    
                    # Flash effect
                    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 3)
                    cv2.imshow("Data Collection", frame)
                    cv2.waitKey(5)
                else:
                    print("No hand detected! Sample not saved.")

    cap.release()
    cv2.destroyAllWindows()
    
    if data:
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        df = pd.DataFrame(data, columns=COLUMNS)
        if os.path.exists(DATASET_PATH):
            df.to_csv(DATASET_PATH, mode='a', header=False, index=False)
        else:
            df.to_csv(DATASET_PATH, mode='w', header=True, index=False)
        print(f"Successfully saved {len(data)} samples to {DATASET_PATH}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    collect_data()
