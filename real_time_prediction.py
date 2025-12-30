import cv2
import pickle
import numpy as np
import os
import sys
import collections

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.landmark_extraction import LandmarkExtractor
from utils.audio_output import AudioOutput

# Config
MODEL_PATH = 'model/sign_model.pkl'
PREDICTION_WINDOW_SIZE = 10 # Frames to smooth prediction
CONFIDENCE_THRESHOLD = 0.8 # Increased threshold for strictly "hand of current database" prediction

def run_prediction():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run train_model.py first.")
        return

    print("Loading model...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    extractor = LandmarkExtractor()
    audio = AudioOutput()

    print("Starting Main Application...")
    print("Press 'q' to quit.")

    # Stability Tracking
    prediction_history = collections.deque(maxlen=PREDICTION_WINDOW_SIZE)
    last_spoken = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # ROI
        roi_x1, roi_y1 = w - 350, 50 
        roi_x2, roi_y2 = w - 50, 350
        
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        rgb_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        
        landmarks, hand_landmarks, _ = extractor.get_landmarks(rgb_roi)
        
        # Draw ROI Box
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        cv2.putText(frame, "HAND HERE", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        predicted_character = ""
        
        if landmarks:
            # Draw Skeleton
            extractor.draw_landmarks(roi_frame, hand_landmarks)
            
            # Predict
            features = np.array([landmarks])
            
            try:
                # Predict with probability
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    max_prob = np.max(probs)
                    prediction = model.classes_[np.argmax(probs)]
                    
                    # Display confidence
                    cv2.putText(frame, f"Conf: {max_prob:.2f}", (roi_x1, roi_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                    if max_prob < CONFIDENCE_THRESHOLD:
                        prediction = None
                else:
                    # Fallback for models without probability (e.g. some SVM configurations)
                    prediction = model.predict(features)[0]
            except Exception as e:
                print(f"Prediction error: {e}")
                prediction = None
            
            if prediction:
                prediction_history.append(prediction)
            else:
                # If we detected a hand but low confidence, maybe add a None to history to decay old stable prediction?
                # Or just do nothing.
                pass
            
            # Get most common prediction in window
            if len(prediction_history) > 0:
                most_common = collections.Counter(prediction_history).most_common(1)
                if most_common:
                    stable_prediction, count = most_common[0]
                    
                    # High stability requirement
                    if count >= PREDICTION_WINDOW_SIZE * 0.7: 
                        predicted_character = stable_prediction
                        
                        # Logic to trigger speech
                        if predicted_character != last_spoken:
                            audio.speak(predicted_character)
                            last_spoken = predicted_character
        
        # UI Overlay
        cv2.rectangle(frame, (0, 0),  (640, 60), (245, 117, 16), -1)
        display_text = f"Prediction: {predicted_character}" if predicted_character else "Waiting..."
        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Sign Language Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_prediction()
