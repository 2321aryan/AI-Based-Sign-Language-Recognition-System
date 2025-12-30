# AI Sign Language Recognition System

## Abstract
A real-time AI-based system that converts static hand gestures into text and speech using computer vision and machine learning.

## Problem Statement
Bridging the communication gap between sign language users and non-signers using accessible technology.

## System Architecture
Webcam Input -> MediaPipe Landmark Extraction -> Feature Normalization -> Random Forest Classifier -> Text/Speech Output

## Technology Stack
- **Language**: Python
- **Vision**: OpenCV, MediaPipe
- **ML**: Scikit-learn (Random Forest)
- **Audio**: pyttsx3

## Execution Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Collect data: `python src/collect_data.py`
3. Train model: `python src/train_model.py`
4. Run system: `python src/real_time_prediction.py`
