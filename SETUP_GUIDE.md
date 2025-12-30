# Setup Guide for AI Sign Language Recognition System

This guide provides step-by-step instructions to set up the **AI Sign Language Recognition System** on a new laptop (Windows).

## 1. Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.10 or higher** (Python 3.12 is recommended).
  - [Download Python](https://www.python.org/downloads/)
  - **Important:** During installation, check the box that says **"Add Python to PATH"**.
- **Git** (Optional, for cloning the repository).
  - [Download Git](https://git-scm.com/downloads)
- **VS Code** (Optional, recommended code editor).

## 2. Project Setup

### Step 2.1: Copy the Project
Copy the entire `SignLanguageRecognition` folder to the new laptop. You can use a USB drive, Google Drive, or `git clone` if it's hosted on GitHub.

### Step 2.2: Open in Terminal
1. Open the project folder.
2. Right-click in the empty space and select **"Open in Terminal"** (or open Command Prompt and `cd` to the folder path).

### Step 2.3: Create a Virtual Environment (Recommended)
It is good practice to use a virtual environment to isolate dependencies.
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate
```
*You should see `(venv)` appear at the beginning of your terminal line.*

### Step 2.4: Install Dependencies
Install all required libraries using the `requirements.txt` file.
```powershell
pip install -r requirements.txt
```

**Troubleshooting Installation:**
- If you get an error related to `pip`, upgrade it first:
  ```powershell
  python -m pip install --upgrade pip
  ```
- If `pyttsx3` or `opencv` fails, you might need to install C++ build tools or generic system libraries, but usually, the wheels provided by pip are sufficient.

## 3. Project Structure
Ensure your folder structure looks like this:
```
SignLanguageRecognition/
├── dataset/               # Stores collected hand sign data
├── model/                 # (Optional) Intermediate model files
├── models/                # Stores the trained model (model.p)
├── src/                   # Source code
│   ├── collect_data.py        # Script to collect data
│   ├── train_model.py         # Script to train the model
│   └── real_time_prediction.py # Main script for prediction
├── utils/                 # Utility scripts (landmark extraction)
├── requirements.txt       # List of dependencies
└── README.md              # Project overview
```

## 4. How to Run

### Step 4.1: Collect Data (If starting fresh)
If you want to train your own signs, run the data collection script.
```powershell
python src/collect_data.py
```
- It will prompt you for the **number of classes** (signs) and **dataset size**.
- Follow the on-screen instructions to record gestures.

### Step 4.2: Train the Model
After collecting data, train the machine learning model.
```powershell
python src/train_model.py
```
- This will generate a `model.p` file in the `models/` directory.
- *Note: If you copied the `models/` folder from the old laptop, you can skip this step.*

### Step 4.3: Start Prediction
Run the main recognition system.
```powershell
python src/real_time_prediction.py
```
- A webcam window will open.
- Perform the hand gestures you trained.
- The system will allow you to construct sentences and speak them.

## 5. Common Issues

- **Webcam not opening:** Check if another app is using the camera.
- **"Module not found":** Ensure your virtual environment is activated (`.\venv\Scripts\activate`) and you ran `pip install -r requirements.txt`.
- **Low accuracy:** Ensure good lighting and a plain background when using the system.
