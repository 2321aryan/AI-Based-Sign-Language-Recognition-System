import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Config
DATASET_PATH = 'dataset/signs_dataset.csv'
MODEL_PATH = 'model/sign_model.pkl'

def train_model():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}. Please run collect_data.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    if df.empty:
        print("Dataset is empty.")
        return

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    if len(set(y)) < 2:
        print("Warning: Need at least 2 classes.")
        return

    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {set(y)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

    print("Training Gradient Boosting Classifier (this may take a moment)...")
    # Pipeline: Scale data -> Train GBM
    model = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

    # Ensure directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
