# CCSC_submission/test.py
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from model_architecture import MTST
import os 
# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed"
MODEL_PATH = "models/mtst_flare_model.pth"
NUM_FEATURES = 16
TIME_STEPS = 24
PATCH_LENGTH = 2
D_MODEL = 128
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# --- Load Data ---
print("Loading testing data...")
X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MTST(
    num_features=NUM_FEATURES, time_steps=TIME_STEPS, patch_length=PATCH_LENGTH,
    d_model=D_MODEL, num_heads=NUM_HEADS, num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded and ready for evaluation on {device}.")

# --- Generate Predictions ---
print("Generating predictions on the test set...")
with torch.no_grad():
    # Process the entire test set in one batch for speed
    logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)

# --- Calculate and Display Performance Metrics ---
print("\n--- Final Model Performance Metrics ---")
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
tss = recall + specificity - 1

print(f"Confusion Matrix (TN, FP, FN, TP): ({tn}, {fp}, {fn}, {tp})")
print(f"True Skill Statistic (TSS): {tss:.4f}")