# CCSC_submission/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from tqdm import tqdm
from model_architecture import MTST # Import from our file

# --- Configuration (from your notebook) ---
PROCESSED_DATA_DIR = 'data' # Assumes data is in a subfolder
MODELS_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'mtst_flare_model.pth')
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Model Hyperparameters ---
NUM_FEATURES = 16 # As determined by feature selection
TIME_STEPS = 24
PATCH_LENGTH = 2
D_MODEL = 128
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# --- Training Parameters ---
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 2e-5

# --- Load Data ---
print("Loading preprocessed training data...")
X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Data loaded.")

# --- Initialize Model and Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MTST(
    num_features=NUM_FEATURES, time_steps=TIME_STEPS, patch_length=PATCH_LENGTH,
    d_model=D_MODEL, num_heads=NUM_HEADS, num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- Training Loop ---
print(f"\n--- Starting Model Training on {device} ---")
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    for features, labels in progress_bar:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")

# --- Save the Trained Model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n--- Training complete! Model saved to {MODEL_SAVE_PATH} ---")