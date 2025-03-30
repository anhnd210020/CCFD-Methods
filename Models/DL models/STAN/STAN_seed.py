import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import csv
import random
from tqdm import tqdm
import os

########################################
# Focal Loss Implementation for Imbalanced Data
########################################

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

########################################
# Load Preprocessed Data
########################################

num_temporal_slices = 30    # e.g., last 30 days per user (adjust as needed)
num_spatial_slices = 4      # e.g., state, city, zip, merchant
feature_dim = 3             # total_amt, avg_amt, trans_count

data = torch.load('/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/preprocessed_data.pt')
input_tensor = data['input_tensor']
labels_tensor = data['labels_tensor']

print("Loaded input tensor shape:", input_tensor.shape)
print("Loaded labels tensor shape:", labels_tensor.shape)

########################################
# Model Definition Section
########################################

class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, lambda_t=0.1):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.lambda_t = lambda_t

    def forward(self, x, is_night=None):
        batch_size, T, S, F_dim = x.size()
        if is_night is None:
            is_night = torch.zeros(batch_size, T, device=x.device)
        x_temp = x.mean(dim=2)  # (B, T, F_dim)
        scores = F.relu(self.fc(x_temp))  # (B, T, hidden_dim)
        scores = scores.mean(dim=2)       # (B, T)
        scores = (1 - self.lambda_t) * scores
        beta = 1.0  # Bonus for nighttime transactions.
        scores = scores + beta * is_night
        attn_weights = F.softmax(scores, dim=1)  # (B, T)
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        x_attn = (x * attn_weights).sum(dim=1)  # (B, S, F_dim)
        x_attn = x_attn.unsqueeze(1)  # (B, 1, S, F_dim)
        return x_attn

class STAN(nn.Module):
    def __init__(self, temporal_slices, spatial_slices, feature_dim, 
                 temp_hidden_dim=16, conv_channels=8):
        super(STAN, self).__init__()
        self.temp_attn = TemporalAttention(feature_dim, temp_hidden_dim, lambda_t=0.1)
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, conv_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(conv_channels, conv_channels * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(conv_channels * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_temp = self.temp_attn(x)  # (B, 1, S, F)
        x_conv = x_temp.unsqueeze(2)  # (B, 1, 1, S, F)
        conv_out = self.conv3d(x_conv)  # (B, conv_channels*2, 1, 1, 1)
        conv_out = conv_out.view(x.size(0), -1)  # (B, conv_channels*2)
        out = self.fc(conv_out)  # (B, 1)
        return out

########################################
# Evaluation Function
########################################

def evaluate_model(data_loader, model, device):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch).squeeze()
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_threshold = 0.5
    best_f1 = 0.0
    for thresh in thresholds:
        preds = (all_outputs >= thresh).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    preds = (all_outputs >= best_threshold).astype(int)
    f1 = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_outputs)
    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds)
    rec = recall_score(all_labels, preds)
    combined = (f1 + auc) / 2  # Combined metric
    return best_threshold, f1, auc, combined, acc, prec, rec

########################################
# Training Function with Early Stopping
########################################

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    best_combined_metric_test = -float('inf')
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        
        # Evaluate on train and test sets
        _, _, _, _, _, _, _ = evaluate_model(train_loader, model, device)
        test_threshold, test_f1, test_auc, test_combined, _, _, _ = evaluate_model(test_loader, model, device)
        
        if test_combined > best_combined_metric_test:
            best_combined_metric_test = test_combined
            best_epoch = epoch + 1
            best_test_f1 = test_f1
            best_test_auc = test_auc
        
        if average_loss < best_loss:
            best_loss = average_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 8:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nBest Test Combined Metric: {best_combined_metric_test:.4f} at epoch {best_epoch}")
    print(f"Best Test F1: {best_test_f1:.4f}, Best Test AUC: {best_test_auc:.4f}")
    return best_combined_metric_test, best_epoch, best_test_f1, best_test_auc

########################################
# Main Experiment Loop (Seed 37)
########################################

csv_filename = 'results_seeds_1_to_100.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    fieldnames = ['seed', 'best_combined_metric', 'best_epoch', 'best_test_f1', 'best_test_auc']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use seed 37 for this example
    for seed in [37]:
        print(f"\nRunning experiment for seed: {seed}")
        
        # Set seeds for reproducibility and enforce determinism
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Create a new dataset split
        dataset = TensorDataset(input_tensor, labels_tensor)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model, loss, and optimizer
        model = STAN(temporal_slices=num_temporal_slices, spatial_slices=num_spatial_slices, feature_dim=feature_dim)
        model.to(device)
        criterion = FocalLoss(gamma=2, alpha=0.25, reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 100
        
        print(f"Seed {seed} - Running training...")
        res = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)
        
        # Save results for this seed
        writer.writerow({
            'seed': seed,
            'best_combined_metric': res[0],
            'best_epoch': res[1],
            'best_test_f1': res[2],
            'best_test_auc': res[3]
        })

print(f"\nExperiment complete. Results saved to {csv_filename}.")
