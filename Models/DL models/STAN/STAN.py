import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

##########################
# Data Processing Section
##########################

# Load CSV and parse datetime column
df = pd.read_csv('/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv', parse_dates=['trans_date_trans_time'])

# Create a daily date column
df['date'] = df['trans_date_trans_time'].dt.date

# Define a function to compute spatial features for a given day.
def compute_spatial_features(df_day):
    spatial_features = {}
    # For each spatial level, we compute aggregated features: total amount, average amount, transaction count.
    for level in ['state', 'city', 'zip', 'merchant']:
        if df_day.empty:
            spatial_features[level] = np.array([0, 0, 0])
        else:
            grouped = df_day.groupby(level).agg({
                'amt': ['sum', 'mean', 'count']
            })
            grouped.columns = ['total_amt', 'avg_amt', 'trans_count']
            if grouped.empty:
                spatial_features[level] = np.array([0, 0, 0])
            else:
                # For simplicity, select the group with maximum transactions as the representative for that level.
                best = grouped.loc[grouped['trans_count'].idxmax()]
                spatial_features[level] = best.values  # shape (3,)
    # Order: state, city, zip, merchant → resulting shape: (4, 3)
    return np.array([
        spatial_features['state'],
        spatial_features['city'],
        spatial_features['zip'],
        spatial_features['merchant']
    ])

# Parameters for tensor dimensions:
num_temporal_slices = 7    # e.g., last 7 days per user
num_spatial_slices = 4     # state, city, zip, merchant
feature_dim = 3            # total_amt, avg_amt, trans_count

# Build a tensor per user.
users = df['cc_num'].unique()
user_tensors = []
labels_list = []

for user in users:
    user_df = df[df['cc_num'] == user]
    dates = sorted(user_df['date'].unique())
    daily_features = []
    for d in dates:
        df_day = user_df[user_df['date'] == d]
        spatial_feat = compute_spatial_features(df_day)  # shape: (4, 3)
        daily_features.append(spatial_feat)
    daily_features = np.array(daily_features)  # shape: (num_days, 4, 3)
    # Pad if user has fewer than num_temporal_slices days
    if daily_features.shape[0] < num_temporal_slices:
        pad_shape = (num_temporal_slices - daily_features.shape[0], num_spatial_slices, feature_dim)
        padding = np.zeros(pad_shape)
        daily_features = np.concatenate([padding, daily_features], axis=0)
    else:
        daily_features = daily_features[-num_temporal_slices:, :, :]
    user_tensors.append(daily_features)
    
    # For label, for example, if any transaction for the user is fraud, label the user as fraud.
    user_label = user_df['is_fraud'].max()
    labels_list.append(user_label)

# Convert lists into tensors.
input_tensor = torch.tensor(np.array(user_tensors), dtype=torch.float32)  # shape: (num_users, 7, 4, 3)
labels_tensor = torch.tensor(np.array(labels_list), dtype=torch.float32)

print("Input tensor shape:", input_tensor.shape)
print("Labels tensor shape:", labels_tensor.shape)

##########################
# Model Definition Section
##########################

# Temporal Attention Layer
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, lambda_t=0.1):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.lambda_t = lambda_t

    def forward(self, x):
        # x: (B, T, S, F)
        batch_size, T, S, F_dim = x.size()
        # Aggregate spatially (average over S)
        x_temp = x.mean(dim=2)  # (B, T, F_dim)
        scores = F.relu(self.fc(x_temp))  # (B, T, hidden_dim)
        scores = scores.mean(dim=2)       # (B, T)
        scores = (1 - self.lambda_t) * scores
        attn_weights = F.softmax(scores, dim=1)  # (B, T)
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        # Weighted sum over temporal slices
        x_attn = (x * attn_weights).sum(dim=1)  # (B, S, F_dim)
        # Add dummy temporal dimension for 3D conv (to create a 4D tensor later)
        x_attn = x_attn.unsqueeze(1)  # (B, 1, S, F_dim)
        return x_attn

# (Optional) Spatial Attention Layer - provided for completeness.
class SpatialAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, lambda_s=0.15):
        super(SpatialAttention, self).__init__()
        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.lambda_s = lambda_s

    def forward(self, x):
        # x: (B, S, F)
        scores = F.relu(self.fc(x))  # (B, S, hidden_dim)
        scores = scores.mean(dim=2)  # (B, S)
        scores = (1 - self.lambda_s) * scores
        attn_weights = F.softmax(scores, dim=1)  # (B, S)
        attn_weights = attn_weights.unsqueeze(-1)  # (B, S, 1)
        x_attn = (x * attn_weights).sum(dim=1)  # (B, F)
        return x_attn

# Full STAN Model integrating temporal attention and 3D convolution
class STAN(nn.Module):
    def __init__(self, temporal_slices, spatial_slices, feature_dim, 
                 temp_hidden_dim=16, spat_hidden_dim=16, conv_channels=8):
        super(STAN, self).__init__()
        self.temp_attn = TemporalAttention(feature_dim, temp_hidden_dim, lambda_t=0.1)
        # For this example, we use the 3D convolution output for detection.
        # 3D Convolution expects input shape: (B, channels, D, H, W)
        # We use a dummy depth D=1; H = spatial_slices, W = feature_dim.
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, conv_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(conv_channels, conv_channels * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((1, 1, 1))
        )
        # Detection layer: two fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_channels * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, T, S, F)
        x_temp = self.temp_attn(x)  # (B, 1, S, F)
        # Prepare for 3D conv: add dummy depth dimension → (B, 1, D=1, H=S, W=F)
        x_conv = x_temp.unsqueeze(2)
        conv_out = self.conv3d(x_conv)  # (B, conv_channels*2, 1, 1, 1)
        conv_out = conv_out.view(x.size(0), -1)  # Flatten to (B, conv_channels*2)
        out = self.fc(conv_out)  # (B, 1)
        return out

##########################
# Evaluation Function
##########################

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
    
    # Find best threshold based on F1 score
    best_threshold = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0, 1, 0.01):
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
    combined = (f1 + auc) / 2  # A simple combined metric
    return best_threshold, f1, auc, combined, acc, prec, rec

##########################
# Training Function
##########################

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    best_combined_metric_test = -float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
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
        print(f'\nEpoch {epoch+1}, Loss: {average_loss:.4f}')
        
        # Evaluate on training set
        train_threshold, train_f1, train_auc, train_combined, train_acc, train_prec, train_rec = evaluate_model(train_loader, model, device)
        print(f"Train Metrics - Best Threshold: {train_threshold:.2f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, Combined: {train_combined:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        
        # Evaluate on test set
        test_threshold, test_f1, test_auc, test_combined, test_acc, test_prec, test_rec = evaluate_model(test_loader, model, device)
        print(f"Test Metrics  - Best Threshold: {test_threshold:.2f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Combined: {test_combined:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
        
        # Save checkpoint based on test combined metric
        if test_combined > best_combined_metric_test:
            best_combined_metric_test = test_combined
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
                'train_f1': train_f1,
                'train_auc': train_auc,
                'train_combined': train_combined,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'test_combined': test_combined,
            }
            torch.save(checkpoint, 'best_checkpoint.pth')
            print(f'Checkpoint saved at epoch {epoch+1} with test combined metric: {test_combined:.4f}')
        
        # Early stopping: if loss does not improve for 8 consecutive epochs
        if average_loss < best_loss:
            best_loss = average_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 8:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
                
    print("\nTraining complete.")
    print(f"Best Test Combined Metric achieved: {best_combined_metric_test:.4f}")

##########################
# Main Training Loop
##########################

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and split into training and testing sets (80/20 split)
    dataset = TensorDataset(input_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, loss function and optimizer
    model = STAN(temporal_slices=num_temporal_slices, spatial_slices=num_spatial_slices, feature_dim=feature_dim)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 50
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)
    
    # (Optional) Load the best checkpoint and report final metrics
    checkpoint = torch.load('best_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("\nLoaded Best Checkpoint:")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Train F1: {checkpoint['train_f1']:.4f}, Train AUC: {checkpoint['train_auc']:.4f}, Train Combined: {checkpoint['train_combined']:.4f}")
    print(f"Test F1: {checkpoint['test_f1']:.4f}, Test AUC: {checkpoint['test_auc']:.4f}, Test Combined: {checkpoint['test_combined']:.4f}")
