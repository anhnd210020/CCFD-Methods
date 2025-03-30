import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

########################################
# Focal Loss Implementation for Imbalanced Data
########################################

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        """
        Focal loss for binary classification.
        
        Args:
            gamma (float): Focusing parameter.
            alpha (float): Weighting factor for the rare class.
            reduction (str): Reduction method ('mean' or 'sum').
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Ensure inputs and targets are flattened.
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # Compute binary cross entropy loss.
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # Compute the probability of the true class.
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        # Compute the focal loss.
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

########################################
# Contrastive Loss Implementation
########################################

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Contrastive loss that encourages embeddings of the same class to be close
        and embeddings of different classes to be separated by at least a margin.
        
        Args:
            margin (float): Margin for dissimilar pairs.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape (B, D) where D is the embedding dimension.
            labels: Tensor of shape (B,) with binary labels.
        Returns:
            Scalar contrastive loss.
        """
        batch_size = embeddings.size(0)
        # Compute pairwise Euclidean distances.
        # Expand embeddings to compute differences between every pair.
        expanded_a = embeddings.unsqueeze(1).expand(batch_size, batch_size, -1)
        expanded_b = embeddings.unsqueeze(0).expand(batch_size, batch_size, -1)
        distances = torch.sqrt(torch.sum((expanded_a - expanded_b) ** 2, dim=2) + 1e-8)
        
        # Create a label matrix where entry (i, j)=1 if same class, 0 otherwise.
        labels = labels.view(-1)
        label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        
        # Contrastive loss formula:
        # For similar pairs (label_matrix==1): loss = 0.5 * distance^2
        # For dissimilar pairs (label_matrix==0): loss = 0.5 * max(0, margin - distance)^2
        loss_similar = 0.5 * (distances ** 2)
        loss_dissimilar = 0.5 * (F.relu(self.margin - distances) ** 2)
        
        # Combine losses.
        loss = label_matrix * loss_similar + (1 - label_matrix) * loss_dissimilar
        
        # We consider only each pair once (i<j) to avoid redundancy.
        mask = torch.triu(torch.ones_like(loss), diagonal=1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

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
        """
        x: Tensor of shape (B, T, S, F)
        is_night: Tensor of shape (B, T) with binary indicators (0 or 1).
                  If None, assume all values are 0.
        """
        batch_size, T, S, F_dim = x.size()
        if is_night is None:
            is_night = torch.zeros(batch_size, T, device=x.device)
        
        # Step 1: Aggregate over the S dimension (e.g., averaging)
        x_temp = x.mean(dim=2)  # (B, T, F_dim)
        
        # Step 2: Project to hidden dimension and apply ReLU activation
        scores = F.relu(self.fc(x_temp))  # (B, T, hidden_dim)
        
        # Step 3: Collapse the hidden dimension to get a single score per time step
        scores = scores.mean(dim=2)       # (B, T)
        
        # Step 4: Scale the scores
        scores = (1 - self.lambda_t) * scores
        
        # Step 5: Add extra bonus for nighttime transactions.
        beta = 1.0  # Adjust or make learnable as needed.
        scores = scores + beta * is_night
        
        # Step 6: Softmax normalization over time steps
        attn_weights = F.softmax(scores, dim=1)  # (B, T)
        
        # Step 7: Reshape the attention weights for broadcasting
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        
        # Step 8: Weight the original input and sum over the temporal dimension
        x_attn = (x * attn_weights).sum(dim=1)  # (B, S, F_dim)
        
        # Optional: add an extra dimension if needed by later layers
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
        # Define a two-part fully connected network.
        # fc_part extracts embeddings, and fc_out produces the final prediction.
        self.fc_part = nn.Sequential(
            nn.Linear(conv_channels * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, T, S, F)
        # Using default is_night = zeros (or modify to pass an is_night tensor if available)
        x_temp = self.temp_attn(x)  # (B, 1, S, F)
        x_conv = x_temp.unsqueeze(2)  # (B, 1, D=1, H=S, W=F)
        conv_out = self.conv3d(x_conv)  # (B, conv_channels*2, 1, 1, 1)
        conv_out = conv_out.view(x.size(0), -1)  # (B, conv_channels*2)
        # Extract embeddings
        embeddings = self.fc_part(conv_out)  # (B, 32)
        # Final classification output
        out = self.fc_out(embeddings)  # (B, 1)
        return out, embeddings

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
            outputs, _ = model(X_batch)
            outputs = outputs.squeeze()
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    
    # Test fixed thresholds and choose the best based on F1
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
    combined = (f1 + auc) / 2  # Simple combined metric
    return best_threshold, f1, auc, combined, acc, prec, rec

########################################
# Training Function with Early Stopping and Combined Loss
########################################

def train_model(model, train_loader, test_loader, focal_criterion, contrastive_criterion, optimizer, num_epochs, device, lambda_contrastive=0.1):
    best_loss = float('inf')
    best_combined_metric_test = -float('inf')
    best_epoch = -1
    best_test_f1 = 0.0
    best_test_auc = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs, embeddings = model(X_batch)
            outputs = outputs.squeeze()
            # Classification loss.
            loss_class = focal_criterion(outputs, y_batch)
            # Contrastive loss.
            loss_contrastive = contrastive_criterion(embeddings, y_batch)
            # Combined loss.
            loss = loss_class + lambda_contrastive * loss_contrastive
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1}, Loss: {average_loss:.4f}')
        
        # Evaluate on training set.
        train_threshold, train_f1, train_auc, train_combined, train_acc, train_prec, train_rec = evaluate_model(train_loader, model, device)
        print(f"Train Metrics - Best Threshold: {train_threshold:.2f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, Combined: {train_combined:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        
        # Evaluate on test set.
        test_threshold, test_f1, test_auc, test_combined, test_acc, test_prec, test_rec = evaluate_model(test_loader, model, device)
        print(f"Test Metrics  - Best Threshold: {test_threshold:.2f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Combined: {test_combined:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
        
        # Update best results based on test combined metric.
        if test_combined > best_combined_metric_test:
            best_combined_metric_test = test_combined
            best_epoch = epoch + 1
            best_test_f1 = test_f1
            best_test_auc = test_auc
        
        # Early stopping: if loss does not improve for 8 consecutive epochs.
        if average_loss < best_loss:
            best_loss = average_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 8:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
                
    print("\nTraining complete.")
    print(f"Best Test Combined Metric: {best_combined_metric_test:.4f} at epoch {best_epoch}")
    print(f"Best Test F1: {best_test_f1:.4f}, Best Test AUC: {best_test_auc:.4f}")
    return best_combined_metric_test, best_epoch, best_test_f1, best_test_auc

########################################
# Main Training Loop
########################################

if __name__ == "__main__":
    # Device configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and split into training and testing sets (80/20 split).
    dataset = TensorDataset(input_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, loss functions and optimizer.
    model = STAN(temporal_slices=num_temporal_slices, spatial_slices=num_spatial_slices, feature_dim=feature_dim)
    model.to(device)
    
    focal_criterion = FocalLoss(gamma=2, alpha=0.25, reduction='mean')
    contrastive_criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    best_combined_metric, best_epoch, best_test_f1, best_test_auc = train_model(
     model, train_loader, test_loader, focal_criterion, contrastive_criterion, optimizer, num_epochs, device, lambda_contrastive=0.1)
    


