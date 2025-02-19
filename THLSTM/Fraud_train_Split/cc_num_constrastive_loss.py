#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# ------------------------------
# Contrastive Loss Function with Multiple Metrics
# ------------------------------
def contrastive_loss(embeddings, labels, margin=1.0, similarity_metric='euclid', cosine_weight=0.5):
    """
    Computes the contrastive loss using either Euclidean, cosine, or a combination of both distances.
    
    For a pair of examples (i, j):
      L = 0.5 * (similarity) * D^2 + 0.5 * (1 - similarity) * max(0, margin - D)^2
      
    where similarity = 1 if labels are the same, and 0 otherwise.
    
    Args:
        embeddings (Tensor): shape (batch_size, embedding_dim)
        labels (Tensor): shape (batch_size, 1) or (batch_size,) with binary labels (0 or 1)
        margin (float): margin for dissimilar pairs.
        similarity_metric (str): one of 'euclid', 'cosine', or 'both'. Determines which distance metric to use.
        cosine_weight (float): weight applied to the cosine distance when using the 'both' option.
        
    Returns:
        Tensor: the contrastive loss value.
    """
    batch_size = embeddings.size(0)
    
    # Compute pairwise Euclidean distances
    diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # shape: (batch_size, batch_size, embedding_dim)
    euclid_distances = torch.norm(diff, p=2, dim=2)  # shape: (batch_size, batch_size)
    
    # Compute pairwise Cosine distances
    normalized_embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    cosine_sim = torch.mm(normalized_embeddings, normalized_embeddings.t())  # similarity in [-1,1]
    cosine_distances = 1 - cosine_sim  # transform similarity into a distance measure

    # Store both metrics in a dictionary if needed
    metrics = {'euclid': euclid_distances, 'cosine': cosine_distances}
    
    # Select the distance measure based on the similarity_metric parameter
    if similarity_metric == 'euclid':
        distances = euclid_distances
    elif similarity_metric == 'cosine':
        distances = cosine_distances
    elif similarity_metric == 'both':
        distances = (euclid_distances + cosine_weight * cosine_distances) / (1 + cosine_weight)
    else:
        raise ValueError("Invalid similarity_metric. Choose among 'euclid', 'cosine', or 'both'.")
    
    # Create similarity matrix: 1 if labels are the same, 0 otherwise.
    labels = labels.view(-1, 1)
    label_matrix = (labels == labels.t()).float()  # shape: (batch_size, batch_size)
    
    # Remove self-comparisons (diagonal elements)
    mask = torch.eye(batch_size, device=embeddings.device).bool()
    distances = distances[~mask].view(batch_size, -1)
    label_matrix = label_matrix[~mask].view(batch_size, -1)
    
    # Calculate contrastive loss
    loss_similar = label_matrix * distances**2
    loss_dissimilar = (1 - label_matrix) * torch.clamp(margin - distances, min=0)**2
    loss = 0.5 * (loss_similar + loss_dissimilar)
    return loss.mean()

# ------------------------------
# Dataset Definition
# ------------------------------
class CreditCardFraudDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, seq_len):
        self.data = pd.read_csv(file_path)
        self.data['trans_date_trans_time'] = pd.to_datetime(self.data['trans_date_trans_time']).apply(lambda x: x.timestamp())
        self.label_encoder = LabelEncoder()
        self.data['category'] = self.label_encoder.fit_transform(self.data['category'])
        scaler = MinMaxScaler()
        self.data[['amt']] = scaler.fit_transform(self.data[['amt']])
        self.seq_len = seq_len
        self.sequences = []
        grouped = self.data.groupby('cc_num')
        for _, group in grouped:
            group = group[['category', 'amt', 'is_fraud', 'trans_date_trans_time']].values
            for i in range(len(group)):
                if i < self.seq_len - 1:
                    padding = [group[0]] * (self.seq_len - i - 1)
                    seq = padding + group[:i+1].tolist()
                else:
                    seq = group[i - self.seq_len + 1 : i + 1].tolist()
                label = group[i, 2]
                time_intervals = np.diff([s[-1] for s in seq], prepend=seq[0][-1])
                time_intervals = time_intervals.reshape(-1, 1)
                seq_features = np.array([s[:2] for s in seq])
                seq_features = np.concatenate((seq_features, time_intervals), axis=1)
                self.sequences.append((seq_features, label))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x_seq, y_label = self.sequences[idx]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.float32)

# ------------------------------
# Model Definition: TH_LSTM
# ------------------------------
class TH_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size):
        super(TH_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size

        # Time-aware state computation
        self.Wsh = nn.Linear(hidden_dim, hidden_dim)
        self.Wsx = nn.Linear(input_dim, hidden_dim)
        self.Wst = nn.Linear(1, hidden_dim)  # for delta_t
        self.bs = nn.Parameter(torch.zeros(hidden_dim))
        
        # Forget gate
        self.Wfh = nn.Linear(hidden_dim, hidden_dim)
        self.Wfx = nn.Linear(input_dim, hidden_dim)
        self.Wfs = nn.Linear(hidden_dim, hidden_dim)
        self.bf = nn.Parameter(torch.zeros(hidden_dim))
        
        # Input gate
        self.Wih = nn.Linear(hidden_dim, hidden_dim)
        self.Wix = nn.Linear(input_dim, hidden_dim)
        self.Wis = nn.Linear(hidden_dim, hidden_dim)
        self.bi = nn.Parameter(torch.zeros(hidden_dim))
        
        # Candidate cell state
        self.Wuh = nn.Linear(hidden_dim, hidden_dim)
        self.Wux = nn.Linear(input_dim, hidden_dim)
        self.Wus = nn.Linear(hidden_dim, hidden_dim)
        self.bu = nn.Parameter(torch.zeros(hidden_dim))
        
        # Time-aware gate
        self.WTh = nn.Linear(hidden_dim, hidden_dim)
        self.WTx = nn.Linear(input_dim, hidden_dim)
        self.WTs = nn.Linear(hidden_dim, hidden_dim)
        self.bT = nn.Parameter(torch.zeros(hidden_dim))
        
        # Output gate
        self.Woh = nn.Linear(hidden_dim, hidden_dim)
        self.Wox = nn.Linear(input_dim, hidden_dim)
        self.Wos = nn.Linear(hidden_dim, hidden_dim)
        self.bo = nn.Parameter(torch.zeros(hidden_dim))
        
        # Attention module
        self.Waq = nn.Linear(hidden_dim * 2, hidden_dim)
        self.Wah = nn.Linear(hidden_dim, hidden_dim)
        self.ba = nn.Parameter(torch.zeros(hidden_dim))
        self.vt = nn.Parameter(torch.randn(hidden_dim, 1))
        
        # Transactional representation expansion
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.We = nn.Linear(hidden_dim, hidden_dim)
        self.Wg = nn.Linear(input_dim, hidden_dim)  # from last transaction (excluding delta_t)
        self.bh = nn.Parameter(torch.zeros(hidden_dim))
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, X_seq):
        batch_size = X_seq.size(0)
        seq_len = X_seq.size(1)
        device = X_seq.device
        h_prev = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_prev = torch.zeros(batch_size, self.hidden_dim, device=device)
        historical_states = torch.zeros(batch_size, self.memory_size, self.hidden_dim, device=device)
        
        for t in range(seq_len):
            # x_t: first 2 features (category and amt)
            x_t = X_seq[:, t, :-1]
            # delta_t: last feature
            delta_t = X_seq[:, t, -1].view(-1, 1)
            
            s_t = torch.tanh(self.Wsh(h_prev) + self.Wsx(x_t) + self.Wst(delta_t) + self.bs)
            f_t = torch.sigmoid(self.Wfh(h_prev) + self.Wfx(x_t) + self.Wfs(s_t) + self.bf)
            i_t = torch.sigmoid(self.Wih(h_prev) + self.Wix(x_t) + self.Wis(s_t) + self.bi)
            T_t = torch.sigmoid(self.WTh(h_prev) + self.WTx(x_t) + self.WTs(s_t) + self.bT)
            zeta_t = torch.tanh(self.Wuh(h_prev) + self.Wux(x_t) + self.Wus(s_t) + self.bu)
            c_t = f_t * c_prev + i_t * zeta_t + T_t * s_t
            o_t = torch.sigmoid(self.Woh(h_prev) + self.Wox(x_t) + self.Wos(s_t) + self.bo)
            h_tilde_t = o_t * torch.tanh(c_t)
            
            historical_states = torch.cat((historical_states[:, 1:], h_tilde_t.unsqueeze(1)), dim=1)
            h_prev, c_prev = h_tilde_t, c_t

        q_t = torch.cat((h_tilde_t, c_t), dim=1)
        o_t_i = torch.tanh(self.Waq(q_t).unsqueeze(1) + self.Wah(historical_states))
        alpha_t_i = torch.exp(torch.matmul(o_t_i, self.vt)).squeeze(-1)
        alpha_t_i = alpha_t_i / torch.sum(alpha_t_i, dim=1, keepdim=True)
        e_t = torch.sum(alpha_t_i.unsqueeze(-1) * historical_states, dim=1)
        h_t = torch.tanh(self.W_h(h_tilde_t) + self.We(e_t) + self.Wg(X_seq[:, -1, :-1]) + self.bh)
        y_pred = torch.sigmoid(self.classifier(h_t))
        return y_pred, h_t  # Return both the prediction and the embedding

# ------------------------------
# Main Training and Evaluation
# ------------------------------
def main():
    # Choose device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 32
    input_dim = 2
    hidden_dim = 64
    memory_size = 10
    seq_len = 5
    epochs = 10
    contrastive_weight = 0.1  # Hyperparameter to balance contrastive loss
    similarity_metric = 'both'  # 'euclid', 'cosine', or 'both'
    cosine_weight = 0.5  # Only used if similarity_metric == 'both'
    
    train_file = "/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/Fraud_Train/fraudTraincc_num.csv"
    test_file  = "/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/Fraud_Train/fraudTestcc_num.csv"
    
    train_dataset = CreditCardFraudDataset(train_file, seq_len=seq_len)
    test_dataset  = CreditCardFraudDataset(test_file, seq_len=seq_len)
    
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = TH_LSTM(input_dim=input_dim, hidden_dim=hidden_dim, memory_size=memory_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    best_score = -float('inf')
    best_checkpoint = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.view(-1, 1).to(device)
            optimizer.zero_grad()
            
            # Get both prediction and embedding from the model
            y_pred, embeddings = model(X_batch)
            
            # Compute classification loss
            loss_cls = criterion(y_pred, y_batch)
            # Compute contrastive loss using the embeddings and true labels with the selected metric
            loss_ctr = contrastive_loss(embeddings, y_batch, margin=1.0, 
                                        similarity_metric=similarity_metric, cosine_weight=cosine_weight)
            # Total loss: classification loss + weighted contrastive loss
            loss = loss_cls + contrastive_weight * loss_ctr
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation on the test set
        model.eval()
        y_true_test, y_pred_test, y_pred_test_prob = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.view(-1, 1).to(device)
                y_pred, _ = model(X_batch)  # embeddings not needed for evaluation
                y_true_test.extend(y_batch.cpu().numpy())
                y_pred_test_prob.extend(y_pred.cpu().numpy())
                y_pred_test.extend((y_pred.cpu().numpy() >= 0.5).astype(int))
        
        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_precision = precision_score(y_true_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_true_test, y_pred_test, zero_division=0)
        test_f1  = f1_score(y_true_test, y_pred_test, zero_division=0)
        test_auc = roc_auc_score(y_true_test, y_pred_test_prob)
        
        print(f"\nEpoch {epoch+1} Test Results:")
        print(f"   Accuracy : {test_acc:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall   : {test_recall:.4f}")
        print(f"   F1 Score : {test_f1:.4f}")
        print(f"   AUC      : {test_auc:.4f}")
        
        # Save the best checkpoint (you can choose your own criteria)
        if test_f1 + test_auc > best_score:
            best_score = test_f1 + test_auc
            best_checkpoint = {
                "epoch": epoch+1,
                "test_accuracy": test_acc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_auc": test_auc
            }
            with open("/home/ducanh/Credit Card Transactions Fraud Detection/THLSTM/Fraud_train_Split/best_checkpoint_results_cc_num.json", "w") as f:
                json.dump(best_checkpoint, f, indent=4)
            print("New best checkpoint saved!")

    # Save the best model
    if best_checkpoint:
        torch.save(model.state_dict(), "/home/ducanh/Credit Card Transactions Fraud Detection/THLSTM/Fraud_train_Split/best_model_cc_num.pth")
        print("Best model saved!")

if __name__ == "__main__":
    main()
