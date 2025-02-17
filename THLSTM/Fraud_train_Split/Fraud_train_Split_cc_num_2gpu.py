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
        return y_pred
# ------------------------------
# Main Training and Evaluation
# ------------------------------
def main():
    # Chọn thiết bị chạy (GPU nếu có, nếu không dùng CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 32
    input_dim = 2
    hidden_dim = 64
    memory_size = 10
    seq_len = 5
    epochs = 10
    
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
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Đánh giá trên tập test
        model.eval()
        y_true_test, y_pred_test, y_pred_test_prob = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.view(-1, 1).to(device)
                y_pred = model(X_batch)
                y_true_test.extend(y_batch.cpu().numpy())
                y_pred_test_prob.extend(y_pred.cpu().numpy())
                y_pred_test.extend((y_pred.cpu().numpy() >= 0.5).astype(int))
        
        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_f1  = f1_score(y_true_test, y_pred_test, zero_division=0)
        test_auc = roc_auc_score(y_true_test, y_pred_test_prob)
        
        print(f"\nEpoch {epoch+1} Test Results:")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   F1 Score: {test_f1:.4f}")
        print(f"   AUC: {test_auc:.4f}")
        
        # Lưu checkpoint tốt nhất
        if test_f1 + test_auc > best_score:
            best_score = test_f1 + test_auc
            best_checkpoint = {
                "epoch": epoch+1,
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "test_auc": test_auc
            }
            with open("/home/ducanh/Credit Card Transactions Fraud Detection/THLSTM/Fraud_train_Split/best_checkpoint_results_cc_num.json", "w") as f:
                json.dump(best_checkpoint, f, indent=4)
            print(f"New best checkpoint saved!")

    # Lưu mô hình tốt nhất
    if best_checkpoint:
        torch.save(model.state_dict(), "/home/ducanh/Credit Card Transactions Fraud Detection/THLSTM/Fraud_train_Split/best_model_cc_num.pth")
        print(f"Best model saved!")

if __name__ == "__main__":
    main()
