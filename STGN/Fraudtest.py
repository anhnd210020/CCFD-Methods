import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from geopy.distance import geodesic  # Compute distance between transactions
from tqdm import tqdm
import json

# Dataset class with Transactional Extension Process
class CreditCardFraudDataset(Dataset):
    def __init__(self, file_path, seq_len, memory_size, scaler=None):
        # Read and preprocess the data
        self.data = pd.read_csv(file_path)
        # Convert transaction time to a timestamp (seconds)
        self.data['trans_date_trans_time'] = pd.to_datetime(self.data['trans_date_trans_time']).apply(lambda x: x.timestamp())
        
        # Compute previous coordinates and distance between transactions
        self.data['prev_lat'] = self.data.groupby('cc_num')['lat'].shift(1)
        self.data['prev_long'] = self.data.groupby('cc_num')['long'].shift(1)
        self.data['distance_km'] = self.data.apply(
            lambda row: geodesic((row['lat'], row['long']),
                                 (row['prev_lat'], row['prev_long'])).km 
                        if pd.notnull(row['prev_lat']) else 0, axis=1
        )
        
        # Encode the categorical variable
        self.label_encoder = LabelEncoder()
        self.data['category'] = self.label_encoder.fit_transform(self.data['category'])
        
        # Scale numerical features (only amt and distance_km)
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.data[['amt', 'distance_km']] = self.scaler.fit_transform(self.data[['amt', 'distance_km']])
        else:
            self.scaler = scaler
            self.data[['amt', 'distance_km']] = self.scaler.transform(self.data[['amt', 'distance_km']])
        
        self.seq_len = seq_len
        self.memory_size = memory_size
        self.sequences = []
        
        # Group by credit card number and create transaction sequences
        grouped = self.data.groupby('cc_num')
        for cc_num, group in grouped:
            # For each transaction, we extract four columns:
            # [category, amt, trans_date_trans_time, distance_km]
            # Later we will use the first two as base features and compute intervals for the last two.
            features = group[['category', 'amt', 'trans_date_trans_time', 'distance_km']].values
            labels = group['is_fraud'].values  
            
            for i in range(len(features)):
                # Pad sequence if needed
                if i < self.seq_len - 1:
                    padding = [features[0]] * (self.seq_len - i - 1)
                    seq = padding + features[:i + 1].tolist()
                else:
                    seq = features[i - self.seq_len + 1:i + 1].tolist()
                
                label = labels[i]
                
                # Compute time intervals from the transaction timestamps (column index 2)
                time_intervals = np.diff([s[2] for s in seq], prepend=seq[0][2]).reshape(-1, 1)
                # Base features: use first two columns (category, amt)
                base_features = np.array([s[:2] for s in seq])
                # Use the precomputed location difference (distance_km) as location interval (column index 3)
                location_intervals = np.array([s[3] for s in seq]).reshape(-1, 1)
                # Concatenate to form the final feature vector per transaction:
                # [category, amt, time_interval, location_interval]
                seq_features = np.concatenate((base_features, time_intervals, location_intervals), axis=1)
                
                # Create transactional extension memory E^u_i: select latest memory_size transactions
                memory = seq_features[-self.memory_size:]
                if len(memory) < self.memory_size:
                    memory = np.pad(memory, ((self.memory_size - len(memory), 0), (0, 0)), mode='edge')
                
                self.sequences.append((memory, seq_features[-1], label))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        memory, x_seq, y_label = self.sequences[idx]
        return (torch.tensor(memory, dtype=torch.float32), 
                torch.tensor(x_seq, dtype=torch.float32), 
                torch.tensor(y_label, dtype=torch.float32))
    
class STGN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size):
        super(STGN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size

        # Forget Gate
        self.Wfh = nn.Linear(hidden_dim, hidden_dim)
        self.Wfx = nn.Linear(input_dim, hidden_dim)
        self.bf = nn.Parameter(torch.zeros(hidden_dim))
        
        # Input Gate
        self.Wih = nn.Linear(hidden_dim, hidden_dim)
        self.Wix = nn.Linear(input_dim, hidden_dim)
        self.bi = nn.Parameter(torch.zeros(hidden_dim))
        
        # Time-aware Gate
        self.WTh = nn.Linear(hidden_dim, hidden_dim)
        self.WTx = nn.Linear(input_dim, hidden_dim)
        self.WTt = nn.Linear(1, hidden_dim)  # Time interval
        self.bT = nn.Parameter(torch.zeros(hidden_dim))
        
        # Location-aware Gate
        self.WLh = nn.Linear(hidden_dim, hidden_dim)
        self.WLx = nn.Linear(input_dim, hidden_dim)
        self.WLdelta = nn.Linear(1, hidden_dim)  # Location interval
        self.bL = nn.Parameter(torch.zeros(hidden_dim))
        
        # Spatial-Temporal Attention
        self.WzT = nn.Linear(1, hidden_dim)
        self.WzL = nn.Linear(1, hidden_dim)
        
        # Candidate Cell State
        self.Wuh = nn.Linear(hidden_dim, hidden_dim)
        self.Wux = nn.Linear(input_dim, hidden_dim)
        self.bu = nn.Parameter(torch.zeros(hidden_dim))
        
        # Output Gate
        self.Woh = nn.Linear(hidden_dim, hidden_dim)
        self.Wox = nn.Linear(input_dim, hidden_dim)
        self.Woz = nn.Linear(hidden_dim, hidden_dim)
        self.bo = nn.Parameter(torch.zeros(hidden_dim))
        
        # Spatial-Temporal Attention Module
        self.WIh = nn.Linear(hidden_dim, hidden_dim)
        self.WWh = nn.Linear(hidden_dim, hidden_dim)
        self.WIAT = nn.Linear(1, hidden_dim)
        self.WIAL = nn.Linear(1, hidden_dim)
        self.bo_attention = nn.Parameter(torch.zeros(hidden_dim))

        # Representation Interaction Module
        self.Wr = nn.Linear(hidden_dim, hidden_dim)
        self.br = nn.Parameter(torch.zeros(hidden_dim))
        self.Wih_final = nn.Linear(hidden_dim, hidden_dim)
        self.Ws_final = nn.Linear(hidden_dim, hidden_dim)
        self.Wr_final = nn.Linear(hidden_dim, hidden_dim)
        self.bh_final = nn.Parameter(torch.zeros(hidden_dim))
        
        # Prediction Layer
        self.Wy = nn.Linear(hidden_dim, 1)
        self.by = nn.Parameter(torch.zeros(1))
    
    def forward(self, memory, X_seq):
        batch_size, seq_len, _ = X_seq.shape
        h_prev = torch.zeros(batch_size, self.hidden_dim).to(X_seq.device)
        c_prev = torch.zeros(batch_size, self.hidden_dim).to(X_seq.device)
        
        memory = []  # Store past states for attention

        for t in range(seq_len):
            x_t = X_seq[:, t, :-2]  # Features except the last two (time, location intervals)
            delta_t = X_seq[:, t, -2].view(-1, 1)  # Time interval
            delta_L = X_seq[:, t, -1].view(-1, 1)  # Location interval
            
            # Forget gate
            f_t = torch.sigmoid(self.Wfh(h_prev) + self.Wfx(x_t) + self.bf)
            
            # Input gate
            i_t = torch.sigmoid(self.Wih(h_prev) + self.Wix(x_t) + self.bi)
            
            # Time-aware gate
            T_t = torch.sigmoid(self.WTh(h_prev) + self.WTx(x_t) + self.WTt(delta_t) + self.bT)
            
            # Location-aware gate
            L_t = torch.sigmoid(self.WLh(h_prev) + self.WLx(x_t) + self.WLdelta(delta_L) + self.bL)
            
            # Spatial-Temporal Attention
            zeta = torch.tanh(self.WzT(delta_t) + self.WzL(delta_L))
            
            # Candidate cell state
            c_tilde = torch.tanh(self.Wuh(h_prev) + self.Wux(x_t) + self.bu)
            
            # Cell state update
            c_t = f_t * c_prev + i_t * c_tilde * T_t * L_t
            
            # Output gate
            o_t = torch.sigmoid(self.Woh(h_prev) + self.Wox(x_t) + self.Woz(zeta) + self.bo)
            
            # Hidden state update
            h_t = o_t * torch.tanh(c_t)
            
            memory.append(h_t)
            
            # Spatial-Temporal Attention Calculation
            if len(memory) > self.memory_size:
                memory.pop(0)  # Maintain memory size constraint
            
            memory_tensor = torch.stack(memory, dim=1)
            
            I_t = self.WIh(h_t) + self.WWh(memory_tensor) + self.WIAT(delta_t) + self.WIAL(delta_L)
            o_t_attention = torch.tanh(I_t + self.bo_attention)
            alpha_t = torch.softmax(torch.matmul(o_t_attention, memory_tensor.transpose(1, 2)), dim=-1)
            
            # Compute final representation s_t
            s_t = torch.sum(alpha_t * memory_tensor, dim=1)
            
             # Compute user representation r_u
            r_u = torch.tanh(self.Wr(memory_tensor.mean(dim=1)) + self.br)
            
            # Compute final transaction representation
            h_t = torch.tanh(self.Wih_final(h_t) + self.Ws_final(s_t) + self.Wr_final(r_u) + self.bh_final)
            
        # Prediction layer
        y_hat = torch.sigmoid(self.Wy(h_t) + self.by)
        
        return y_hat  # Return the predicted fraud probability

# Training Pipeline
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_file = "/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv"
    test_file = "/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTest.csv"
    
    seq_len = 5
    memory_size = 10
    batch_size = 32
    input_dim = 2  # Features: category, amt, distance_km (excluding timestamp for LSTM)
    hidden_dim = 64
    epochs = 10
    
    train_dataset = CreditCardFraudDataset(train_file, seq_len=seq_len, memory_size=memory_size)
    test_dataset = CreditCardFraudDataset(test_file, seq_len=seq_len, memory_size=memory_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = STGN_LSTM(input_dim=input_dim, hidden_dim=hidden_dim, memory_size=memory_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    best_score = -float('inf')
    best_checkpoint = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for memory_batch, X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            memory_batch, X_batch, y_batch = memory_batch.to(device), X_batch.to(device), y_batch.view(-1, 1).to(device)
            optimizer.zero_grad()
            y_pred = model(memory_batch, X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        y_true_test, y_pred_test, y_pred_test_prob = [], [], []
        with torch.no_grad():
            for memory_batch, X_batch, y_batch in test_loader:
                memory_batch, X_batch, y_batch = memory_batch.to(device), X_batch.to(device), y_batch.view(-1, 1).to(device)
                y_pred = model(memory_batch, X_batch)
                y_true_test.extend(y_batch.cpu().numpy())
                y_pred_test_prob.extend(y_pred.cpu().numpy())
                y_pred_test.extend((y_pred.cpu().numpy() >= 0.5).astype(int))
        
        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
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
            with open("/home/ducanh/Credit Card Transactions Fraud Detection/STGN/best_checkpoint_results.json", "w") as f:
                json.dump(best_checkpoint, f, indent=4)
            print(f"New best checkpoint saved!")

    # Lưu mô hình tốt nhất
    if best_checkpoint:
        torch.save(model.state_dict(), "/home/ducanh/Credit Card Transactions Fraud Detection/STGN/best_model.pth")
        print(f"Best model saved!")
    
if __name__ == "__main__":
    main()
