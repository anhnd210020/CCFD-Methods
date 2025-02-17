import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from geopy.distance import geodesic  # Compute distance between transactions

# Dataset class with only 4 features: category, amt, distance_km, delta_t
class CreditCardFraudDataset(Dataset):
    def __init__(self, file_path, seq_len):
        self.data = pd.read_csv(file_path)

        # Convert transaction date to timestamp
        self.data['trans_date_trans_time'] = pd.to_datetime(self.data['trans_date_trans_time']).apply(lambda x: x.timestamp())

        # Encode category column
        self.label_encoder = LabelEncoder()
        self.data['category'] = self.label_encoder.fit_transform(self.data['category'])

        # Normalize numerical features
        scaler = MinMaxScaler()
        self.data[['amt']] = scaler.fit_transform(self.data[['amt']])

        # Compute distance between transactions
        self.data['prev_lat'] = self.data.groupby('cc_num')['lat'].shift(1)
        self.data['prev_long'] = self.data.groupby('cc_num')['long'].shift(1)
        self.data['distance_km'] = self.data.apply(
            lambda row: geodesic((row['lat'], row['long']), (row['prev_lat'], row['prev_long'])).km
            if not pd.isnull(row['prev_lat']) else 0, axis=1
        )

        # Normalize distance
        self.data[['distance_km']] = scaler.fit_transform(self.data[['distance_km']])

        # Group transactions by `cc_num` and create sequences
        self.seq_len = seq_len
        self.sequences = []
        grouped = self.data.groupby('cc_num')

        for cc_num, group in grouped:
            # Select features without is_fraud
            features = group[['category', 'amt', 'distance_km', 'trans_date_trans_time']].values
            labels = group['is_fraud'].values  # Get labels separately

            for i in range(len(features)):
                if i < self.seq_len - 1:
                    padding = [features[0]] * (self.seq_len - i - 1)
                    seq = padding + features[:i + 1].tolist()
                else:
                    seq = features[i - self.seq_len + 1:i + 1].tolist()
                
                label = labels[i]  # Correct label assignment
                time_intervals = np.diff([s[-1] for s in seq], prepend=seq[0][-1])  # Time differences
                time_intervals = time_intervals.reshape(-1, 1)

                # Final sequence with only 4 features: category, amt, distance_km, delta_t
                seq_features = np.array([s[:-1] for s in seq])  # Remove timestamp
                seq_features = np.concatenate((seq_features, time_intervals), axis=1)  # Add time intervals

                self.sequences.append((seq_features, label))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x_seq, y_label = self.sequences[idx]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.float32)


# LSTM Model for Fraud Detection with 4 features
class STGN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(STGN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM Gates with Time-aware module
        self.Wfh = nn.Linear(hidden_dim, hidden_dim)
        self.Wfx = nn.Linear(input_dim, hidden_dim)
        self.bf = nn.Parameter(torch.zeros(hidden_dim))

        self.Wih = nn.Linear(hidden_dim, hidden_dim)
        self.Wix = nn.Linear(input_dim, hidden_dim)
        self.bi = nn.Parameter(torch.zeros(hidden_dim))

        self.WTh = nn.Linear(hidden_dim, hidden_dim)
        self.WTx = nn.Linear(input_dim, hidden_dim)
        self.WTt = nn.Linear(1, hidden_dim)
        self.bT = nn.Parameter(torch.zeros(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, X_seq):
        batch_size, seq_len, _ = X_seq.shape
        h_prev = torch.zeros(batch_size, self.hidden_dim).to(X_seq.device)
        c_prev = torch.zeros(batch_size, self.hidden_dim).to(X_seq.device)

        for t in range(seq_len):
            x_t = X_seq[:, t, :-1]  # Features excluding delta_t
            delta_t = X_seq[:, t, -1].view(-1, 1)  # Time interval (delta_t)  
            
            # Gates
            f_t = torch.sigmoid(self.Wfh(h_prev) + self.Wfx(x_t) + self.bf)
            i_t = torch.sigmoid(self.Wih(h_prev) + self.Wix(x_t) + self.bi)
            T_t = torch.sigmoid(self.WTh(h_prev) + self.WTx(x_t) + self.WTt(delta_t) + self.bT)
            
            c_t = f_t * c_prev + i_t * torch.tanh(T_t * delta_t)
            h_t = torch.tanh(c_t)

        return torch.sigmoid(self.classifier(h_t))

batch_size = 32
input_dim = 2 
hidden_dim = 64
seq_len = 5

train_dataset = CreditCardFraudDataset("/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv", seq_len)
test_dataset = CreditCardFraudDataset("/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTest.csv", seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)