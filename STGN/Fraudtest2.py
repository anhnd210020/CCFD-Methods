import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from geopy.distance import geodesic
from tqdm import tqdm
class CreditCardFraudDataset(Dataset):
    def __init__(self, file_path, memory_size=10, scaler=None):
        self.data = pd.read_csv(file_path)
        self.data['trans_date_trans_time'] = pd.to_datetime(self.data['trans_date_trans_time']).apply(lambda x: x.timestamp())
        self.data['prev_lat'] = self.data.groupby('cc_num')['lat'].shift(1)
        self.data['prev_long'] = self.data.groupby('cc_num')['long'].shift(1)
        self.data['distance_km'] = self.data.apply(lambda row: geodesic((row['lat'], row['long']), (row['prev_lat'], row['prev_long'])).km if pd.notnull(row['prev_lat']) else 0, axis=1)
        self.label_encoder = LabelEncoder()
        self.data['category'] = self.label_encoder.fit_transform(self.data['category'])
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.data[['amt', 'distance_km']] = self.scaler.fit_transform(self.data[['amt', 'distance_km']])
        else:
            self.scaler = scaler
            self.data[['amt', 'distance_km']] = self.scaler.transform(self.data[['amt', 'distance_km']])
        self.memory_size = memory_size
        self.sequences = []
        grouped = self.data.groupby('cc_num')
        for cc_num, group in tqdm(grouped):
            features = group[['category', 'amt', 'trans_date_trans_time', 'distance_km']].values
            labels = group['is_fraud'].values
            for i in range(len(features)):
                start_idx = max(0, i - self.memory_size + 1)
                cluster = features[start_idx : i + 1]
                if len(cluster) < self.memory_size:
                    shortfall = self.memory_size - len(cluster)
                    padding = [cluster[0]] * shortfall
                    cluster = np.concatenate([padding, cluster], axis=0)
                time_stamps = cluster[:, 2]
                time_intervals = np.diff(time_stamps, prepend=time_stamps[0]).reshape(-1, 1)
                base_features = cluster[:, :2]
                location_intervals = cluster[:, 3].reshape(-1, 1)
                final_cluster = np.concatenate([base_features, time_intervals, location_intervals], axis=1)
                label = labels[i]
                self.sequences.append((final_cluster, label))
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        cluster, label = self.sequences[idx]
        cluster_tensor = torch.tensor(cluster, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return cluster_tensor, label_tensor

class STGN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size):
        super(STGN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
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
        self.WLh = nn.Linear(hidden_dim, hidden_dim)
        self.WLx = nn.Linear(input_dim, hidden_dim)
        self.WLdelta = nn.Linear(1, hidden_dim)
        self.bL = nn.Parameter(torch.zeros(hidden_dim))
        self.WzT = nn.Linear(1, hidden_dim)
        self.WzL = nn.Linear(1, hidden_dim)
        self.Wuh = nn.Linear(hidden_dim, hidden_dim)
        self.Wux = nn.Linear(input_dim, hidden_dim)
        self.bu = nn.Parameter(torch.zeros(hidden_dim))
        self.Woh = nn.Linear(hidden_dim, hidden_dim)
        self.Wox = nn.Linear(input_dim, hidden_dim)
        self.Woz = nn.Linear(hidden_dim, hidden_dim)
        self.bo = nn.Parameter(torch.zeros(hidden_dim))
        self.WIh = nn.Linear(hidden_dim, hidden_dim)
        self.WWh = nn.Linear(hidden_dim, hidden_dim)
        self.WIAT = nn.Linear(1, hidden_dim)
        self.WIAL = nn.Linear(1, hidden_dim)
        self.bo_attention = nn.Parameter(torch.zeros(hidden_dim))
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, X_seq):
        batch_size, mem_len, _ = X_seq.shape
        h_prev = torch.zeros(batch_size, self.hidden_dim, device=X_seq.device)
        c_prev = torch.zeros(batch_size, self.hidden_dim, device=X_seq.device)
        memory = []
        for t in range(mem_len):
            x_t = X_seq[:, t, :2]
            delta_t = X_seq[:, t, 2].view(-1, 1)
            delta_L = X_seq[:, t, 3].view(-1, 1)
            f_t = torch.sigmoid(self.Wfh(h_prev) + self.Wfx(x_t) + self.bf)
            i_t = torch.sigmoid(self.Wih(h_prev) + self.Wix(x_t) + self.bi)
            T_t = torch.sigmoid(self.WTh(h_prev) + self.WTx(x_t) + self.WTt(delta_t) + self.bT)
            L_t = torch.sigmoid(self.WLh(h_prev) + self.WLx(x_t) + self.WLdelta(delta_L) + self.bL)
            zeta = torch.tanh(self.WzT(delta_t) + self.WzL(delta_L))
            c_tilde = torch.tanh(self.Wuh(h_prev) + self.Wux(x_t) + self.bu)
            c_t = f_t * c_prev + i_t * c_tilde * T_t * L_t
            o_t = torch.sigmoid(self.Woh(h_prev) + self.Wox(x_t) + self.Woz(zeta) + self.bo)
            h_t = o_t * torch.tanh(c_t)
            memory.append(h_t)
            if len(memory) > self.memory_size:
                memory.pop(0)
            memory_tensor = torch.stack(memory, dim=1)
            h_t_unsq = h_t.unsqueeze(1)
            delta_t_unsq = delta_t.unsqueeze(1)
            delta_L_unsq = delta_L.unsqueeze(1)
            I_t = self.WIh(h_t_unsq) + self.WWh(memory_tensor) + self.WIAT(delta_t_unsq) + self.WIAL(delta_L_unsq)
            o_t_attention = torch.tanh(I_t + self.bo_attention)
            alpha_t = torch.matmul(o_t_attention, memory_tensor.transpose(1, 2))
            alpha_t = torch.softmax(alpha_t, dim=-1)
            s_t = torch.sum(alpha_t.unsqueeze(-1) * memory_tensor.unsqueeze(2), dim=1)
            s_t = torch.mean(s_t, dim=1)
            h_prev = s_t
            c_prev = c_t
        logits = self.fc(h_prev)
        return logits

def main():
    import json
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    batch_size = 32
    input_dim = 2      # using first 2 features (e.g., category and amt)
    hidden_dim = 64
    memory_size = 10
    epochs = 5

    # File paths (adjust paths as needed)
    train_file = "/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv"
    test_file  = "/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTest.csv"

    # Create datasets and dataloaders
    train_dataset = CreditCardFraudDataset(train_file, memory_size=memory_size)
    test_dataset  = CreditCardFraudDataset(test_file, memory_size=memory_size)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, optimizer, and loss function
    model = STGN_LSTM(input_dim=input_dim, hidden_dim=hidden_dim, memory_size=memory_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_score = -float('inf')
    best_checkpoint = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for cluster_batch, label_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            cluster_batch = cluster_batch.to(device)
            # Add an extra dimension to labels to match the output shape
            label_batch = label_batch.unsqueeze(1).to(device)

            logits = model(cluster_batch)
            loss = criterion(logits, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}, Average Training Loss: {avg_loss:.4f}")

        # Evaluation on the test set
        model.eval()
        y_true_test, y_pred_test, y_pred_test_prob = [], [], []
        with torch.no_grad():
            for cluster_batch, label_batch in test_loader:
                cluster_batch = cluster_batch.to(device)
                label_batch = label_batch.unsqueeze(1).to(device)
                logits = model(cluster_batch)
                # Since the model outputs logits, apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                y_true_test.extend(label_batch.cpu().numpy())
                y_pred_test_prob.extend(probs.cpu().numpy())
                y_pred_test.extend((probs.cpu().numpy() >= 0.5).astype(int))

        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_f1  = f1_score(y_true_test, y_pred_test, zero_division=0)
        test_auc = roc_auc_score(y_true_test, y_pred_test_prob)

        print(f"Epoch {epoch+1} Test Results:")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   F1 Score: {test_f1:.4f}")
        print(f"   AUC: {test_auc:.4f}")

        # Save the best checkpoint based on test F1 + AUC score
        if test_f1 + test_auc > best_score:
            best_score = test_f1 + test_auc
            best_checkpoint = {
                "epoch": epoch + 1,
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "test_auc": test_auc
            }
            checkpoint_path = "/home/ducanh/Credit Card Transactions Fraud Detection/STGN_LSTM/best_checkpoint_results.json"
            with open(checkpoint_path, "w") as f:
                json.dump(best_checkpoint, f, indent=4)
            print("New best checkpoint saved!")

    # Save the best model after training
    if best_checkpoint:
        model_path = "/home/ducanh/Credit Card Transactions Fraud Detection/STGN_LSTM/best_model.pth"
        torch.save(model.state_dict(), model_path)
        print("Best model saved!")

if __name__ == "__main__":
    main()

