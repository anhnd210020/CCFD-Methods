import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm  # Thư viện để hiển thị progress bar

# Dataset class for grouped sequences of transactions by `cc_num`
class CreditCardFraudDataset(Dataset):
    def __init__(self, file_path, seq_len):
        self.data = pd.read_csv(file_path)

        # Convert date to timestamp
        self.data['trans_date_trans_time'] = pd.to_datetime(self.data['trans_date_trans_time']).apply(lambda x: x.timestamp())

        # Encode category column
        self.label_encoder = LabelEncoder()
        self.data['category'] = self.label_encoder.fit_transform(self.data['category'])

        # Normalize features (chỉ normalize cột số tiền)
        scaler = MinMaxScaler()
        self.data[['amt']] = scaler.fit_transform(self.data[['amt']])

        # Group transactions by `cc_num` and create sequences
        self.seq_len = seq_len
        self.sequences = []
        grouped = self.data.groupby('cc_num')
        for _, group in grouped:
            # Chọn các cột theo thứ tự: category, amt, is_fraud, trans_date_trans_time
            group = group[['category', 'amt', 'is_fraud', 'trans_date_trans_time']].values
            for i in range(len(group)):
                # Tạo sequence có độ dài cố định seq_len
                if i < self.seq_len - 1:
                    padding = [group[0]] * (self.seq_len - i - 1)
                    seq = padding + group[:i + 1].tolist()
                else:
                    seq = group[i - self.seq_len + 1:i + 1].tolist()

                # Lấy nhãn của giao dịch hiện tại (is_fraud) từ cột thứ 3 (index 2)
                label = group[i, 2]

                # Tính khoảng cách thời gian giữa các giao dịch (delta_t)
                time_intervals = np.diff([s[-1] for s in seq], prepend=seq[0][-1])
                time_intervals = time_intervals.reshape(-1, 1)

                # Chỉ giữ lại các đặc trưng: category và amt (loại bỏ is_fraud)
                seq_features = np.array([s[:2] for s in seq])
                # Nối thêm delta_t làm đặc trưng cuối cùng
                seq_features = np.concatenate((seq_features, time_intervals), axis=1)
                # Kết quả: mỗi giao dịch có 3 đặc trưng: [category, amt, delta_t]

                self.sequences.append((seq_features, label))

        # (Tuỳ chọn) Lưu dữ liệu đã xử lý nếu cần thiết
        processed_data_path = "/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/Processed_data/Processed_Fraud_Detection_Data.csv"
        self.data.to_csv(processed_data_path, index=False)
        print(f"Processed dataset saved at {processed_data_path}!")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x_seq, y_label = self.sequences[idx]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.float32)

# Define the TH_LSTM model
class TH_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size):
        super(TH_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size

        # Time-aware state computation
        self.Wsh = nn.Linear(hidden_dim, hidden_dim)
        self.Wsx = nn.Linear(input_dim, hidden_dim)
        self.Wst = nn.Linear(1, hidden_dim)  # Dùng cho delta_t
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
        self.Wg = nn.Linear(input_dim, hidden_dim)  # Dùng cho đặc trưng của giao dịch cuối
        self.bh = nn.Parameter(torch.zeros(hidden_dim))

        # Output layer
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, X_seq):
        batch_size = X_seq.size(0)
        seq_len = X_seq.size(1)
        h_prev = torch.zeros(batch_size, self.hidden_dim).to(X_seq.device)
        c_prev = torch.zeros(batch_size, self.hidden_dim).to(X_seq.device)
        historical_states = torch.zeros(batch_size, self.memory_size, self.hidden_dim).to(X_seq.device)

        for t in range(seq_len):
            # Lấy các đặc trưng (loại bỏ cột cuối - delta_t)
            x_t = X_seq[:, t, :-1]  # shape: (batch_size, input_dim) với input_dim = 2
            delta_t = X_seq[:, t, -1].view(-1, 1)  # delta_t có shape: (batch_size, 1)

            # Tính time-aware state
            s_t = torch.tanh(self.Wsh(h_prev) + self.Wsx(x_t) + self.Wst(delta_t) + self.bs)

            # Forget gate
            f_t = torch.sigmoid(self.Wfh(h_prev) + self.Wfx(x_t) + self.Wfs(s_t) + self.bf)
            # Input gate
            i_t = torch.sigmoid(self.Wih(h_prev) + self.Wix(x_t) + self.Wis(s_t) + self.bi)
            # Time-aware gate
            T_t = torch.sigmoid(self.WTh(h_prev) + self.WTx(x_t) + self.WTs(s_t) + self.bT)

            # Candidate cell state
            zeta_t = torch.tanh(self.Wuh(h_prev) + self.Wux(x_t) + self.Wus(s_t) + self.bu)

            # New cell state
            c_t = f_t * c_prev + i_t * zeta_t + T_t * s_t

            # Output gate và hidden state
            o_t = torch.sigmoid(self.Woh(h_prev) + self.Wox(x_t) + self.Wos(s_t) + self.bo)
            h_tilde_t = o_t * torch.tanh(c_t)

            # Cập nhật historical states cho cơ chế attention
            historical_states = torch.cat((historical_states[:, 1:], h_tilde_t.unsqueeze(1)), dim=1)
            h_prev, c_prev = h_tilde_t, c_t

        # Attention mechanism: kết hợp hidden state và cell state hiện tại
        q_t = torch.cat((h_tilde_t, c_t), dim=1)  # shape: (batch_size, hidden_dim * 2)
        o_t_i = torch.tanh(self.Waq(q_t).unsqueeze(1) + self.Wah(historical_states))
        alpha_t_i = torch.exp(torch.matmul(o_t_i, self.vt)).squeeze(-1)
        alpha_t_i = alpha_t_i / torch.sum(alpha_t_i, dim=1, keepdim=True)
        e_t = torch.sum(alpha_t_i.unsqueeze(-1) * historical_states, dim=1)

        # Final transactional representation
        # Lấy đặc trưng của giao dịch cuối cùng (trước delta_t) từ X_seq[:, -1, :-1]
        h_t = torch.tanh(self.W_h(h_tilde_t) + self.We(e_t) + self.Wg(X_seq[:, -1, :-1]) + self.bh)

        # Phân loại cuối cùng
        y_pred = torch.sigmoid(self.classifier(h_t))
        return y_pred

# --- Main ---
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    input_dim = 2   # Chỉ gồm 2 đặc trưng: category và amt (delta_t được thêm sau)
    hidden_dim = 64
    memory_size = 10
    seq_len = 5

    # Đường dẫn file CSV (cập nhật đường dẫn phù hợp với hệ thống của bạn)
    train_file = "/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv"
    test_file = "/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTest.csv"

    # Load datasets
    train_dataset = CreditCardFraudDataset(train_file, seq_len=seq_len)
    test_dataset = CreditCardFraudDataset(test_file, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TH_LSTM(input_dim=input_dim, hidden_dim=hidden_dim, memory_size=memory_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Lưu toàn bộ mô hình
    full_model_path = "/home/ducanh/Credit Card Transactions Fraud Detection/ModelSave/Fraudtest_data_model.pth"
    torch.save(model, full_model_path)
    print(f"Full model saved successfully at {full_model_path}!")

    # Training loop với tqdm để hiển thị progress bar
    epochs = 1  # Thay đổi số epoch nếu cần
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        y_true_epoch, y_pred_epoch = [], []
        
        # Sử dụng tqdm để hiển thị progress bar cho từng batch trong epoch
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            X_batch, y_batch = X_batch.to(device), y_batch.view(-1, 1).to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            y_true_epoch.extend(y_batch.cpu().numpy())
            y_pred_epoch.extend((y_pred.detach().cpu().numpy() >= 0.5).astype(int))

        epoch_accuracy = accuracy_score(y_true_epoch, y_pred_epoch)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(train_loader):.4f} | Accuracy: {epoch_accuracy:.4f}")

    # Testing loop
    model.eval()
    y_true_test, y_pred_test_prob = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.view(-1, 1).to(device)
            y_pred = model(X_batch)
            y_true_test.extend(y_batch.cpu().numpy())
            y_pred_test_prob.extend(y_pred.cpu().numpy())

    # Đánh giá mô hình với các ngưỡng khác nhau
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        y_pred_test_binary = (np.array(y_pred_test_prob) >= threshold).astype(int)
        accuracy = accuracy_score(y_true_test, y_pred_test_binary)
        precision = precision_score(y_true_test, y_pred_test_binary)
        recall = recall_score(y_true_test, y_pred_test_binary)
        f1 = f1_score(y_true_test, y_pred_test_binary)
        auc = roc_auc_score(y_true_test, y_pred_test_binary)
        print(f"Threshold: {threshold:.2f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Đánh giá ngưỡng cụ thể
    threshold = 0.9999999
    y_pred_test_binary = (np.array(y_pred_test_prob) >= threshold).astype(int)
    accuracy = accuracy_score(y_true_test, y_pred_test_binary)
    precision = precision_score(y_true_test, y_pred_test_binary)
    recall = recall_score(y_true_test, y_pred_test_binary)
    f1 = f1_score(y_true_test, y_pred_test_binary)
    auc = roc_auc_score(y_true_test, y_pred_test_binary)
    print(f"Threshold: {threshold:.7f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"Min Prob: {min(y_pred_test_prob)} | Max Prob: {max(y_pred_test_prob)}")
