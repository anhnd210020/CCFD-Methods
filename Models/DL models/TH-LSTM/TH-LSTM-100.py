# %% Import các thư viện cần thiết
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# %% Đọc và xử lý dữ liệu
df = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv')

# Xử lý thời gian
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])
df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 else 
                                              ('10-20' if x < 20 else 
                                               ('20-30' if x < 30 else 
                                                ('30-40' if x < 40 else 
                                                 ('40-50' if x < 50 else 
                                                  ('50-60' if x < 60 else 
                                                   ('60-70' if x < 70 else 
                                                    ('70-80' if x < 80 else 'Above 80'))))))))

# Drop các cột không cần thiết (nhưng giữ lại 'trans_date_trans_time' và 'cc_num')
drop_col = ['Unnamed: 0', 'merchant', 'first', 'last', 'street', 'city', 'state', 'lat',
            'long','dob', 'unix_time', 'cust_age', 'merch_lat', 'merch_long', 'city_pop']
df.drop(drop_col, axis=1, inplace=True)

# Pivot table cho cust_age_groups
age_piv = pd.pivot_table(data=df, index='cust_age_groups', columns='is_fraud', values='amt', aggfunc=np.mean)
age_piv.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for k, v in zip(age_piv.index.values, age_piv.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Pivot table cho category
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for k, v in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# Pivot table cho job
job_piv = pd.pivot_table(data=df, index='job', columns='is_fraud', values='amt', aggfunc=np.mean)
job_dic = {k: v for k, v in zip(job_piv.index.values, job_piv.reset_index().index.values)}
df['job'] = df['job'].map(job_dic)

df['trans_hour'] = df['trans_hour'].astype('int')
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# Chuyển đổi trans_date_trans_time thành timestamp để scale
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

# Train-test split
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.to_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_fraudTrain.csv', index=False)
test.to_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_fraudTest.csv', index=False)

# Drop cột 'trans_num' nếu có
if 'trans_num' in train.columns:
    train.drop('trans_num', axis=1, inplace=True)
    test.drop('trans_num', axis=1, inplace=True)

# Tách features và label
y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)
y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)
print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# Scaling dữ liệu
sc = StandardScaler()
X_train_sc = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
X_test_sc = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)


# %% Tạo các sequence
sequence_length = 100
def create_sequences(df, sequence_length):
    sequences, labels = [], []
    grouped = df.groupby('cc_num')
    for user_id, group in grouped:
        group = group.sort_values(by='trans_date_trans_time')
        values = group.drop(columns=['is_fraud', 'cc_num']).values
        targets = group['is_fraud'].values
        n = len(group)
        for i in range(n):
            if i < sequence_length:
                pad_needed = sequence_length - (i + 1)
                pad = np.repeat(values[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, values[:i+1]), axis=0)
            else:
                seq = values[i-sequence_length+1:i+1]
            sequences.append(seq)
            labels.append(targets[i])
    return np.array(sequences), np.array(labels)

train_seq_df = X_train_sc.copy()
train_seq_df['is_fraud'] = y_train.values
test_seq_df = X_test_sc.copy()
test_seq_df['is_fraud'] = y_test.values

X_train_seq, y_train_seq = create_sequences(train_seq_df, sequence_length)
X_test_seq, y_test_seq = create_sequences(test_seq_df, sequence_length)
print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

# 2. Xây dựng mô hình TH‑LSTM
class THLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(THLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        # Tính trạng thái thời gian s_t nhận h_{t-1}, x_t, và Δt (đã reshape về 1 chiều)
        self.linear_s = nn.Linear(hidden_size + input_size + 1, hidden_size)
        # Các cổng: forget (f_t), input (i_t), time-aware (T_t), candidate (ζ_t) và output (o_t)
        self.linear_f = nn.Linear(hidden_size + input_size + hidden_size, hidden_size)
        self.linear_i = nn.Linear(hidden_size + input_size + hidden_size, hidden_size)
        self.linear_T = nn.Linear(hidden_size + input_size + hidden_size, hidden_size)
        self.linear_u = nn.Linear(hidden_size + input_size + hidden_size, hidden_size)
        self.linear_o = nn.Linear(hidden_size + input_size + hidden_size, hidden_size)
    def forward(self, x_t, delta_t, h_prev, c_prev):
        # delta_t: [batch, 1]
        s_t = torch.tanh(self.linear_s(torch.cat([h_prev, x_t, delta_t], dim=1)))
        f_t = torch.sigmoid(self.linear_f(torch.cat([h_prev, x_t, s_t], dim=1)))
        i_t = torch.sigmoid(self.linear_i(torch.cat([h_prev, x_t, s_t], dim=1)))
        T_t = torch.sigmoid(self.linear_T(torch.cat([h_prev, x_t, s_t], dim=1)))
        zeta_t = torch.tanh(self.linear_u(torch.cat([h_prev, x_t, s_t], dim=1)))
        c_t = f_t * c_prev + i_t * zeta_t + T_t * s_t
        o_t = torch.sigmoid(self.linear_o(torch.cat([h_prev, x_t, s_t], dim=1)))
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class THLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_memory=10, time_idx=0):
        super(THLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_memory = attention_memory  # số bước lịch sử để tính attention
        self.time_idx = time_idx  # chỉ số của cột thời gian trong vector đầu vào
        self.cells = nn.ModuleList([THLSTMCell(input_size if i==0 else hidden_size, hidden_size) for i in range(num_layers)])
        # Module Attention: chuyển q_t và các hidden state lịch sử
        self.linear_aq = nn.Linear(2 * hidden_size, hidden_size)
        self.linear_ah = nn.Linear(hidden_size, hidden_size)
        self.bias_a = nn.Parameter(torch.zeros(hidden_size))
        self.v = nn.Parameter(torch.ones(hidden_size))
        # Module tương tác (Interaction module)
        self.linear_interact = nn.Linear(2 * hidden_size, hidden_size)
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        batch_size, seq_len, _ = x.size()
        device = x.device
        h = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        hidden_states = []  # lưu hidden state của layer cuối từng bước
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_size]
            # Tính Δt từ cột thời gian (giả sử self.time_idx là chỉ số của cột thời gian)
            if t == 0:
                delta_t = torch.zeros(batch_size, 1, device=device)
            else:
                time_t = x[:, t, self.time_idx].unsqueeze(1)  # [batch, 1]
                time_prev = x[:, t-1, self.time_idx].unsqueeze(1)
                delta_t = time_t - time_prev
            # Lặp qua các layer
            for layer in range(self.num_layers):
                inp = x_t if layer == 0 else h[layer - 1]
                h[layer], c[layer] = self.cells[layer](inp, delta_t, h[layer], c[layer])
            hidden_states.append(h[-1].unsqueeze(1))
        H = torch.cat(hidden_states, dim=1)  # [batch, seq_len, hidden_size]
        # Tính attention cho bước cuối cùng (sử dụng attention_memory bước)
        if seq_len < self.attention_memory:
            mem_indices = list(range(seq_len))
        else:
            mem_indices = list(range(seq_len - self.attention_memory, seq_len))
        q_t = torch.cat([h[-1], c[-1]], dim=1)  # [batch, 2*hidden_size]
        q_t_proj = self.linear_aq(q_t).unsqueeze(1)  # [batch, 1, hidden_size]
        H_mem = H[:, mem_indices, :]  # [batch, m, hidden_size]
        H_proj = self.linear_ah(H_mem)  # [batch, m, hidden_size]
        scores = torch.tanh(q_t_proj + H_proj + self.bias_a)  # [batch, m, hidden_size]
        scores = torch.matmul(scores, self.v)  # [batch, m]
        alpha = torch.softmax(scores, dim=1).unsqueeze(2)  # [batch, m, 1]
        e_t = torch.sum(alpha * H_mem, dim=1)  # [batch, hidden_size]
        final_repr = torch.tanh(self.linear_interact(torch.cat([h[-1], e_t], dim=1)))  # [batch, hidden_size]
        return final_repr

class THLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_memory=10, time_idx=0):
        super(THLSTMClassifier, self).__init__()
        self.th_lstm = THLSTM(input_size, hidden_size, num_layers, attention_memory, time_idx)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        repr = self.th_lstm(x)  # [batch, hidden_size]
        out = self.fc(repr)
        return self.sigmoid(out)

# 3. Dataset và DataLoader
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # [num_sequences, seq_len, num_features]
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
batch_size = 64
train_dataset = FraudDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                               torch.tensor(y_train_seq, dtype=torch.float32))
test_dataset = FraudDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                              torch.tensor(y_test_seq, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. Evaluation và Training (checkpoint, early stopping)
def evaluate_model(loader, model, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze().cpu().numpy()
            all_preds.extend(outputs)
            all_targets.extend(y_batch.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    thresholds = [0.1 * i for i in range(1, 10)]
    best_f1 = 0
    best_threshold = 0.5
    for t in thresholds:
        binary_preds = (all_preds > t).astype(int)
        f1 = f1_score(all_targets, binary_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    auc = roc_auc_score(all_targets, all_preds)
    combined_metric = (best_f1 + auc) / 2
    binary_preds = (all_preds > best_threshold).astype(int)
    cm = confusion_matrix(all_targets, binary_preds)
    TP = cm[1,1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    FP = cm[0,1] if cm.shape[1] > 1 else 0
    FN = cm[1,0] if cm.shape[0] > 1 else 0
    TN = cm[0,0]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return best_threshold, best_f1, auc, combined_metric, accuracy, precision, recall

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
        train_threshold, train_f1, train_auc, train_combined, train_acc, train_prec, train_rec = evaluate_model(train_loader, model, device)
        print(f"Train Metrics - Threshold: {train_threshold:.2f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, Combined: {train_combined:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}")
        test_threshold, test_f1, test_auc, test_combined, test_acc, test_prec, test_rec = evaluate_model(test_loader, model, device)
        print(f"Test Metrics  - Threshold: {test_threshold:.2f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Combined: {test_combined:.4f}, Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}")
        if test_combined > best_combined_metric_test:
            best_combined_metric_test = test_combined
            checkpoint = {
                'epoch': epoch+1,
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
        if average_loss < best_loss:
            best_loss = average_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 8:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

# 5. Khởi tạo và Training mô hình
input_size = X_train_seq.shape[2]    # số feature (bao gồm cả cột thời gian)
hidden_size = 64
num_layers = 2
time_idx = 0            # Giả sử cột thời gian nằm ở vị trí 0 trong vector feature
attention_memory = 10   # Số bước lịch sử để tính attention

model = THLSTMClassifier(input_size, hidden_size, num_layers, attention_memory, time_idx)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 30
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

# 6. (Optional) Load Checkpoint tốt nhất và đánh giá cuối cùng
checkpoint = torch.load('best_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print("\nLoaded Best Checkpoint:")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Train F1: {checkpoint['train_f1']:.4f}, Train AUC: {checkpoint['train_auc']:.4f}, Train Combined: {checkpoint['train_combined']:.4f}")
print(f"Test F1: {checkpoint['test_f1']:.4f}, Test AUC: {checkpoint['test_auc']:.4f}, Test Combined: {checkpoint['test_combined']:.4f}")