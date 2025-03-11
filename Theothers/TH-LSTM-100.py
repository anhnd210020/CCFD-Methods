# %% Import các thư viện cần thiết
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# %% Đọc và xử lý dữ liệu
df_train = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv')
df_test = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTest.csv')
df = pd.concat([df_train, df_test])

# Xử lý thời gian
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 else ('10-20' if x >= 10 and x < 20 else ('20-30' if x >= 20 and x < 30 else ('30-40' if x >= 30 and x < 40 else ('40-50' if x >= 40 and x < 50 else ('50-60' if x >= 50 and x < 60 else ('60-70' if x >= 60 and x < 70 else ('70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

drop_col = ['Unnamed: 0', 'merchant', 'first', 'last', 'street', 'city', 'state', 'lat',
            'long','dob', 'unix_time', 'cust_age', 'merch_lat', 'merch_long', 'city_pop']
df.drop(drop_col, axis=1, inplace=True)

age_piv_2 = pd.pivot_table(data=df,
                           index='cust_age_groups',
                           columns='is_fraud',
                           values='amt',
                           aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

job_txn_piv_2 = pd.pivot_table(data=df,
                               index='job',
                               columns='is_fraud',
                               values='amt',
                               aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

df['trans_hour'] = df['trans_hour'].astype('int')
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# Chuyển đổi trans_date_trans_time thành timestamp (dạng số)
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

# %% Train-test split
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Drop cột trans_num
train.drop('trans_num', axis=1, inplace=True)
test.drop('trans_num', axis=1, inplace=True)

y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)
y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# %% Scaling dữ liệu
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

# %% Tạo các sequence
sequence_length = 100  # số giao dịch trong 1 sequence

def create_sequences_predict_all(df, sequence_length):
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

X_train_seq, y_train_seq = create_sequences_predict_all(train_seq_df, sequence_length)
X_test_seq, y_test_seq = create_sequences_predict_all(test_seq_df, sequence_length)

print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

# %% Định nghĩa mô hình TH-LSTM với sigmoid (1 output)
class THLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(THLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_sh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_sx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_st = nn.Linear(1, hidden_dim, bias=False)
        self.b_s  = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_fh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_fx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_fs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_f  = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_ih = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ix = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_is = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_i  = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_Th = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_Tx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_Ts = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_T  = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_uh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ux = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_us = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_u  = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_oh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ox = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_os = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_o  = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_t, delta_t, h_prev, c_prev):
        s_t = torch.tanh(self.W_sh(h_prev) + self.W_sx(x_t) + self.W_st(delta_t) + self.b_s)
        f_t = torch.sigmoid(self.W_fh(h_prev) + self.W_fx(x_t) + self.W_fs(s_t) + self.b_f)
        i_t = torch.sigmoid(self.W_ih(h_prev) + self.W_ix(x_t) + self.W_is(s_t) + self.b_i)
        T_t = torch.sigmoid(self.W_Th(h_prev) + self.W_Tx(x_t) + self.W_Ts(s_t) + self.b_T)
        zeta = torch.tanh(self.W_uh(h_prev) + self.W_ux(x_t) + self.W_us(s_t) + self.b_u)
        c_t = f_t * c_prev + i_t * zeta + T_t * s_t
        o_t = torch.sigmoid(self.W_oh(h_prev) + self.W_ox(x_t) + self.W_os(s_t) + self.b_o)
        h_tilde = o_t * torch.tanh(c_t)
        return h_tilde, c_t

class CurrentHistoricalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CurrentHistoricalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.Waq = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.Wah = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ba  = nn.Parameter(torch.zeros(hidden_dim))
        self.v_t = nn.Parameter(torch.randn(hidden_dim))
    def forward(self, h_tilde, c_t, h_history):
        q_t = torch.cat([h_tilde, c_t], dim=1)  # (B, 2H)
        h_hist_stack = torch.stack(h_history, dim=1)  # (B, d, H)
        q_t_proj = self.Waq(q_t)  # (B, H)
        q_t_expanded = q_t_proj.unsqueeze(1)  # (B,1,H)
        h_hist_proj = self.Wah(h_hist_stack)    # (B,d,H)
        score_pre = torch.tanh(q_t_expanded + h_hist_proj + self.ba)  # (B,d,H)
        alpha_unscaled = torch.einsum('bdh,h->bd', score_pre, self.v_t)
        alpha = F.softmax(alpha_unscaled, dim=1)  # (B,d)
        alpha_expanded = alpha.unsqueeze(-1)
        e_t = (alpha_expanded * h_hist_stack).sum(dim=1)  # (B,H)
        return e_t, alpha

class InteractionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InteractionModule, self).__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_g = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
    def forward(self, h_tilde, e_t, g_t):
        out = self.W_h(h_tilde) + self.W_e(e_t) + self.W_g(g_t) + self.b_h
        h_t = torch.tanh(out)
        return h_t

# Sửa TH-LSTMModel: chuyển classifier thành 1 output (sigmoid dựa trên BCEWithLogitsLoss)
class THLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size):
        super(THLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.thlstm_cell = THLSTMCell(input_dim, hidden_dim)
        self.attention_module = CurrentHistoricalAttention(hidden_dim)
        self.interaction_module = InteractionModule(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)  # 1 output for binary classification
        
    def forward(self, x_seq, delta_seq, g_seq):
        # x_seq: (B, seq_len, input_dim)
        # delta_seq: (B, seq_len, 1)
        # g_seq: (B, seq_len, input_dim)
        batch_size, seq_len, _ = x_seq.shape
        h_prev = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        c_prev = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        all_h = []
        outputs = []
        for t in range(seq_len):
            x_t     = x_seq[:, t, :]
            delta_t = delta_seq[:, t, :]
            g_t     = g_seq[:, t, :]
            h_tilde, c_t = self.thlstm_cell(x_t, delta_t, h_prev, c_prev)
            start_idx = max(0, t - self.memory_size)
            h_history = all_h[start_idx:t]
            if len(h_history) == 0:
                e_t = torch.zeros_like(h_tilde)
            else:
                e_t, _ = self.attention_module(h_tilde, c_t, h_history)
            h_t = self.interaction_module(h_tilde, e_t, g_t)
            all_h.append(h_t)
            h_prev, c_prev = h_t, c_t
            logits = self.classifier(h_t)  # (B, 1)
            outputs.append(logits.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (B, seq_len, 1)
        return outputs

# %% Tạo DataLoader
batch_size = 64
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
delta_train_tensor = torch.abs(torch.randn(X_train_tensor.size(0), sequence_length, 1))
delta_test_tensor = torch.abs(torch.randn(X_test_tensor.size(0), sequence_length, 1))
g_train_tensor = X_train_tensor.clone()
g_test_tensor = X_test_tensor.clone()
# Chuyển y sang float và kích thước (N,1)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(-1)

train_dataset = TensorDataset(X_train_tensor, delta_train_tensor, g_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, delta_test_tensor, g_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Training loop sử dụng BCEWithLogitsLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = THLSTMModel(input_dim=X_train_tensor.size(2), hidden_dim=64, memory_size=3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, delta_batch, g_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            delta_batch = delta_batch.to(device)
            g_batch = g_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, delta_batch, g_batch)  # (B, seq_len, 1)
            final_logits = outputs[:, -1, :]  # Lấy giao dịch cuối trong sequence (B, 1)
            loss = criterion(final_logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

num_epochs = 20
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# %% Dự đoán và đánh giá mô hình
model.eval()
y_pred_train_proba = []
with torch.no_grad():
    for X_batch, delta_batch, g_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        delta_batch = delta_batch.to(device)
        g_batch = g_batch.to(device)
        outputs = model(X_batch, delta_batch, g_batch)  # (B, seq_len, 1)
        final_logits = outputs[:, -1, :]  # (B, 1)
        # BCEWithLogitsLoss trả về logits, áp sigmoid để tính xác suất
        prob = torch.sigmoid(final_logits)
        y_pred_train_proba.extend(prob.cpu().numpy())
y_pred_train_proba = np.array(y_pred_train_proba)

y_pred_test_proba = []
with torch.no_grad():
    for X_batch, delta_batch, g_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        delta_batch = delta_batch.to(device)
        g_batch = g_batch.to(device)
        outputs = model(X_batch, delta_batch, g_batch)
        final_logits = outputs[:, -1, :]
        prob = torch.sigmoid(final_logits)
        y_pred_test_proba.extend(prob.cpu().numpy())
y_pred_test_proba = np.array(y_pred_test_proba)

# Tạo DataFrame kết quả dự đoán
y_train_results = pd.DataFrame(y_pred_train_proba, columns=['pred_fraud'])
y_train_results['pred_not_fraud'] = 1 - y_train_results['pred_fraud']
y_train_results['y_train_actual'] = y_train_seq

y_test_results = pd.DataFrame(y_pred_test_proba, columns=['pred_fraud'])
y_test_results['pred_not_fraud'] = 1 - y_test_results['pred_fraud']
y_test_results['y_test_actual'] = y_test_seq

# Đánh giá với các ngưỡng khác nhau
numbers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cutoff_df = pd.DataFrame(columns=['Threshold', 'Accuracy', 'Precision_score', 'Recall_score', 'F1_score'])
for thr in numbers:
    # Với mỗi ngưỡng, chuyển xác suất thành nhãn (1 nếu > ngưỡng, 0 nếu không)
    y_train_results[thr] = y_train_results['pred_fraud'].map(lambda x: 1 if x > thr else 0)
    cm = confusion_matrix(y_train_results['y_train_actual'], y_train_results[thr])
    TP, FP, FN, TN = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
    precision = TP / (TP+FP) if (TP+FP) > 0 else 0
    recall = TP / (TP+FN) if (TP+FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP+TN) / (TP+FP+FN+TN)
    cutoff_df.loc[thr] = [thr, accuracy, precision, recall, f1]

print("Train Evaluation:")
print(cutoff_df)

best_idx = cutoff_df['F1_score'].idxmax()
best_thr = cutoff_df.loc[best_idx, 'Threshold']
best_accuracy = cutoff_df.loc[best_idx, 'Accuracy']
best_precision = cutoff_df.loc[best_idx, 'Precision_score']
best_recall = cutoff_df.loc[best_idx, 'Recall_score']
best_f1_score = cutoff_df.loc[best_idx, 'F1_score']
best_auc = roc_auc_score(y_train_results['y_train_actual'], y_train_results['pred_fraud'])

print(f'Best Threshold (Train): {best_thr:.4f}')
print(f'Best Accuracy (Train): {best_accuracy:.4f}')
print(f'Best Precision (Train): {best_precision:.4f}')
print(f'Best Recall (Train): {best_recall:.4f}')
print(f'Best F1 Score (Train): {best_f1_score:.4f}')
print(f'Best ROC_AUC Score (Train): {best_auc:.4f}')

cutoff_df_test = pd.DataFrame(columns=['Threshold', 'Accuracy', 'Precision_score', 'Recall_score', 'F1_score'])
for thr in numbers:
    y_test_results[thr] = y_test_results['pred_fraud'].map(lambda x: 1 if x > thr else 0)
    cm = confusion_matrix(y_test_results['y_test_actual'], y_test_results[thr])
    TP, FP, FN, TN = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
    precision = TP / (TP+FP) if (TP+FP) > 0 else 0
    recall = TP / (TP+FN) if (TP+FN) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    cutoff_df_test.loc[thr] = [thr, accuracy, precision, recall, f1]

print("Test Evaluation:")
print(cutoff_df_test)

best_idx_test = cutoff_df_test['F1_score'].idxmax()
best_thr_test = cutoff_df_test.loc[best_idx_test, 'Threshold']
best_precision_test = cutoff_df_test.loc[best_idx_test, 'Precision_score']
best_recall_test = cutoff_df_test.loc[best_idx_test, 'Recall_score']
best_f1_score_test = cutoff_df_test.loc[best_idx_test, 'F1_score']
best_accuracy_test = cutoff_df_test.loc[best_idx_test, 'Accuracy']
best_auc_test = roc_auc_score(y_test_results['y_test_actual'], y_test_results['pred_fraud'])

print(f'Best Threshold (Test): {best_thr_test:.4f}')
print(f'Best Accuracy (Test): {best_accuracy_test:.4f}')
print(f'Best Precision (Test): {best_precision_test:.4f}')
print(f'Best Recall (Test): {best_recall_test:.4f}')
print(f'Best F1 Score (Test): {best_f1_score_test:.4f}')
print(f'Best ROC_AUC Score (Test): {best_auc_test:.4f}')