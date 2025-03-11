# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

# %%

df = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv')

# Xử lý thời gian
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 else ('10-20' if x >= 10 and x < 20 else ('20-30' if x >= 20 and x < 30 else ('30-40' if x >= 30 and x < 40 else ('40-50' if x >= 40 and x < 50 else ('50-60' if x >= 50 and x < 60 else ('60-70' if x >= 60 and x < 70 else ('70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

# =============================================================================
# Drop các cột không cần thiết, tuy nhiên không drop 'trans_date_trans_time' và 'cc_num'
# =============================================================================
drop_col = ['Unnamed: 0', 'merchant', 'first', 'last', 'street', 'city', 'state', 'lat',
            'long','dob', 'unix_time', 'cust_age', 'merch_lat', 'merch_long', 'city_pop']
df.drop(drop_col, axis=1, inplace=True)

# Pivot table cho cust_age_groups
age_piv_2 = pd.pivot_table(data=df,
                           index='cust_age_groups',
                           columns='is_fraud',
                           values='amt',
                           aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Pivot table cho category
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# Pivot table cho job
job_txn_piv_2 = pd.pivot_table(data=df,
                               index='job',
                               columns='is_fraud',
                               values='amt',
                               aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

df['trans_hour'] = df['trans_hour'].astype('int')
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# =============================================================================
# Chuyển đổi trans_date_trans_time thành timestamp (dạng số) để có thể scale
# =============================================================================
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

# =============================================================================
# 2️⃣ Train_test_split
# =============================================================================

train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train.shape)
print("Test shape:", test.shape)
# Lưu dữ liệu train
train.to_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_fraudTrain.csv', index=False)

# Lưu dữ liệu test
test.to_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_fraudTest.csv', index=False)

# Drop cột trans_num từ cả train và test
train.drop('trans_num', axis=1, inplace=True)
test.drop('trans_num', axis=1, inplace=True)

# Tách features và label
y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)

y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# =============================================================================
# 3️⃣ Scaling dữ liệu
# =============================================================================
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Convert lại thành DataFrame
X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

# %%
sequence_length = 1000  # Số giao dịch cần trong 1 sequence

def create_sequences_predict_all(df, sequence_length):
    sequences, labels = [], []
    # Nhóm theo cc_num
    grouped = df.groupby('cc_num')
    for user_id, group in grouped:
        # Sắp xếp theo trans_date_trans_time (đã chuyển thành timestamp)
        group = group.sort_values(by='trans_date_trans_time')
        # Lấy các giá trị: loại bỏ 'is_fraud' và 'cc_num'
        values = group.drop(columns=['is_fraud', 'cc_num']).values
        targets = group['is_fraud'].values
        n = len(group)
        for i in range(n):
            if i < sequence_length:
                # Nếu số giao dịch hiện có nhỏ hơn sequence_length
                pad_needed = sequence_length - (i + 1)
                # Replicate giao dịch đầu tiên cho đủ số lượng pad
                pad = np.repeat(values[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, values[:i+1]), axis=0)
            else:
                # Nếu đủ giao dịch, lấy sequence gồm các giao dịch từ (i-sequence_length+1) đến i
                seq = values[i-sequence_length+1:i+1]
            sequences.append(seq)
            labels.append(targets[i])
    return np.array(sequences), np.array(labels)

# Gộp thêm cột 'is_fraud' vào DataFrame scale để tạo sequence
train_seq_df = X_train_sc.copy()
train_seq_df['is_fraud'] = y_train.values

test_seq_df = X_test_sc.copy()
test_seq_df['is_fraud'] = y_test.values

X_train_seq, y_train_seq = create_sequences_predict_all(train_seq_df, sequence_length)
X_test_seq, y_test_seq = create_sequences_predict_all(test_seq_df, sequence_length)

print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

# %%
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # shape: (num_sequences, sequence_length, num_features)
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

# Shuffle chỉ xáo trộn thứ tự các sequence, không làm xáo trộn bên trong mỗi sequence
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
# 6. Xây dựng mô hình GRU
# =============================================================================
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FraudGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.gru(x)
        # Lấy trạng thái ẩn của bước thờyi gian cuối cùng
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# %%
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {average_loss:.4f}')

input_size = X_train_seq.shape[2]  # số feature (sau khi loại bỏ cc_num)
hidden_size = 64
num_layers = 2
model = FraudGRU(input_size, hidden_size, num_layers)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# =============================================================================
# 8️⃣ Dự đoán và đánh giá mô hình
# =============================================================================
model.eval()
y_pred_train_proba = []
with torch.no_grad():
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch).squeeze().cpu().numpy()
        y_pred_train_proba.extend(outputs)
y_pred_train_proba = np.array(y_pred_train_proba)

# --- 2. Lấy predicted probabilities cho tập test ---
y_pred_test_proba = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch).squeeze().cpu().numpy()
        y_pred_test_proba.extend(outputs)
y_pred_test_proba = np.array(y_pred_test_proba)

# --- 3. Tạo DataFrame kết quả cho train và test ---
# Lưu ý: ở đây chúng ta dùng y_train_seq và y_test_seq (là nhãn của sequence)
y_train_results = pd.DataFrame(y_pred_train_proba, columns=['pred_fraud'])
y_train_results['pred_not_fraud'] = 1 - y_train_results['pred_fraud']
y_train_results['y_train_actual'] = y_train_seq  # y_train_seq là nhãn thực của các sequence

y_test_results = pd.DataFrame(y_pred_test_proba, columns=['pred_fraud'])
y_test_results['pred_not_fraud'] = 1 - y_test_results['pred_fraud']
y_test_results['y_test_actual'] = y_test_seq

# --- 4. Đánh giá với các ngưỡng khác nhau ---
numbers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i in numbers:
    y_train_results[i] = y_train_results['pred_fraud'].map(lambda x: 1 if x > i else 0)
    y_test_results[i] = y_test_results['pred_fraud'].map(lambda x: 1 if x > i else 0)

cutoff_df = pd.DataFrame(columns=['Threshold', 'Accuracy', 'precision_score', 'recall_score', 'F1_score'])

# %%
for i in numbers:
    cm1 = confusion_matrix(y_train_results['y_train_actual'], y_train_results[i])
    TP, FP, FN, TN = cm1[1,1], cm1[0,1], cm1[1,0], cm1[0,0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    cutoff_df.loc[i] = [i, accuracy, precision, recall, f1_score_value]

print("Train Evaluation:")
print(cutoff_df)

best_idx = cutoff_df['F1_score'].idxmax()
best_threshold = cutoff_df.loc[best_idx, 'Threshold']
best_accuracy = cutoff_df.loc[best_idx, 'Accuracy']
best_precision = cutoff_df.loc[best_idx, 'precision_score']
best_recall = cutoff_df.loc[best_idx, 'recall_score']
best_f1_score = cutoff_df.loc[best_idx, 'F1_score']
best_auc = roc_auc_score(y_train_results['y_train_actual'], y_train_results['pred_fraud'])

print(f'Best Threshold (Train): {best_threshold:.4f}')
print(f'Best Accuracy (Train): {best_accuracy:.4f}')
print(f'Best Precision (Train): {best_precision:.4f}')
print(f'Best Recall (Train): {best_recall:.4f}')
print(f'Best F1 Score (Train): {best_f1_score:.4f}')
print(f'Best ROC_AUC Score (Train): {best_auc:.4f}')

# %%
cutoff_df_test = pd.DataFrame(columns=['Threshold', 'Accuracy', 'precision_score', 'recall_score', 'F1_score'])
for i in numbers:
    cm1 = confusion_matrix(y_test_results['y_test_actual'], y_test_results[i])
    TP, FP, FN, TN = cm1[1,1], cm1[0,1], cm1[1,0], cm1[0,0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    cutoff_df_test.loc[i] = [i, accuracy, precision, recall, f1_score_value]

print("Test Evaluation:")
print(cutoff_df_test)

best_idx_test = cutoff_df_test['F1_score'].idxmax()
best_threshold_test = cutoff_df_test.loc[best_idx_test, 'Threshold']
best_accuracy_test = cutoff_df_test.loc[best_idx_test, 'Accuracy']
best_precision_test = cutoff_df_test.loc[best_idx_test, 'precision_score']
best_recall_test = cutoff_df_test.loc[best_idx_test, 'recall_score']
best_f1_score_test = cutoff_df_test.loc[best_idx_test, 'F1_score']
best_auc_test = roc_auc_score(y_test_results['y_test_actual'], y_test_results['pred_fraud'])

print(f'Best Threshold (Test): {best_threshold_test:.4f}')
print(f'Best Accuracy (Test): {best_accuracy_test:.4f}')
print(f'Best Precision (Test): {best_precision_test:.4f}')
print(f'Best Recall (Test): {best_recall_test:.4f}')
print(f'Best F1 Score (Test): {best_f1_score_test:.4f}')
print(f'Best ROC_AUC Score (Test): {best_auc_test:.4f}')


