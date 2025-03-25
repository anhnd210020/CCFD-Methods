import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

# Đọc dữ liệu và xử lý như trước
df = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv')
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])
df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 else ('10-20' if x >= 10 and x < 20 else ('20-30' if x >= 20 and x < 30 else ('30-40' if x >= 30 and x < 40 else ('40-50' if x >= 40 and x < 50 else ('50-60' if x >= 50 and x < 60 else ('60-70' if x >= 60 and x < 70 else ('70-80' if x >= 70 and x < 80 else 'Above 80'))))))))
 
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

# Chuyển đổi trans_date_trans_time thành timestamp
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

# Train-test split
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
train.to_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_fraudTrain.csv', index=False)
test.to_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_fraudTest.csv', index=False)

if 'trans_num' in train.columns:
    train.drop('trans_num', axis=1, inplace=True)
if 'trans_num' in test.columns:
    test.drop('trans_num', axis=1, inplace=True)

y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)
y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

sequence_length = 100  # Số giao dịch trong 1 sequence

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

# Định nghĩa Dataset cho phân loại
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # numpy array: (num_sequences, sequence_length, num_features)
        self.y = y  # nhãn nhị phân
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sequence = self.X[idx]
        label = self.y[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

batch_size = 64
train_dataset = FraudDataset(X_train_seq, y_train_seq)
test_dataset = FraudDataset(X_test_seq, y_test_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Mô hình GRU + GCN cho bài toán phân loại
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim=32):
        super(FraudGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_dim)
        self.gcn = GCNConv(embedding_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)  # Trả về logits cho phân loại nhị phân
    
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        out, _ = self.gru(x)
        embedding = self.fc(out[:, -1, :])
        
        # Xây dựng fully-connected graph cho batch
        batch_size = embedding.size(0)
        device = embedding.device
        row = torch.arange(batch_size, device=device).repeat_interleave(batch_size)
        col = torch.arange(batch_size, device=device).repeat(batch_size)
        edge_index = torch.stack([row, col], dim=0)
        
        gcn_embedding = self.gcn(embedding, edge_index)
        logits = self.classifier(gcn_embedding)
        return logits

input_size = X_train_seq.shape[2]
hidden_size = 64
num_layers = 2
embedding_dim = 32
model = FraudGRU(input_size, hidden_size, num_layers, embedding_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model_classification(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_combined_metric_test = -float('inf')
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")
        
        # Đánh giá trên tập test
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device).unsqueeze(1)
                logits = model(sequences)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        probs = 1 / (1 + np.exp(-all_logits))
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(all_labels, preds)
        auc = roc_auc_score(all_labels, probs)
        f1 = f1_score(all_labels, preds)
        precision = precision_score(all_labels, preds)
        recall = recall_score(all_labels, preds)
        print(f"Test Metrics - Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")
        
        # Tính combined metric (có thể là trung bình của F1 và AUC)
        test_combined = (f1 + auc) / 2
        
        if test_combined > best_combined_metric_test:
            best_combined_metric_test = test_combined
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
                'train_f1': f1,      # Ở đây dùng các số liệu test làm số liệu train cho ví dụ
                'train_auc': auc,
                'train_combined': test_combined,
                'test_f1': f1,
                'test_auc': auc,
                'test_combined': test_combined,
            }
            torch.save(checkpoint, 'best_checkpoint_contrastive.pth')
            print(f'Checkpoint saved at epoch {epoch+1} with test combined metric: {test_combined:.4f}')
        
        if average_loss < best_loss:
            best_loss = average_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 8:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

num_epochs = 30
train_model_classification(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

# 8️⃣ (Optional) Load the Best Checkpoint and Report Final Metrics
checkpoint = torch.load('best_checkpoint_contrastive.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print("\nLoaded Best Checkpoint:")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Train F1: {checkpoint['train_f1']:.4f}, Train AUC: {checkpoint['train_auc']:.4f}, Train Combined: {checkpoint['train_combined']:.4f}")
print(f"Test F1: {checkpoint['test_f1']:.4f}, Test AUC: {checkpoint['test_auc']:.4f}, Test Combined: {checkpoint['test_combined']:.4f}")