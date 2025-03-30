import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# %% Data Loading and Preprocessing
df = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv')

# Process time
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_date_trans_time_numeric'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 else 
                                              ('10-20' if x >= 10 and x < 20 else 
                                               ('20-30' if x >= 20 and x < 30 else 
                                                ('30-40' if x >= 30 and x < 40 else 
                                                 ('40-50' if x >= 40 and x < 50 else 
                                                  ('50-60' if x >= 50 and x < 60 else 
                                                   ('60-70' if x >= 60 and x < 70 else 
                                                    ('70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

# Pivot table for cust_age_groups
age_piv_2 = pd.pivot_table(data=df,
                           index='cust_age_groups',
                           columns='is_fraud',
                           values='amt',
                           aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Process category
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# Process job
job_txn_piv_2 = pd.pivot_table(data=df,
                               index='job',
                               columns='is_fraud',
                               values='amt',
                               aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

# Factorize categorical features
df['merchant_num'] = pd.factorize(df['merchant'])[0]
df['last_num'] = pd.factorize(df['last'])[0]
df['street_num'] = pd.factorize(df['street'])[0]
df['city_num'] = pd.factorize(df['city'])[0]
df['zip_num'] = pd.factorize(df['zip'])[0]
df['state_num'] = pd.factorize(df['state'])[0]

df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'merchant', 'first', 'last', 'street', 'city', 'state', 'lat', 'long', 'dob',
             'unix_time', 'merch_lat', 'merch_long', 'city_pop']
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# %% Train-test Split
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Drop column trans_num if present
if 'trans_num' in train.columns:
    train.drop('trans_num', axis=1, inplace=True)
    test.drop('trans_num', axis=1, inplace=True)

# Separate features and labels
y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)

y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# %% Scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Convert back to DataFrame
X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

# %% Sequence Creation using Transactional Expansion
def create_sequences_transactional_expansion(df, memory_size):
    sequences, labels = [], []
    
    # Group by 'cc_num' (credit card number)
    grouped = df.groupby('cc_num')
    
    for user_id, group in grouped:
        # Sort by time (ensure trans_date_trans_time_numeric is used)
        group = group.sort_values(by='trans_date_trans_time_numeric')
        values = group.drop(columns=['is_fraud', 'cc_num']).values
        targets = group['is_fraud'].values
        n = len(group)
        
        # Create sequence for each transaction
        for i in range(n):
            if i < memory_size:
                pad_needed = memory_size - (i + 1)
                pad = np.repeat(values[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, values[:i+1]), axis=0)
            else:
                seq = values[i-memory_size+1:i+1]
            sequences.append(seq)
            labels.append(targets[i])
    
    return np.array(sequences), np.array(labels)

memory_size = 400  # Define the sequence length
train_seq_df = X_train_sc.copy()
train_seq_df['is_fraud'] = y_train.values

test_seq_df = X_test_sc.copy()
test_seq_df['is_fraud'] = y_test.values

X_train_seq, y_train_seq = create_sequences_transactional_expansion(train_seq_df, memory_size)
X_test_seq, y_test_seq = create_sequences_transactional_expansion(test_seq_df, memory_size)

print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

# %% Dataset and DataLoader
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Define the TH‑LSTM Model Components
class THLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(THLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        # Compute time state s_t from previous hidden state, current input, and Δt (reshaped to 1D)
        self.linear_s = nn.Linear(hidden_size + input_size + 1, hidden_size)
        # Gates: forget (f_t), input (i_t), time-aware (T_t), candidate (ζ_t) and output (o_t)
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
        self.attention_memory = attention_memory  # number of historical steps for attention
        self.time_idx = time_idx  # index of the time column in the input vector
        self.cells = nn.ModuleList([THLSTMCell(input_size if i==0 else hidden_size, hidden_size) 
                                    for i in range(num_layers)])
        # Attention module: process q_t and historical hidden states
        self.linear_aq = nn.Linear(2 * hidden_size, hidden_size)
        self.linear_ah = nn.Linear(hidden_size, hidden_size)
        self.bias_a = nn.Parameter(torch.zeros(hidden_size))
        self.v = nn.Parameter(torch.ones(hidden_size))
        # Interaction module
        self.linear_interact = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        batch_size, seq_len, _ = x.size()
        device = x.device
        h = [torch.zeros(batch_size, self.hidden_size, device=device) 
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=device) 
             for _ in range(self.num_layers)]
        hidden_states = []  # store the last layer hidden state at each time step
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_size]
            # Compute Δt using the time column (assumes self.time_idx is the time column index)
            if t == 0:
                delta_t = torch.zeros(batch_size, 1, device=device)
            else:
                time_t = x[:, t, self.time_idx].unsqueeze(1)  # [batch, 1]
                time_prev = x[:, t-1, self.time_idx].unsqueeze(1)
                delta_t = time_t - time_prev
            # Pass through each layer
            for layer in range(self.num_layers):
                inp = x_t if layer == 0 else h[layer - 1]
                h[layer], c[layer] = self.cells[layer](inp, delta_t, h[layer], c[layer])
            hidden_states.append(h[-1].unsqueeze(1))
        H = torch.cat(hidden_states, dim=1)  # [batch, seq_len, hidden_size]
        # Apply attention on the last attention_memory steps
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

# %% Evaluation Function
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
    
    # Compute ROC AUC score
    auc = roc_auc_score(all_targets, all_preds)
    thresholds = [0.1 * i for i in range(1, 10)]
    best_f1 = 0
    best_threshold = 0.5
    for t in thresholds:
        binary_preds = (all_preds > t).astype(int)
        f1 = f1_score(all_targets, binary_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    combined_metric = (best_f1 + auc) / 2
    
    # Compute additional metrics with the best threshold
    binary_preds = (all_preds > best_threshold).astype(int)
    cm = confusion_matrix(all_targets, binary_preds)
    TP = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    FP = cm[0, 1] if cm.shape[1] > 1 else 0
    FN = cm[1, 0] if cm.shape[0] > 1 else 0
    TN = cm[0, 0]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return best_threshold, best_f1, auc, combined_metric, accuracy, precision, recall

# %% Training Function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    best_combined_metric_test = -float('inf')
    epochs_without_improvement = 0

    best_epoch = None
    best_train_metrics = None
    best_test_metrics = None

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
        
        # Evaluate on training set
        train_threshold, train_f1, train_auc, train_combined, train_acc, train_prec, train_rec = evaluate_model(train_loader, model, device)
        print(f"Train Metrics - Best Threshold: {train_threshold:.2f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, Combined: {train_combined:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        
        # Evaluate on test set
        test_threshold, test_f1, test_auc, test_combined, test_acc, test_prec, test_rec = evaluate_model(test_loader, model, device)
        print(f"Test Metrics  - Best Threshold: {test_threshold:.2f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Combined: {test_combined:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
        
        if test_combined > best_combined_metric_test:
            best_combined_metric_test = test_combined
            best_epoch = epoch + 1
            best_train_metrics = (train_f1, train_auc, train_combined)
            best_test_metrics = (test_f1, test_auc, test_combined)
            print(f'*** Best metrics updated at epoch {epoch+1} ***')
        
        if average_loss < best_loss:
            best_loss = average_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 8:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

    print("\n========== Final Best Results ==========")
    print(f"Best Epoch: {best_epoch}")
    print(f"Train Metrics - F1: {best_train_metrics[0]:.4f}, AUC: {best_train_metrics[1]:.4f}, Combined: {best_train_metrics[2]:.4f}")
    print(f"Test Metrics  - F1: {best_test_metrics[0]:.4f}, AUC: {best_test_metrics[1]:.4f}, Combined: {best_test_metrics[2]:.4f}")

# %% Model Initialization and Training
input_size = X_train_seq.shape[2]  # number of features
hidden_size = 64
num_layers = 2
# Use attention_memory (e.g., 10) and time_idx=0 (assuming the first feature is the timestamp)
attention_memory = 10
time_idx = 0

model = THLSTMClassifier(input_size, hidden_size, num_layers, attention_memory, time_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs for training.")
    model = nn.DataParallel(model)

model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)