import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

##############################################
# 1. DATA LOADING & PREPROCESSING (Unified)  #
##############################################

# Read train and test CSV files and concatenate them
df_train = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv')
df_test  = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTest.csv')
df = pd.concat([df_train, df_test])

# Process datetime and extract hour
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

# Process DOB, compute customer age and age groups
df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 
                                              else ('10-20' if x < 20 
                                                    else ('20-30' if x < 30 
                                                          else ('30-40' if x < 40 
                                                                else ('40-50' if x < 50 
                                                                      else ('50-60' if x < 60 
                                                                            else ('60-70' if x < 70 
                                                                                  else ('70-80' if x < 80 else 'Above 80'))))))))

# Drop unneeded columns (keeping 'cc_num' and 'trans_date_trans_time' for grouping/ordering)
drop_col = ['Unnamed: 0', 'merchant', 'first', 'last', 'street', 'city', 'state', 
            'lat', 'long', 'dob', 'unix_time', 'cust_age', 'city_pop']
df.drop(drop_col, axis=1, inplace=True)

# Process 'cust_age_groups' using pivot table mapping
age_piv = pd.pivot_table(data=df,
                         index='cust_age_groups',
                         columns='is_fraud',
                         values='amt',
                         aggfunc=np.mean)
age_piv.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv.index.values, age_piv.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Process 'category'
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# Process 'job'
job_piv = pd.pivot_table(data=df,
                         index='job',
                         columns='is_fraud',
                         values='amt',
                         aggfunc=np.mean)
job_dic = {k: v for (k, v) in zip(job_piv.index.values, job_piv.reset_index().index.values)}
df['job'] = df['job'].map(job_dic)

# Convert 'trans_hour' to int and one-hot encode 'gender'
df['trans_hour'] = df['trans_hour'].astype('int')
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# Convert trans_date_trans_time to timestamp
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

# Train-test split (stratified on 'is_fraud')
train_df, test_df = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Drop 'trans_num' if present
if 'trans_num' in train_df.columns:
    train_df.drop('trans_num', axis=1, inplace=True)
if 'trans_num' in test_df.columns:
    test_df.drop('trans_num', axis=1, inplace=True)

# Separate features and labels
y_train = train_df['is_fraud']
X_train = train_df.drop('is_fraud', axis=1)
y_test  = test_df['is_fraud']
X_test  = test_df.drop('is_fraud', axis=1)
print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# Scale features
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)
X_train_sc = pd.DataFrame(X_train_sc, columns=X_train.columns)
X_test_sc  = pd.DataFrame(X_test_sc, columns=X_test.columns)

#####################################
# 2. SEQUENCE CREATION (STGN style)
#####################################
sequence_length = 10

def create_transactional_expansion(df, sequence_length):
    sequences, labels = [], []
    # Group by user (assumes 'cc_num' is the identifier)
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

# Append label column for sequence creation
train_seq_df = X_train_sc.copy()
train_seq_df['is_fraud'] = y_train.values
test_seq_df  = X_test_sc.copy()
test_seq_df['is_fraud'] = y_test.values

X_train_seq, y_train_seq = create_transactional_expansion(train_seq_df, sequence_length)
X_test_seq,  y_test_seq  = create_transactional_expansion(test_seq_df, sequence_length)
print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

####################################################
# 3. RANDOM FOREST TRAINING (Aggregated Features)
####################################################
# Aggregate each sequence (mean over time) for Random Forest
X_train_rf = X_train_seq.mean(axis=1)
X_test_rf  = X_test_seq.mean(axis=1)

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
rf_model.fit(X_train_rf, y_train_seq)

#############################################
# 4. DATASET & DATALOADER CREATION FOR GRU
#############################################
class SimpleDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X_seq = X_seq.astype(np.float32)
        self.y_seq = y_seq.astype(np.float32)
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        x = self.X_seq[idx]  # shape: (seq_len, num_features)
        label = np.array([self.y_seq[idx]], dtype=np.float32)
        return torch.tensor(x), torch.tensor(label)

batch_size = 4096
train_dataset = SimpleDataset(X_train_seq, y_train_seq)
test_dataset  = SimpleDataset(X_test_seq, y_test_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#############################################
# 5. EXPERT MODEL DEFINITIONS (GRU & RF)
#############################################
# GRU Expert
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FraudGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# RandomForest Expert (wrapper for scikit-learn model)
class RandomForestExpert(nn.Module):
    def __init__(self, rf_model):
        super(RandomForestExpert, self).__init__()
        self.rf_model = rf_model
    
    def forward(self, x):
        # x: (B, seq_len, input_dim); aggregate via average pooling over time
        x_avg = x.mean(dim=1).cpu().detach().numpy()
        proba = self.rf_model.predict_proba(x_avg)[:, 1]
        proba_tensor = torch.tensor(proba, dtype=torch.float32, device=x.device)
        return proba_tensor.unsqueeze(1)  # shape: (B, 1)

# Gating network for two experts
class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts=2):
        super(GateNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # x: (B, seq_len, input_dim) -> average over time
        x_avg = torch.mean(x, dim=1)
        gate_logits = self.fc(x_avg)
        gate_weights = F.softmax(gate_logits, dim=1)
        return gate_weights

# MOE Model combining GRU and Random Forest experts
class MOEModel(nn.Module):
    def __init__(self, input_size, gru_hidden_size, gru_num_layers, rf_model):
        super(MOEModel, self).__init__()
        self.expert_gru = FraudGRU(input_size, gru_hidden_size, gru_num_layers)
        self.expert_rf  = RandomForestExpert(rf_model)
        self.gate = GateNetwork(input_size, num_experts=2)
    
    def forward(self, x):
        # x: (B, seq_len, input_size)
        out_gru = self.expert_gru(x)   # (B, 1)
        out_rf  = self.expert_rf(x)    # (B, 1)
        expert_outputs = torch.cat([out_gru, out_rf], dim=1)  # (B, 2)
        gate_weights = self.gate(x)    # (B, 2)
        combined = torch.sum(gate_weights * expert_outputs, dim=1, keepdim=True)
        return combined

#############################################
# 6. TRAINING & EVALUATION
#############################################
input_size = X_train_seq.shape[2]
gru_hidden_size = 64
gru_num_layers = 2

model = MOEModel(input_size, gru_hidden_size, gru_num_layers, rf_model)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, labels in train_loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch).squeeze()
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, labels in data_loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            outputs = model(x_batch)
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

num_epochs = 50
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# Get predicted probabilities on train and test sets
y_pred_train_proba, y_train_seq_array = evaluate_model(model, train_loader, device)
y_pred_test_proba, y_test_seq_array   = evaluate_model(model, test_loader, device)

#############################################
# 7. THRESHOLD-BASED EVALUATION
#############################################
# Create results DataFrames for train and test sets
y_train_results = pd.DataFrame(y_pred_train_proba, columns=['pred_fraud'])
y_train_results['pred_not_fraud'] = 1 - y_train_results['pred_fraud']
y_train_results['y_train_actual'] = y_train_seq_array  # Actual labels for sequences

y_test_results = pd.DataFrame(y_pred_test_proba, columns=['pred_fraud'])
y_test_results['pred_not_fraud'] = 1 - y_test_results['pred_fraud']
y_test_results['y_test_actual'] = y_test_seq_array

# --- 4. Evaluation with different thresholds ---
numbers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cutoff_df = pd.DataFrame(columns=['Threshold', 'Accuracy', 'precision_score', 'recall_score', 'F1_score'])

# Map the probabilities to binary predictions for each threshold (Train)
for thr in numbers:
    y_train_results[thr] = y_train_results['pred_fraud'].map(lambda x: 1 if x > thr else 0)

print("Train Evaluation:")
for thr in numbers:
    cm1 = confusion_matrix(y_train_results['y_train_actual'], y_train_results[thr])
    TP, FP, FN, TN = cm1[1,1], cm1[0,1], cm1[1,0], cm1[0,0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    cutoff_df.loc[thr] = [thr, accuracy, precision, recall, f1_score_value]

print(cutoff_df)

best_idx = cutoff_df['F1_score'].idxmax()
best_threshold = cutoff_df.loc[best_idx, 'Threshold']
best_precision = cutoff_df.loc[best_idx, 'precision_score']
best_recall = cutoff_df.loc[best_idx, 'recall_score']
best_f1_score = cutoff_df.loc[best_idx, 'F1_score']
best_accuracy = cutoff_df.loc[best_idx, 'Accuracy']
best_auc = roc_auc_score(y_train_results['y_train_actual'], y_train_results['pred_fraud'])

print(f'Best Threshold (Train): {best_threshold:.4f}')
print(f'Best Accuracy (Train): {best_accuracy:.4f}')
print(f'Best Precision (Train): {best_precision:.4f}')
print(f'Best Recall (Train): {best_recall:.4f}')
print(f'Best F1 Score (Train): {best_f1_score:.4f}')
print(f'Best ROC_AUC Score (Train): {best_auc:.4f}')

# Repeat evaluation for Test set
cutoff_df_test = pd.DataFrame(columns=['Threshold', 'Accuracy', 'precision_score', 'recall_score', 'F1_score'])
for thr in numbers:
    y_test_results[thr] = y_test_results['pred_fraud'].map(lambda x: 1 if x > thr else 0)
    cm1 = confusion_matrix(y_test_results['y_test_actual'], y_test_results[thr])
    TP, FP, FN, TN = cm1[1,1], cm1[0,1], cm1[1,0], cm1[0,0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    cutoff_df_test.loc[thr] = [thr, accuracy, precision, recall, f1_score_value]

print("Test Evaluation:")
print(cutoff_df_test)

best_idx_test = cutoff_df_test['F1_score'].idxmax()
best_threshold_test = cutoff_df_test.loc[best_idx_test, 'Threshold']
best_precision_test = cutoff_df_test.loc[best_idx_test, 'precision_score']
best_recall_test = cutoff_df_test.loc[best_idx_test, 'recall_score']
best_f1_score_test = cutoff_df_test.loc[best_idx_test, 'F1_score']
best_accuracy_test = cutoff_df_test.loc[best_idx_test, 'Accuracy']
best_auc_test = roc_auc_score(y_test_results['y_test_actual'], y_test_results['pred_fraud'])

print(f'Best Threshold (Test): {best_threshold_test:.4f}')
print(f'Best Accuracy (Test): {best_accuracy_test:.4f}')
print(f'Best Precision (Test): {best_precision_test:.4f}')
print(f'Best Recall (Test): {best_recall_test:.4f}')
print(f'Best F1 Score (Test): {best_f1_score_test:.4f}')
print(f'Best ROC_AUC Score (Test): {best_auc_test:.4f}')