import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import random

#############################
# 1. Load & Preprocess Data #
#############################

# Loading the data
file_path = '/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv'
df = pd.read_csv(file_path)

# ------------------ Data Preprocessing ------------------
# Chuyển đổi thời gian và tạo các feature liên quan
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_date_trans_time_numeric'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())
# Lấy giờ giao dịch: lấy phần đầu của chuỗi time (dạng hh)
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

# Xử lý ngày sinh và tính tuổi khách hàng
df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 else (
    '10-20' if x >= 10 and x < 20 else (
    '20-30' if x >= 20 and x < 30 else (
    '30-40' if x >= 30 and x < 40 else (
    '40-50' if x >= 40 and x < 50 else (
    '50-60' if x >= 50 and x < 60 else (
    '60-70' if x >= 60 and x < 70 else (
    '70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

# Mapping nhóm tuổi theo giá trị trung bình của amt trong giao dịch gian lận
age_piv_2 = pd.pivot_table(data=df,
                           index='cust_age_groups',
                           columns='is_fraud',
                           values='amt',
                           aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Encode các biến category và job dựa trên giá trị trung bình của amt trong giao dịch gian lận
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

# Encode các biến định danh bằng factorize
df['merchant_num'] = pd.factorize(df['merchant'])[0]
df['last_num'] = pd.factorize(df['last'])[0]
df['street_num'] = pd.factorize(df['street'])[0]
df['city_num'] = pd.factorize(df['city'])[0]
df['zip_num'] = pd.factorize(df['zip'])[0]
df['state_num'] = pd.factorize(df['state'])[0]
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# Bỏ các cột không cần thiết
drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'street', 'city', 'state', 
             'lat', 'long', 'dob', 'unix_time', 'merch_lat', 'merch_long', 'city_pop']
df.drop(columns=drop_cols, errors='ignore', inplace=True)

#############################################
# 2. Generate Synthetic Text from Transactions
#############################################

def generate_synthetic_text(row):
    """
    Generates synthetic text based on transaction details.
    Requires: 'merchant', 'trans_hour', 'city', and 'amt' in the row.
    """
    time_str = row['trans_hour']
    templates = [
        f"I purchased {row['merchant_num']} at {time_str} in {row['city_num']}. It's really a good experience.",
        f"At {time_str}, I visited {row['merchant_num']} located in {row['city_num']} and spent ${row['amt']:.2f} on it. It was great.",
        f"I made a transaction at {row['merchant_num']} around {time_str} in {row['city_num']} and paid ${row['amt']:.2f}. I loved the service.",
        f"I went to {row['merchant_num']} at {time_str} in {row['city_num']} and enjoyed the purchase, spending ${row['amt']:.2f}.",
        f"During {time_str}, I bought from {row['merchant_num']} in {row['city_num']} for an amount of ${row['amt']:.2f}. It was satisfactory."
    ]
    return random.choice(templates)

df['text_prompt'] = df.apply(generate_synthetic_text, axis=1)

#############################################
# 3. Split Data & Scale Numerical Features
#############################################

# Split into training and testing sets (stratified by fraud label)
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Shape of training data:", train.shape)
print("Shape of testing data:", test.shape)

# Loại bỏ cột trans_num nếu có (sẽ bị drop nếu không tồn tại)
train.drop('trans_num', axis=1, inplace=True, errors='ignore')
test.drop('trans_num', axis=1, inplace=True, errors='ignore')

# Tách dữ liệu thành features và label
y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)

y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data: ', (X_train.shape, y_train.shape))
print('Shape of testing data: ', (X_test.shape, y_test.shape))

sc = StandardScaler()

# Reconstruct DataFrames by adding back identifier and text columns
X_train_sc['cc_num'] = X_train['cc_num']
X_train_sc['text_prompt'] = X_train['text_prompt']

X_test_sc['cc_num'] = X_test['cc_num']
X_test_sc['text_prompt'] = X_test['text_prompt']

# -------------------------------
# **Modification Step:**
# Merge the 'is_fraud' labels back into the scaled DataFrames
X_train_sc['is_fraud'] = y_train.values
X_test_sc['is_fraud'] = y_test.values
# -------------------------------

#######################################################
# 4. Create Sequences and Collect Text for Each Sample
#######################################################

def create_sequences_transactional_expansion(df, memory_size):
    sequences, labels = [], []
    
    # Nhóm theo 'cc_num' (số thẻ tín dụng của người dùng)
    grouped = df.groupby('cc_num')
    
    for user_id, group in grouped:
        # Sắp xếp theo thời gian (đảm bảo rằng trans_date_trans_time đã là timestamp)
        group = group.sort_values(by='trans_date_trans_time_numeric')
        
        # Lấy các giá trị (loại bỏ 'is_fraud' và 'cc_num' vì đây là features)
        values = group.drop(columns=['is_fraud', 'cc_num']).values
        targets = group['is_fraud'].values
        
        n = len(group)
        
        # Tạo chuỗi giao dịch cho mỗi giao dịch
        for i in range(n):
            if i < memory_size:
                # Nếu số giao dịch hiện tại ít hơn 'memory_size', sao chép giao dịch đầu tiên
                pad_needed = memory_size - (i + 1)
                # Sao chép giao dịch đầu tiên cho đủ số lượng pad
                pad = np.repeat(values[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, values[:i+1]), axis=0)
            else:
                # Nếu đủ giao dịch, lấy sequence gồm các giao dịch từ (i - memory_size + 1) đến i
                seq = values[i-memory_size+1:i+1]
            
            # Thêm sequence và label vào danh sách
            sequences.append(seq)
            labels.append(targets[i])
    
    return np.array(sequences), np.array(labels)

memory_size = 800  # Chọn memory_size như một giá trị cố định (hoặc có thể thử các giá trị khác như 30, 50)
train_seq_df = X_train_sc.copy()
train_seq_df['is_fraud'] = y_train.values

test_seq_df = X_test_sc.copy()
test_seq_df['is_fraud'] = y_test.values

# Tạo các chuỗi theo phương pháp Transactional Expansion
X_train_seq, y_train_seq = create_sequences_transactional_expansion(train_seq_df, memory_size)
X_test_seq, y_test_seq = create_sequences_transactional_expansion(test_seq_df, memory_size)

print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

#################################################
# 5. Build a Simple Text Tokenizer & Vocabulary #
#################################################

def build_vocab(texts, min_freq=1):
    from collections import Counter
    counter = Counter()
    for text in texts:
        tokens = text.lower().split()
        counter.update(tokens)
    # Reserve indices: 0 for <PAD> and 1 for <UNK>
    vocab = {word: idx+2 for idx, (word, count) in enumerate(counter.items()) if count >= min_freq}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

vocab = build_vocab(train_texts, min_freq=1)
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)

def tokenize(text, vocab, max_length):
    tokens = text.lower().split()
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(token_ids) < max_length:
        token_ids = [vocab["<PAD>"]] * (max_length - len(token_ids)) + token_ids
    else:
        token_ids = token_ids[:max_length]
    return token_ids

text_seq_len = 10  # maximum token length for the text
train_text_tokens = [tokenize(text, vocab, text_seq_len) for text in train_texts]
test_text_tokens = [tokenize(text, vocab, text_seq_len) for text in test_texts]

train_text_tokens = np.array(train_text_tokens)
test_text_tokens = np.array(test_text_tokens)

##############################################
# 6. Create a Multi-Modal Dataset Class      #
##############################################

class MultiModalFraudDataset(Dataset):
    def __init__(self, transaction_data, text_data, labels, override_text=None):
        """
        transaction_data: numpy array of shape (num_samples, sequence_length, num_transaction_features)
        text_data: numpy array of shape (num_samples, text_seq_len) containing token ids
        labels: numpy array of fraud labels
        override_text: if provided (as token ids), it will override the text input for every sample
        """
        self.transaction_data = transaction_data
        self.text_data = text_data
        self.labels = labels
        self.override_text = override_text

    def __len__(self):
        return len(self.transaction_data)

    def __getitem__(self, idx):
        trans = torch.tensor(self.transaction_data[idx], dtype=torch.float32)
        if self.override_text is not None:
            text = torch.tensor(self.override_text, dtype=torch.long)
        else:
            text = torch.tensor(self.text_data[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return trans, text, label

batch_size = 64
train_dataset = MultiModalFraudDataset(X_train_seq, train_text_tokens, y_train_seq)
test_dataset = MultiModalFraudDataset(X_test_seq, test_text_tokens, y_test_seq)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

##############################################
# 7. Define the Multi-Modal Neural Model     #
##############################################

class MultiModalFraudModel(nn.Module):
    def __init__(self, 
                 transaction_input_size, 
                 transaction_hidden_size, 
                 text_vocab_size, 
                 text_embed_dim, 
                 text_hidden_size, 
                 num_layers, 
                 fc_hidden_size=64):
        super(MultiModalFraudModel, self).__init__()
        # Transaction branch (GRU)
        self.transaction_gru = nn.GRU(transaction_input_size, transaction_hidden_size, num_layers, batch_first=True)
        # Text branch: Embedding + GRU
        self.text_embedding = nn.Embedding(text_vocab_size, text_embed_dim)
        self.text_gru = nn.GRU(text_embed_dim, text_hidden_size, num_layers, batch_first=True)
        # Combine both branches
        self.fc = nn.Linear(transaction_hidden_size + text_hidden_size, fc_hidden_size)
        self.out = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, transaction_data, text_data):
        # Process transaction data
        trans_out, _ = self.transaction_gru(transaction_data)
        trans_feat = trans_out[:, -1, :]  # last hidden state
        # Process text data
        text_embed = self.text_embedding(text_data)
        text_out, _ = self.text_gru(text_embed)
        text_feat = text_out[:, -1, :]  # last hidden state
        # Concatenate features and predict
        combined = torch.cat((trans_feat, text_feat), dim=1)
        x = torch.relu(self.fc(combined))
        x = self.out(x)
        return self.sigmoid(x)

# Model hyperparameters
transaction_input_size = X_train_seq.shape[2]
transaction_hidden_size = 64
text_embed_dim = 16
text_hidden_size = 32
num_layers = 2

model = MultiModalFraudModel(transaction_input_size, transaction_hidden_size,
                             vocab_size, text_embed_dim, text_hidden_size, num_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

######################################
# 8. Evaluation and Training Functions
######################################

# 5️⃣ Evaluation Function for Multi-Modal Data
def evaluate_model(loader, model, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for trans_batch, text_batch, label_batch in loader:
            trans_batch = trans_batch.to(device)
            text_batch = text_batch.to(device)
            outputs = model(trans_batch, text_batch).squeeze().cpu().numpy()
            all_preds.extend(outputs)
            all_targets.extend(label_batch.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute ROC AUC score
    auc = roc_auc_score(all_targets, all_preds)
    
    # Search for the best threshold (using thresholds 0.1 to 0.9) based on F1 score
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_f1 = 0
    best_threshold = 0.5
    for t in thresholds:
        binary_preds = (all_preds > t).astype(int)
        f1 = f1_score(all_targets, binary_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    combined_metric = (best_f1 + auc) / 2
    
    # Compute additional metrics using the best threshold
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

# 6️⃣ Training Function for Multi-Modal Data (without checkpoint saving)
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    best_combined_metric_test = -float('inf')
    epochs_without_improvement = 0

    # Variables to store the best epoch and metrics
    best_epoch = None
    best_train_metrics = None
    best_test_metrics = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for trans_batch, text_batch, label_batch in train_loader:
            trans_batch = trans_batch.to(device)
            text_batch = text_batch.to(device)
            label_batch = label_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(trans_batch, text_batch).squeeze()
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1}, Loss: {average_loss:.4f}')
        
        # Evaluate on the training set
        train_threshold, train_f1, train_auc, train_combined, train_acc, train_prec, train_rec = evaluate_model(train_loader, model, device)
        print(f"Train Metrics - Best Threshold: {train_threshold:.2f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, Combined: {train_combined:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        
        # Evaluate on the test set
        test_threshold, test_f1, test_auc, test_combined, test_acc, test_prec, test_rec = evaluate_model(test_loader, model, device)
        print(f"Test Metrics  - Best Threshold: {test_threshold:.2f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Combined: {test_combined:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
        
        # Update best metrics based on test_combined
        if test_combined > best_combined_metric_test:
            best_combined_metric_test = test_combined
            best_epoch = epoch + 1
            best_train_metrics = (train_f1, train_auc, train_combined)
            best_test_metrics = (test_f1, test_auc, test_combined)
            print(f'*** Best metrics updated at epoch {epoch+1} ***')
        
        # Early stopping: if loss does not improve in 8 consecutive epochs
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

######################################
# 9. Train the Multi-Modal Model
######################################

num_epochs = 100  # Adjust as needed
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

######################################
# 10. Evaluate the Model Using an Inference Prompt
######################################

# For inference, override the text input with a custom prompt.
inference_prompt = "I hate this man. He is my exlover. I want to check if he is fraud or not."
inference_prompt_tokens = tokenize(inference_prompt, vocab, text_seq_len)

# Create a test dataset that overrides the text with the inference prompt for all samples.
test_dataset_inference = MultiModalFraudDataset(X_test_seq, test_text_tokens, y_test_seq, override_text=inference_prompt_tokens)
test_loader_inference = DataLoader(test_dataset_inference, batch_size=batch_size, shuffle=False)

model.eval()
y_pred_test_proba = []
with torch.no_grad():
    for trans_batch, text_batch, label_batch in test_loader_inference:
        trans_batch = trans_batch.to(device)
        text_batch = text_batch.to(device)
        outputs = model(trans_batch, text_batch).squeeze().cpu().numpy()
        y_pred_test_proba.extend(outputs)
y_pred_test_proba = np.array(y_pred_test_proba)
y_pred_test = (y_pred_test_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test_seq, y_pred_test)
precision = precision_score(y_test_seq, y_pred_test, zero_division=0)
recall = recall_score(y_test_seq, y_pred_test, zero_division=0)
f1 = f1_score(y_test_seq, y_pred_test, zero_division=0)
roc_auc = roc_auc_score(y_test_seq, y_pred_test_proba)

print("\nTest Evaluation with Inference Prompt:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")