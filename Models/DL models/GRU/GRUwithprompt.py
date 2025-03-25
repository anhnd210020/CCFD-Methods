import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import random

#############################
# 1. Load & Preprocess Data #
#############################

df = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv')

# Process time: convert to datetime and extract hour
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

# Process date of birth: convert to datetime, compute age, and categorize
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

# Drop unnecessary columns (keep 'trans_date_trans_time' and 'cc_num')
drop_col = ['Unnamed: 0', 'first', 'last', 'street', 'state', 'lat',
            'long','dob', 'unix_time', 'cust_age', 'merch_lat', 'merch_long', 'city_pop', 'trans_num']
df.drop(drop_col, axis=1, inplace=True)

# Pivot table for cust_age_groups mapping to numeric indices
age_piv_2 = pd.pivot_table(data=df,
                           index='cust_age_groups',
                           columns='is_fraud',
                           values='amt',
                           aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Pivot table for category
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# Pivot table for job
job_txn_piv_2 = pd.pivot_table(data=df,
                               index='job',
                               columns='is_fraud',
                               values='amt',
                               aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

df['trans_hour'] = df['trans_hour'].astype('int')
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')


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
        f"I purchased {row['merchant']} at {time_str} in {row['city']}. It's really a good experience.",
        f"At {time_str}, I visited {row['merchant']} located in {row['city']} and spent ${row['amt']:.2f} on it. It was great.",
        f"I made a transaction at {row['merchant']} around {time_str} in {row['city']} and paid ${row['amt']:.2f}. I loved the service.",
        f"I went to {row['merchant']} at {time_str} in {row['city']} and enjoyed the purchase, spending ${row['amt']:.2f}.",
        f"During {time_str}, I bought from {row['merchant']} in {row['city']} for an amount of ${row['amt']:.2f}. It was satisfactory."
    ]
    return random.choice(templates)

df['text_prompt'] = df.apply(generate_synthetic_text, axis=1)

# Convert transaction time to numeric timestamp (after text generation)
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

#############################################
# 3. Split Data & Scale Numerical Features
#############################################

# Split into training and testing sets (stratified by fraud label)
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Separate features and labels
y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)
y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# For numerical scaling, drop columns used for text
train_numeric = X_train.drop(['cc_num', 'text_prompt', 'merchant', 'city'], axis=1)
test_numeric = X_test.drop(['cc_num', 'text_prompt', 'merchant', 'city'], axis=1)

sc = StandardScaler()
X_train_sc_numeric = pd.DataFrame(sc.fit_transform(train_numeric),
                                  columns=train_numeric.columns,
                                  index=train_numeric.index)
X_test_sc_numeric = pd.DataFrame(sc.transform(test_numeric),
                                 columns=test_numeric.columns,
                                 index=test_numeric.index)

# Reconstruct DataFrames by adding back identifier and text columns
X_train_sc = X_train_sc_numeric.copy()
X_train_sc['cc_num'] = X_train['cc_num']
X_train_sc['text_prompt'] = X_train['text_prompt']

X_test_sc = X_test_sc_numeric.copy()
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

sequence_length = 10  # number of transactions per sequence

def create_sequences_with_text(df, sequence_length):
    """
    Groups transactions by cc_num and creates fixed-length sequences.
    For each sequence, collects the synthetic text prompt from the latest transaction.
    """
    sequences, text_prompts, labels = [], [], []
    grouped = df.groupby('cc_num')
    for user_id, group in grouped:
        group = group.sort_values(by='trans_date_trans_time')
        numeric_data = group.drop(columns=['is_fraud', 'cc_num', 'text_prompt']).values
        targets = group['is_fraud'].values
        for i in range(len(group)):
            if i < sequence_length:
                pad_needed = sequence_length - (i + 1)
                pad = np.repeat(numeric_data[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, numeric_data[:i+1]), axis=0)
            else:
                seq = numeric_data[i-sequence_length+1:i+1]
            sequences.append(seq)
            text_prompts.append(group.iloc[i]['text_prompt'])
            labels.append(targets[i])
    return np.array(sequences), text_prompts, np.array(labels)

X_train_seq, train_texts, y_train_seq = create_sequences_with_text(X_train_sc, sequence_length)
X_test_seq, test_texts, y_test_seq = create_sequences_with_text(X_test_sc, sequence_length)

print("Transaction sequence shape (train):", X_train_seq.shape)
print("Transaction sequence shape (test):", X_test_seq.shape)

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

batch_size = 32768
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

##############################################
# 8. Train the Multi-Modal Model             #
##############################################

num_epochs = 80  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
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
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

#######################################################
# 9. Evaluate the Model Using an Inference Prompt      #
#######################################################

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

print("Test Evaluation with Inference Prompt:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
