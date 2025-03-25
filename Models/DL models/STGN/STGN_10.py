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

df_train = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv')
df_test = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTest.csv')
df = pd.concat([df_train, df_test])

# Xử lý thời gian
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 else ('10-20' if x >= 10 and x < 20 else ('20-30' if x >= 20 and x < 30 else ('30-40' if x >= 30 and x < 40 else ('40-50' if x >= 40 and x < 50 else ('50-60' if x >= 50 and x < 60 else ('60-70' if x >= 60 and x < 70 else ('70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

drop_col = ['Unnamed: 0', 'merchant', 'first', 'last', 'street', 'city', 'state', 
            'dob', 'unix_time', 'cust_age', 'city_pop']
df.drop(drop_col, axis=1, inplace=True)

# %% Process 'cust_age_groups'
age_piv_2 = pd.pivot_table(data=df,
                           index='cust_age_groups',
                           columns='is_fraud',
                           values='amt',
                           aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# %% Process 'category'
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# %% Process 'job'
job_txn_piv_2 = pd.pivot_table(data=df,
                               index='job',
                               columns='is_fraud',
                               values='amt',
                               aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

# %% Process 'trans_hour' and one-hot encode 'gender'
df['trans_hour'] = df['trans_hour'].astype('int')
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# %% Convert 'trans_date_trans_time' to timestamp (numerical)
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

# %% Train-test split (stratified on 'is_fraud')
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# %% Drop the 'trans_num' column from train and test sets
train.drop('trans_num', axis=1, inplace=True)
test.drop('trans_num', axis=1, inplace=True)

y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)
y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# %% Scale the data using StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

# Set the desired sequence length (d + 1, where d is the memory size)
sequence_length = 10  

def create_transactional_expansion(df, sequence_length):
    sequences, labels = [], []
    # Group transactions by user (assuming 'cc_num' is the user identifier)
    grouped = df.groupby('cc_num')
    for user_id, group in grouped:
        # Sort transactions by time (trans_date_trans_time is already in timestamp format)
        group = group.sort_values(by='trans_date_trans_time')
        # Extract feature values: drop 'is_fraud' and 'cc_num' from the features
        values = group.drop(columns=['is_fraud', 'cc_num']).values
        targets = group['is_fraud'].values
        n = len(group)
        for i in range(n):
            if i < sequence_length:
                # Not enough transactions to form a full sequence:
                # Calculate how many pads are needed
                pad_needed = sequence_length - (i + 1)
                # Repeat the earliest transaction to pad the sequence
                pad = np.repeat(values[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, values[:i+1]), axis=0)
            else:
                # When there are enough transactions, take the last 'sequence_length' transactions
                seq = values[i-sequence_length+1:i+1]
            sequences.append(seq)
            labels.append(targets[i])
    return np.array(sequences), np.array(labels)

train_seq_df = X_train_sc.copy()
train_seq_df['is_fraud'] = y_train.values
test_seq_df = X_test_sc.copy()
test_seq_df['is_fraud'] = y_test.values

X_train_seq, y_train_seq = create_transactional_expansion(train_seq_df, sequence_length)
X_test_seq, y_test_seq = create_transactional_expansion(test_seq_df, sequence_length)

print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)
# -------------------------------
# 1. STGN Basic Recurrent Unit
# -------------------------------
class STGNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(STGNCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Forget gate: f_t = σ(W_f^h * h_{t-1} + W_f^x * x_t + b_f)
        self.W_fh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_fx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))
        
        # Input gate: i_t = σ(W_ih * h_{t-1} + W_ix * x_t + b_i)
        self.W_ih = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ix = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))
        
        # Time-aware gate: T_t = σ(W_T^h * h_{t-1} + W_T^x * x_t + W_T^{1T} * ΔT + b_T)
        self.W_Th = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_Tx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_T1T = nn.Linear(1, hidden_dim, bias=False)  # Note: should be < 0 (constraint not enforced here)
        self.b_T = nn.Parameter(torch.zeros(hidden_dim))
        
        # Location-aware gate: L_t = σ(W_L^h * h_{t-1} + W_L^x * x_t + W_L^{1L} * ΔL + b_L)
        self.W_Lh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_Lx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_L1L = nn.Linear(1, hidden_dim, bias=False)  # Note: should be < 0
        self.b_L = nn.Parameter(torch.zeros(hidden_dim))
        
        # Spatial–temporal behavioral state: ζ = tanh(W_ζ^{1T}*ΔT + W_ζ^{1L}*ΔL)
        self.W_zeta_T = nn.Linear(1, hidden_dim, bias=False)
        self.W_zeta_L = nn.Linear(1, hidden_dim, bias=False)
        
        # Candidate cell state: c̃_t = tanh(W_u^h * h_{t-1} + W_u^x * x_t + b_u)
        self.W_uh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ux = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_u = nn.Parameter(torch.zeros(hidden_dim))
        
        # Output gate: o_t = σ(W_o^h * h_{t-1} + W_o^x * x_t + W_o^ζ * ζ + b_o)
        self.W_oh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ox = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_o_zeta = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x_t, deltaT, deltaL, h_prev, c_prev):
        # Forget gate
        f_t = torch.sigmoid(self.W_fh(h_prev) + self.W_fx(x_t) + self.b_f)
        # Input gate
        i_t = torch.sigmoid(self.W_ih(h_prev) + self.W_ix(x_t) + self.b_i)
        # Time-aware gate
        T_t = torch.sigmoid(self.W_Th(h_prev) + self.W_Tx(x_t) + self.W_T1T(deltaT) + self.b_T)
        # Location-aware gate
        L_t = torch.sigmoid(self.W_Lh(h_prev) + self.W_Lx(x_t) + self.W_L1L(deltaL) + self.b_L)
        # Spatial-temporal behavioral state ζ
        zeta = torch.tanh(self.W_zeta_T(deltaT) + self.W_zeta_L(deltaL))
        # Candidate cell state
        c_tilde = torch.tanh(self.W_uh(h_prev) + self.W_ux(x_t) + self.b_u)
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde * T_t * L_t
        # Output gate
        o_t = torch.sigmoid(self.W_oh(h_prev) + self.W_ox(x_t) + self.W_o_zeta(zeta) + self.b_o)
        # Candidate hidden state
        h_tilde = o_t * torch.tanh(c_t)
        return h_tilde, c_t

# -----------------------------------------
# 2. Spatial–Temporal Attention Module
# -----------------------------------------
class SpatialTemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SpatialTemporalAttention, self).__init__()
        # Linear projections for current state and historical state
        self.W_I = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # for current state (h̃ and c)
        self.W_Ih = nn.Linear(hidden_dim, hidden_dim, bias=False)       # for historical state
        self.W_I1T = nn.Linear(1, hidden_dim, bias=False)               # for time gap between history and current
        self.W_I1L = nn.Linear(1, hidden_dim, bias=False)               # for location gap
        self.b_I = nn.Parameter(torch.zeros(hidden_dim))
        
        # Nonlinear scoring for each historical state
        self.W_oI = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_oI = nn.Parameter(torch.zeros(hidden_dim))
        # Importance vector for final score
        self.v_t = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, h_tilde, c_t, h_history, deltaT_hist, deltaL_hist):
        # Current combined state: ĥ_t = [h̃_t; c_t]
        h_hat = torch.cat([h_tilde, c_t], dim=1)  # shape (B, 2*hidden_dim)
        # Project current state
        current_proj = self.W_I(h_hat)  # (B, hidden_dim)
        
        # For each historical time step, compute attention score
        # h_history: list of previous hidden states, each (B, hidden_dim)
        # deltaT_hist and deltaL_hist: list of time and location gap tensors for each historical step, each (B, 1)
        scores = []
        for i, h_i in enumerate(h_history):
            # For each historical hidden state h_i, compute:
            # I_{t,i} = current_proj + W_Ih * h_i + W_I1T * deltaT_hist[i] + W_I1L * deltaL_hist[i] + b_I
            I_ti = current_proj + self.W_Ih(h_i) + self.W_I1T(deltaT_hist[i]) + self.W_I1L(deltaL_hist[i]) + self.b_I
            # Score: o_{t,i} = tanh(W_oI * I_ti + b_oI)
            o_ti = torch.tanh(self.W_oI(I_ti) + self.b_oI)  # (B, hidden_dim)
            # Final scalar score by dot product with v_t:
            score = torch.sum(o_ti * self.v_t, dim=1, keepdim=True)  # (B, 1)
            scores.append(score)
        # Concatenate scores: shape (B, s) where s is the number of history steps
        scores_tensor = torch.cat(scores, dim=1)
        # Attention weights: softmax over the history dimension
        alpha = F.softmax(scores_tensor, dim=1)  # (B, s)
        # Compute context vector: weighted sum of historical hidden states
        # First stack h_history: (B, s, hidden_dim)
        h_history_stack = torch.stack(h_history, dim=1)
        # Expand alpha: (B, s, 1)
        alpha_expanded = alpha.unsqueeze(-1)
        # Context vector s_t:
        s_t = torch.sum(alpha_expanded * h_history_stack, dim=1)  # (B, hidden_dim)
        return s_t, alpha

# -----------------------------------------
# 3. Interaction Module
# -----------------------------------------
class InteractionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InteractionModule, self).__init__()
        # This module fuses the candidate hidden state, attention context, and user representation
        self.W_h_tilde = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ru = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_int = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, h_tilde, s_t, r_u):
        out = self.W_h_tilde(h_tilde) + self.W_s(s_t) + self.W_ru(r_u) + self.b_int
        h_t = torch.tanh(out)
        return h_t

# -----------------------------------------
# 4. STGN Model
# -----------------------------------------
class STGNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size):
        """
        input_dim: number of features per transaction
        hidden_dim: number of nodes in the recurrent unit
        memory_size: number of previous steps to use in attention (s)
        """
        super(STGNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        self.cell = STGNCell(input_dim, hidden_dim)
        self.attention = SpatialTemporalAttention(hidden_dim)
        self.interaction = InteractionModule(input_dim, hidden_dim)
        # To compute user representation: here we use a simple mean of all candidate hidden states in the sequence
        self.W_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))
        
        # Final classifier for binary classification (fraud detection)
        self.classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, x_seq, deltaT_seq, deltaL_seq):
        """
        x_seq: (B, seq_len, input_dim) – input features at each time step
        deltaT_seq: (B, seq_len, 1) – time gap between current and previous transaction (for each step)
        deltaL_seq: (B, seq_len, 1) – location gap between current and previous transaction
        """
        batch_size, seq_len, _ = x_seq.shape
        
        # Initialize hidden and cell states to zero
        h_prev = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        c_prev = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        
        # Lists to store candidate hidden states (for user representation) and final hidden states (for attention)
        candidate_h_list = []
        final_h_list = []
        
        # Also keep lists for attention: we will store final hidden states along with their corresponding time and location gaps
        history_states = []
        history_deltaT = []
        history_deltaL = []
        
        outputs = []  # logits for each time step
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]             # (B, input_dim)
            deltaT_t = deltaT_seq[:, t, :]     # (B, 1)
            deltaL_t = deltaL_seq[:, t, :]     # (B, 1)
            
            # Compute candidate hidden state and cell state using STGNCell
            h_tilde, c_t = self.cell(x_t, deltaT_t, deltaL_t, h_prev, c_prev)
            candidate_h_list.append(h_tilde)
            
            # For attention, we need to consider previous final hidden states.
            # We store final hidden state once computed.
            if len(history_states) == 0:
                # If no history available, set attention context to zeros
                s_t = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
            else:
                # Use the last 'memory_size' states
                recent_states = history_states[-self.memory_size:]
                recent_deltaT = history_deltaT[-self.memory_size:]
                recent_deltaL = history_deltaL[-self.memory_size:]
                s_t, _ = self.attention(h_tilde, c_t, recent_states, recent_deltaT, recent_deltaL)
            
            # Compute user representation r_u as the mean of all candidate hidden states so far
            all_candidates = torch.stack(candidate_h_list, dim=1)  # (B, t+1, hidden_dim)
            r_u = torch.tanh(self.W_r(torch.mean(all_candidates, dim=1)) + self.b_r)  # (B, hidden_dim)
            
            # Interaction module fuses h_tilde, s_t, and r_u to form final representation
            h_t = self.interaction(h_tilde, s_t, r_u)
            final_h_list.append(h_t)
            
            # Append current final hidden state to history (for future attention)
            history_states.append(h_t)
            # For attention, we assume the gaps for history are relative to current time:
            # Here we simply use the current deltaT_t and deltaL_t as proxies (in practice these may be computed pairwise)
            history_deltaT.append(deltaT_t)
            history_deltaL.append(deltaL_t)
            
            # Update h_prev and c_prev for next time step
            h_prev, c_prev = h_t, c_t
            
            # Prediction for current time step using final representation
            logits = self.classifier(h_t)  # (B, 1)
            outputs.append(logits.unsqueeze(1))
        
        # Concatenate outputs along time dimension: (B, seq_len, 1)
        outputs = torch.cat(outputs, dim=1)
        # For binary classification, we apply sigmoid to get probabilities
        y_pred = torch.sigmoid(outputs)
        return y_pred

# %% DataLoader Preparation for STGN
batch_size = 2048

# Convert training and testing sequences to tensors (assumes X_train_seq and X_test_seq are available)
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)

# Create separate tensors for time gap (deltaT) and location gap (deltaL)
deltaT_train_tensor = torch.abs(torch.randn(X_train_tensor.size(0), sequence_length, 1))
deltaT_test_tensor = torch.abs(torch.randn(X_test_tensor.size(0), sequence_length, 1))
deltaL_train_tensor = torch.abs(torch.randn(X_train_tensor.size(0), sequence_length, 1))
deltaL_test_tensor = torch.abs(torch.randn(X_test_tensor.size(0), sequence_length, 1))

# Convert labels to tensors with shape (N, 1)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(-1)

# Create datasets and DataLoaders without the extra "g" tensor
train_dataset = TensorDataset(X_train_tensor, deltaT_train_tensor, deltaL_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, deltaT_test_tensor, deltaL_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Model, Loss, and Optimizer Setup for STGN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the STGN model (input_dim, hidden_dim, memory_size)
model = STGNModel(input_dim=X_train_tensor.size(2), hidden_dim=64, memory_size=3).to(device)

# Since the STGN model returns probabilities (after sigmoid), use BCELoss
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# %% Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, deltaT_batch, deltaL_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            deltaT_batch = deltaT_batch.to(device)
            deltaL_batch = deltaL_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch, deltaT_batch, deltaL_batch)  # (B, seq_len, 1)
            # Use the final time step prediction for classification
            final_output = outputs[:, -1, :]  # shape: (B, 1)
            loss = criterion(final_output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

num_epochs = 20
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# %% Evaluation
model.eval()
y_pred_train_proba = []
with torch.no_grad():
    for X_batch, deltaT_batch, deltaL_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        deltaT_batch = deltaT_batch.to(device)
        deltaL_batch = deltaL_batch.to(device)
        outputs = model(X_batch, deltaT_batch, deltaL_batch)  # (B, seq_len, 1)
        final_output = outputs[:, -1, :]  # (B, 1)
        y_pred_train_proba.extend(final_output.cpu().numpy())
y_pred_train_proba = np.array(y_pred_train_proba)

y_pred_test_proba = []
with torch.no_grad():
    for X_batch, deltaT_batch, deltaL_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        deltaT_batch = deltaT_batch.to(device)
        deltaL_batch = deltaL_batch.to(device)
        outputs = model(X_batch, deltaT_batch, deltaL_batch)
        final_output = outputs[:, -1, :]
        y_pred_test_proba.extend(final_output.cpu().numpy())
y_pred_test_proba = np.array(y_pred_test_proba)

# Create DataFrames with prediction results
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
