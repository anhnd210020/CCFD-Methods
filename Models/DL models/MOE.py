import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
# 3. RANDOM FOREST TRAINING (for RandomForestExpert)
####################################################
# For the RandomForest expert, we aggregate each sequence (mean over time)
X_train_rf = X_train_seq.mean(axis=1)  # shape: (num_sequences, input_size)
X_test_rf  = X_test_seq.mean(axis=1)

# Train a RandomForestClassifier using the aggregated features
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
rf_model.fit(X_train_rf, y_train_seq)  # y_train_seq is a numpy array of labels

####################################################
# 4. DATASET & DATALOADER CREATION FOR MOE MODEL
####################################################

class MOEDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X_seq = X_seq.astype(np.float32)  # (N, seq_len, num_features)
        self.y_seq = y_seq.astype(np.float32)  # (N,)
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        x = self.X_seq[idx]  # (seq_len, num_features)
        # For TH-LSTM: create delta (random positive values) and g (copy of x)
        delta = np.abs(np.random.randn(sequence_length, 1)).astype(np.float32)
        g = x.astype(np.float32)
        # For STGN: create deltaT and deltaL (random positive values)
        deltaT = np.abs(np.random.randn(sequence_length, 1)).astype(np.float32)
        deltaL = np.abs(np.random.randn(sequence_length, 1)).astype(np.float32)
        label = np.array([self.y_seq[idx]], dtype=np.float32)
        return {
            'x': torch.tensor(x),
            'delta': torch.tensor(delta),
            'g': torch.tensor(g),
            'deltaT': torch.tensor(deltaT),
            'deltaL': torch.tensor(deltaL)
        }, torch.tensor(label)

batch_size = 64
train_dataset = MOEDataset(X_train_seq, y_train_seq)
test_dataset  = MOEDataset(X_test_seq, y_test_seq)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

##############################################################
# 5. EXPERT MODEL DEFINITIONS (GRU, LSTM, TH-LSTM, STGN, RF)
##############################################################

# 5.1 GRU Expert
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FraudGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# 5.2 LSTM Expert
class FraudLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FraudLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# 5.3 TH-LSTM Expert (using provided code)
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
        q_t = torch.cat([h_tilde, c_t], dim=1)
        h_hist_stack = torch.stack(h_history, dim=1)
        q_t_proj = self.Waq(q_t)
        q_t_expanded = q_t_proj.unsqueeze(1)
        h_hist_proj = self.Wah(h_hist_stack)
        score_pre = torch.tanh(q_t_expanded + h_hist_proj + self.ba)
        alpha_unscaled = torch.einsum('bdh,h->bd', score_pre, self.v_t)
        alpha = F.softmax(alpha_unscaled, dim=1)
        alpha_expanded = alpha.unsqueeze(-1)
        e_t = (alpha_expanded * h_hist_stack).sum(dim=1)
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

class THLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size):
        super(THLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.thlstm_cell = THLSTMCell(input_dim, hidden_dim)
        self.attention_module = CurrentHistoricalAttention(hidden_dim)
        self.interaction_module = InteractionModule(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_seq, delta_seq, g_seq):
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
            logits = self.classifier(h_t)
            outputs.append(logits.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return torch.sigmoid(outputs)[:, -1, :]

# 5.4 STGN Expert
class STGNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(STGNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_fh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_fx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_ih = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ix = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_Th = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_Tx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_T1T = nn.Linear(1, hidden_dim, bias=False)
        self.b_T = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_Lh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_Lx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_L1L = nn.Linear(1, hidden_dim, bias=False)
        self.b_L = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_zeta_T = nn.Linear(1, hidden_dim, bias=False)
        self.W_zeta_L = nn.Linear(1, hidden_dim, bias=False)
        
        self.W_uh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ux = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_u = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_oh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ox = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_o_zeta = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x_t, deltaT, deltaL, h_prev, c_prev):
        f_t = torch.sigmoid(self.W_fh(h_prev) + self.W_fx(x_t) + self.b_f)
        i_t = torch.sigmoid(self.W_ih(h_prev) + self.W_ix(x_t) + self.b_i)
        T_t = torch.sigmoid(self.W_Th(h_prev) + self.W_Tx(x_t) + self.W_T1T(deltaT) + self.b_T)
        L_t = torch.sigmoid(self.W_Lh(h_prev) + self.W_Lx(x_t) + self.W_L1L(deltaL) + self.b_L)
        zeta = torch.tanh(self.W_zeta_T(deltaT) + self.W_zeta_L(deltaL))
        c_tilde = torch.tanh(self.W_uh(h_prev) + self.W_ux(x_t) + self.b_u)
        c_t = f_t * c_prev + i_t * c_tilde * T_t * L_t
        o_t = torch.sigmoid(self.W_oh(h_prev) + self.W_ox(x_t) + self.W_o_zeta(zeta) + self.b_o)
        h_tilde = o_t * torch.tanh(c_t)
        return h_tilde, c_t

class SpatialTemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SpatialTemporalAttention, self).__init__()
        self.W_I = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.W_Ih = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_I1T = nn.Linear(1, hidden_dim, bias=False)
        self.W_I1L = nn.Linear(1, hidden_dim, bias=False)
        self.b_I = nn.Parameter(torch.zeros(hidden_dim))
        self.W_oI = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_oI = nn.Parameter(torch.zeros(hidden_dim))
        self.v_t = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, h_tilde, c_t, h_history, deltaT_hist, deltaL_hist):
        h_hat = torch.cat([h_tilde, c_t], dim=1)
        current_proj = self.W_I(h_hat)
        scores = []
        for i, h_i in enumerate(h_history):
            I_ti = current_proj + self.W_Ih(h_i) + self.W_I1T(deltaT_hist[i]) + self.W_I1L(deltaL_hist[i]) + self.b_I
            o_ti = torch.tanh(self.W_oI(I_ti) + self.b_oI)
            score = torch.sum(o_ti * self.v_t, dim=1, keepdim=True)
            scores.append(score)
        scores_tensor = torch.cat(scores, dim=1)
        alpha = F.softmax(scores_tensor, dim=1)
        h_history_stack = torch.stack(h_history, dim=1)
        alpha_expanded = alpha.unsqueeze(-1)
        s_t = torch.sum(alpha_expanded * h_history_stack, dim=1)
        return s_t, alpha

class InteractionModuleSTGN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InteractionModuleSTGN, self).__init__()
        self.W_h_tilde = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ru = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_int = nn.Parameter(torch.zeros(hidden_dim))
    def forward(self, h_tilde, s_t, r_u):
        out = self.W_h_tilde(h_tilde) + self.W_s(s_t) + self.W_ru(r_u) + self.b_int
        h_t = torch.tanh(out)
        return h_t

class STGNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size):
        super(STGNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.cell = STGNCell(input_dim, hidden_dim)
        self.attention = SpatialTemporalAttention(hidden_dim)
        self.interaction = InteractionModuleSTGN(input_dim, hidden_dim)
        self.W_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, x_seq, deltaT_seq, deltaL_seq):
        batch_size, seq_len, _ = x_seq.shape
        h_prev = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        c_prev = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        candidate_h_list = []
        final_h_list = []
        history_states = []
        history_deltaT = []
        history_deltaL = []
        outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            deltaT_t = deltaT_seq[:, t, :]
            deltaL_t = deltaL_seq[:, t, :]
            h_tilde, c_t = self.cell(x_t, deltaT_t, deltaL_t, h_prev, c_prev)
            candidate_h_list.append(h_tilde)
            if len(history_states) == 0:
                s_t = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
            else:
                recent_states = history_states[-self.memory_size:]
                recent_deltaT = history_deltaT[-self.memory_size:]
                recent_deltaL = history_deltaL[-self.memory_size:]
                s_t, _ = self.attention(h_tilde, c_t, recent_states, recent_deltaT, recent_deltaL)
            all_candidates = torch.stack(candidate_h_list, dim=1)
            r_u = torch.tanh(self.W_r(torch.mean(all_candidates, dim=1)) + self.b_r)
            h_t = self.interaction(h_tilde, s_t, r_u)
            final_h_list.append(h_t)
            history_states.append(h_t)
            history_deltaT.append(deltaT_t)
            history_deltaL.append(deltaL_t)
            h_prev, c_prev = h_t, c_t
            logits = self.classifier(h_t)
            outputs.append(logits.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        y_pred = torch.sigmoid(outputs)
        return y_pred[:, -1, :]

# 5.5 Random Forest Expert (wrapper for a scikit-learn model)
class RandomForestExpert(nn.Module):
    def __init__(self, rf_model):
        super(RandomForestExpert, self).__init__()
        self.rf_model = rf_model  # a pre-trained scikit-learn model
    
    def forward(self, x):
        # x: (B, seq_len, input_dim) -> aggregate via average pooling over time
        x_avg = x.mean(dim=1).cpu().detach().numpy()
        # Predict probability for class 1 using the RF model
        proba = self.rf_model.predict_proba(x_avg)[:, 1]
        proba_tensor = torch.tensor(proba, dtype=torch.float32, device=x.device)
        return proba_tensor.unsqueeze(1)  # shape: (B, 1)

###############################################################
# 6. MOE MODEL DEFINITION (combining all 5 experts)
###############################################################
# Gating network that uses the average over the sequence to produce expert weights.
class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GateNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # x: (B, seq_len, input_dim) -> average over time dimension
        x_avg = torch.mean(x, dim=1)
        gate_logits = self.fc(x_avg)
        gate_weights = F.softmax(gate_logits, dim=1)
        return gate_weights

# MOE Model combining GRU, LSTM, TH-LSTM, STGN, and RF experts
class MOEModel(nn.Module):
    def __init__(self, input_size, 
                 gru_hidden_size, gru_num_layers,
                 lstm_hidden_size, lstm_num_layers,
                 thlstm_hidden_size, thlstm_memory_size,
                 stgn_hidden_size, stgn_memory_size,
                 num_experts=5):
        super(MOEModel, self).__init__()
        self.expert_gru    = FraudGRU(input_size, gru_hidden_size, gru_num_layers)
        self.expert_lstm   = FraudLSTM(input_size, lstm_hidden_size, lstm_num_layers)
        self.expert_thlstm = THLSTMModel(input_size, thlstm_hidden_size, thlstm_memory_size)
        self.expert_stgn   = STGNModel(input_size, stgn_hidden_size, stgn_memory_size)
        # RandomForestExpert uses the pre-trained rf_model (defined earlier)
        self.expert_rf     = RandomForestExpert(rf_model)
        self.gate = GateNetwork(input_size, num_experts)
    
    def forward(self, inputs):
        # inputs is a dict containing:
        # 'x': (B, seq_len, input_size)
        # 'delta': for TH-LSTM, (B, seq_len, 1)
        # 'g': for TH-LSTM, (B, seq_len, input_size)
        # 'deltaT': for STGN, (B, seq_len, 1)
        # 'deltaL': for STGN, (B, seq_len, 1)
        x = inputs['x']
        delta = inputs['delta']
        g = inputs['g']
        deltaT = inputs['deltaT']
        deltaL = inputs['deltaL']
        
        out_gru    = self.expert_gru(x)                 # (B, 1)
        out_lstm   = self.expert_lstm(x)                # (B, 1)
        out_thlstm = self.expert_thlstm(x, delta, g)      # (B, 1)
        out_stgn   = self.expert_stgn(x, deltaT, deltaL)  # (B, 1)
        out_rf     = self.expert_rf(x)                    # (B, 1)
        
        # Stack expert outputs: shape (B, 5)
        expert_outputs = torch.cat([out_gru, out_lstm, out_thlstm, out_stgn, out_rf], dim=1)
        
        # Get gating weights using x input
        gate_weights = self.gate(x)  # shape (B, 5)
        combined = torch.sum(gate_weights * expert_outputs, dim=1, keepdim=True)
        return combined

###############################################################
# 7. TRAINING & EVALUATION
###############################################################
# Hyperparameters for experts and MOE
input_size = X_train_seq.shape[2]
gru_hidden_size = 64
gru_num_layers = 2
lstm_hidden_size = 64
lstm_num_layers = 2
thlstm_hidden_size = 64
thlstm_memory_size = 3
stgn_hidden_size = 64
stgn_memory_size = 3

# Instantiate the MOE model (with 5 experts)
model = MOEModel(input_size, 
                 gru_hidden_size, gru_num_layers,
                 lstm_hidden_size, lstm_num_layers,
                 thlstm_hidden_size, thlstm_memory_size,
                 stgn_hidden_size, stgn_memory_size,
                 num_experts=5)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_inputs, labels in train_loader:
            # Move all tensors in batch_inputs to device
            for key in batch_inputs:
                batch_inputs[key] = batch_inputs[key].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            outputs = outputs.squeeze()
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
        for batch_inputs, labels in data_loader:
            for key in batch_inputs:
                batch_inputs[key] = batch_inputs[key].to(device)
            labels = labels.to(device)
            outputs = model(batch_inputs)
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

num_epochs = 1
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# Evaluate on training data
y_pred_train, y_true_train = evaluate_model(model, train_loader, device)
roc_auc_train = roc_auc_score(y_true_train, y_pred_train)
print(f'Training ROC AUC: {roc_auc_train:.4f}')

# Evaluate on test data
y_pred_test, y_true_test = evaluate_model(model, test_loader, device)
roc_auc_test = roc_auc_score(y_true_test, y_pred_test)
print(f'Test ROC AUC: {roc_auc_test:.4f}')

# Optionally, evaluate at different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("\nEvaluation at different thresholds (Test):")
for thr in thresholds:
    y_pred_bin = (y_pred_test > thr).astype(int)
    cm = confusion_matrix(y_true_test, y_pred_bin)
    TP = cm[1,1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    FP = cm[0,1] if cm.shape[1] > 1 else 0
    FN = cm[1,0] if cm.shape[0] > 1 else 0
    TN = cm[0,0]
    precision = TP/(TP+FP) if (TP+FP)>0 else 0
    recall = TP/(TP+FN) if (TP+FN)>0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print(f"Threshold: {thr:.2f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")