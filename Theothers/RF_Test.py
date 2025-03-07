# %% Import các thư viện cần thiết
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# %% Đọc dữ liệu
df_train = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv')
df_test = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTest.csv')
df = pd.concat([df_train, df_test])

# %% Xử lý thời gian và tuổi khách hàng
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])
df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 
                                               else ('10-20' if x >= 10 and x < 20 
                                               else ('20-30' if x >= 20 and x < 30 
                                               else ('30-40' if x >= 30 and x < 40 
                                               else ('40-50' if x >= 40 and x < 50 
                                               else ('50-60' if x >= 50 and x < 60 
                                               else ('60-70' if x >= 60 and x < 70 
                                               else ('70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

# %% Loại bỏ các cột không cần thiết
drop_cols = ['Unnamed: 0', 'merchant', 'first', 'last', 'street', 'city', 'state', 
             'lat', 'long', 'dob', 'unix_time', 'cust_age', 'merch_lat', 'merch_long', 'city_pop']
df.drop(drop_cols, axis=1, inplace=True)

# %% Xử lý các biến phân loại với pivot table
# Đối với cust_age_groups:
age_piv_2 = pd.pivot_table(data=df, index='cust_age_groups', columns='is_fraud', values='amt', aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Đối với category:
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# Đối với job:
job_txn_piv_2 = pd.pivot_table(data=df, index='job', columns='is_fraud', values='amt', aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

# %% Chuyển đổi kiểu dữ liệu và mã hóa biến
df['trans_hour'] = df['trans_hour'].astype('int')
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# Chuyển đổi trans_date_trans_time thành timestamp (số)
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

# %% Tách dữ liệu thành train và test (stratify theo is_fraud)
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Loại bỏ cột trans_num khỏi cả train và test
train.drop('trans_num', axis=1, inplace=True)
test.drop('trans_num', axis=1, inplace=True)

# Tách nhãn và features
y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)
y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

# %% Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Chuyển đổi về DataFrame để giữ tên cột
X_train_sc = pd.DataFrame(X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(X_test_sc, columns=X_test.columns)

# %% Hàm tạo sequence (nhóm theo cc_num)
sequence_length = 10  # Số giao dịch trong 1 sequence

def create_sequences_predict_all(df, sequence_length):
    sequences, labels = [], []
    # Nhóm theo cc_num
    grouped = df.groupby('cc_num')
    for user_id, group in grouped:
        # Sắp xếp theo trans_date_trans_time (đã chuyển thành timestamp)
        group = group.sort_values(by='trans_date_trans_time')
        # Lấy các giá trị (loại bỏ 'is_fraud' và 'cc_num')
        values = group.drop(columns=['is_fraud', 'cc_num']).values
        targets = group['is_fraud'].values
        n = len(group)
        for i in range(n):
            if i < sequence_length:
                # Nếu số giao dịch hiện có nhỏ hơn sequence_length: pad các giá trị ban đầu
                pad_needed = sequence_length - (i + 1)
                pad = np.repeat(values[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, values[:i+1]), axis=0)
            else:
                # Nếu đủ giao dịch, lấy sequence gồm các giao dịch từ (i-sequence_length+1) đến i
                seq = values[i-sequence_length+1:i+1]
            sequences.append(seq)
            labels.append(targets[i])
    return np.array(sequences), np.array(labels)

# %% Gộp thêm cột 'is_fraud' vào DataFrame chuẩn hóa để tạo sequence
train_seq_df = X_train_sc.copy()
train_seq_df['is_fraud'] = y_train.values
test_seq_df = X_test_sc.copy()
test_seq_df['is_fraud'] = y_test.values

# Tạo sequence cho train và test
X_train_seq, y_train_seq = create_sequences_predict_all(train_seq_df, sequence_length)
X_test_seq, y_test_seq = create_sequences_predict_all(test_seq_df, sequence_length)

print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

# %% Phẳng hóa các sequence thành vector 1 chiều
num_train_sequences = X_train_seq.shape[0]
num_test_sequences = X_test_seq.shape[0]
X_train_flat = X_train_seq.reshape(num_train_sequences, -1)
X_test_flat = X_test_seq.reshape(num_test_sequences, -1)

# %% Huấn luyện mô hình Random Forest với dữ liệu đã phẳng
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
rf_model.fit(X_train_flat, y_train_seq)

# %% Dự đoán xác suất
y_train_pred_proba = rf_model.predict_proba(X_train_flat)[:, 1]
y_test_pred_proba = rf_model.predict_proba(X_test_flat)[:, 1]

# Lưu các kết quả xác suất vào DataFrame để đánh giá
y_train_results = pd.DataFrame({
    'pred_fraud': y_train_pred_proba,
    'y_train_actual': y_train_seq
})
y_test_results = pd.DataFrame({
    'pred_fraud': y_test_pred_proba,
    'y_test_actual': y_test_seq
})

# %% Tính chỉ số theo ngưỡng: tạo DataFrame riêng cho train và test
numbers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

cutoff_df_train = pd.DataFrame(columns=['Threshold', 'Accuracy', 'precision_score', 'recall_score', 'F1_score'])
for thresh in numbers:
    preds = y_train_results['pred_fraud'].map(lambda x: 1 if x > thresh else 0)
    cm = confusion_matrix(y_train_results['y_train_actual'], preds)
    TP, FP, FN, TN = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
    precision_val = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_val = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_val = (2 * precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    accuracy_val = (TP + TN) / (TP + TN + FP + FN)
    cutoff_df_train.loc[thresh] = [thresh, accuracy_val, precision_val, recall_val, f1_val]

print("Train Evaluation:")
print(cutoff_df_train)

cutoff_df_test = pd.DataFrame(columns=['Threshold', 'Accuracy', 'precision_score', 'recall_score', 'F1_score'])
for thresh in numbers:
    preds = y_test_results['pred_fraud'].map(lambda x: 1 if x > thresh else 0)
    cm = confusion_matrix(y_test_results['y_test_actual'], preds)
    TP, FP, FN, TN = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
    precision_val = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_val = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_val = (2 * precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    accuracy_val = (TP + TN) / (TP + TN + FP + FN)
    cutoff_df_test.loc[thresh] = [thresh, accuracy_val, precision_val, recall_val, f1_val]

print("Test Evaluation:")
print(cutoff_df_test)

# %% Chọn ngưỡng tốt nhất dựa trên F1 Score cho tập Train
best_thresh_train = cutoff_df_train['F1_score'].idxmax()
best_accuracy_train = cutoff_df_train.loc[best_thresh_train, 'Accuracy']
best_precision_train = cutoff_df_train.loc[best_thresh_train, 'precision_score']
best_recall_train = cutoff_df_train.loc[best_thresh_train, 'recall_score']
best_f1_train = cutoff_df_train.loc[best_thresh_train, 'F1_score']
best_auc_train = roc_auc_score(y_train_results['y_train_actual'], y_train_results['pred_fraud'])

print(f'\nTrain Best Threshold: {best_thresh_train:.4f}')
print(f'Train Best Accuracy: {best_accuracy_train:.4f}')
print(f'Train Best Precision: {best_precision_train:.4f}')
print(f'Train Best Recall: {best_recall_train:.4f}')
print(f'Train Best F1 Score: {best_f1_train:.4f}')
print(f'Train Best ROC_AUC Score: {best_auc_train:.4f}')

# %% Chọn ngưỡng tốt nhất dựa trên F1 Score cho tập Test
best_thresh_test = cutoff_df_test['F1_score'].idxmax()
best_accuracy_test = cutoff_df_test.loc[best_thresh_test, 'Accuracy']
best_precision_test = cutoff_df_test.loc[best_thresh_test, 'precision_score']
best_recall_test = cutoff_df_test.loc[best_thresh_test, 'recall_score']
best_f1_test = cutoff_df_test.loc[best_thresh_test, 'F1_score']
best_auc_test = roc_auc_score(y_test_results['y_test_actual'], y_test_results['pred_fraud'])

print(f'\nTest Best Threshold: {best_thresh_test:.4f}')
print(f'Test Best Accuracy: {best_accuracy_test:.4f}')
print(f'Test Best Precision: {best_precision_test:.4f}')
print(f'Test Best Recall: {best_recall_test:.4f}')
print(f'Test Best F1 Score: {best_f1_test:.4f}')
print(f'Test Best ROC_AUC Score: {best_auc_test:.4f}')