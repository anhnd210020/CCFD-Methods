import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

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

# ------------------ Train-test Split ------------------
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

# Loại bỏ cột dạng text không dùng được trong StandardScaler (ví dụ: delta_t_category)
X_train = X_train.drop(columns=['delta_t_category'], errors='ignore')
X_test = X_test.drop(columns=['delta_t_category'], errors='ignore')

print('Shape of training data: ', (X_train.shape, y_train.shape))
print('Shape of testing data: ', (X_test.shape, y_test.shape))

# --- Kiểm tra các features được sử dụng ---
print("\nCác features được đưa vào học (trong X_train):")
print(X_train.columns.tolist())

print("\nTất cả các features hiện có trong DataFrame sau xử lý:")
print(df.columns.tolist())

# ------------------ Scaling ------------------
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

# ------------------ Huấn luyện Random Forest ------------------
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_sc, y_train)

# Dự đoán xác suất cho tập train và test
y_pred_train_proba_rf = rf_model.predict_proba(X_train_sc)
y_pred_test_proba_rf = rf_model.predict_proba(X_test_sc)

# Lưu kết quả dự đoán vào DataFrame
y_train_results_rf = pd.DataFrame(y_pred_train_proba_rf, columns=['pred_not_fraud', 'pred_fraud'])
y_test_results_rf = pd.DataFrame(y_pred_test_proba_rf, columns=['pred_not_fraud', 'pred_fraud'])
y_train_results_rf['y_train_actual'] = y_train.values
y_test_results_rf['y_test_actual'] = y_test.values

# ------------------ Tính Feature Importance ------------------
importances = rf_model.feature_importances_
feature_names = X_train_sc.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df['abs_importance'] = feature_importance_df['Importance'].abs()
feature_importance_df.sort_values(by='abs_importance', ascending=False, inplace=True)

print("\nFeature Importances từ mô hình Random Forest:")
print(feature_importance_df[['Feature', 'Importance']])

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

# ------------------ Tính các chỉ số hiệu năng cho tập train ------------------
numbers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cutoff_train_df_rf = pd.DataFrame(columns=['Threshold', 'Accuracy', 'precision_score', 'recall_score', 'F1_score'])

for thresh in numbers:
    # Dự đoán nhị phân theo threshold
    y_pred_bin = y_train_results_rf['pred_fraud'].map(lambda x: 1 if x > thresh else 0)
    cm = confusion_matrix(y_train_results_rf['y_train_actual'], y_pred_bin)
    TP = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    FP = cm[0, 1] if cm.shape[1] > 1 else 0
    FN = cm[1, 0] if cm.shape[0] > 1 else 0
    TN = cm[0, 0]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    cutoff_train_df_rf.loc[thresh] = [thresh, accuracy, precision, recall, f1_score_value]

print("\n--- Random Forest Train Set Evaluation ---")
print(cutoff_train_df_rf)

best_idx_train_rf = cutoff_train_df_rf['F1_score'].idxmax()
best_threshold_train_rf = cutoff_train_df_rf.loc[best_idx_train_rf, 'Threshold']
best_auc_train_rf = roc_auc_score(y_train_results_rf['y_train_actual'], y_train_results_rf['pred_fraud'])
print(f'Random Forest Train - Best Threshold: {best_threshold_train_rf:.4f}')
print(f'Random Forest Train - Best ROC_AUC Score: {best_auc_train_rf:.4f}')

# ------------------ Tính các chỉ số hiệu năng cho tập test ------------------
cutoff_test_df_rf = pd.DataFrame(columns=['Threshold', 'Accuracy', 'precision_score', 'recall_score', 'F1_score'])

for thresh in numbers:
    y_pred_bin = y_test_results_rf['pred_fraud'].map(lambda x: 1 if x > thresh else 0)
    cm = confusion_matrix(y_test_results_rf['y_test_actual'], y_pred_bin)
    TP = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    FP = cm[0, 1] if cm.shape[1] > 1 else 0
    FN = cm[1, 0] if cm.shape[0] > 1 else 0
    TN = cm[0, 0]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    cutoff_test_df_rf.loc[thresh] = [thresh, accuracy, precision, recall, f1_score_value]

print("\n--- Random Forest Test Set Evaluation ---")
print(cutoff_test_df_rf)

best_idx_test_rf = cutoff_test_df_rf['F1_score'].idxmax()
best_threshold_test_rf = cutoff_test_df_rf.loc[best_idx_test_rf, 'Threshold']
best_accuracy_test_rf = cutoff_test_df_rf.loc[best_idx_test_rf, 'Accuracy']
best_precision_test_rf = cutoff_test_df_rf.loc[best_idx_test_rf, 'precision_score']
best_recall_test_rf = cutoff_test_df_rf.loc[best_idx_test_rf, 'recall_score']
best_f1_score_test_rf = cutoff_test_df_rf.loc[best_idx_test_rf, 'F1_score']
best_auc_test_rf = roc_auc_score(y_test_results_rf['y_test_actual'], y_test_results_rf['pred_fraud'])
print(f'Random Forest Test - Best Threshold: {best_threshold_test_rf:.4f}')
print(f'Random Forest Test - Best Accuracy: {best_accuracy_test_rf:.4f}')
print(f'Random Forest Test - Best Precision: {best_precision_test_rf:.4f}')
print(f'Random Forest Test - Best Recall: {best_recall_test_rf:.4f}')
print(f'Random Forest Test - Best F1 Score: {best_f1_score_test_rf:.4f}')
print(f'Random Forest Test - Best ROC_AUC Score: {best_auc_test_rf:.4f}')

# ------------------ Vẽ ROC Curve cho tập test ------------------
fpr, tpr, _ = roc_curve(y_test_results_rf['y_test_actual'], y_test_results_rf['pred_fraud'])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest ROC Curve (AUC = {best_auc_test_rf:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest (Test Set)')
plt.legend()
plt.grid(True)
plt.show()