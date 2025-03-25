# %%
# Import các thư viện cần thiết
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# %%
# Loading the train data
df_train = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv')
df_test = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTest.csv')
df = pd.concat([df_train, df_test])

# %%
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 else ('10-20' if x >= 10 and x < 20 else ('20-30' if x >= 20 and x < 30 else('30-40' if x >= 30 and x < 40 else('40-50' if x >= 40 and x < 50 else('50-60' if x >= 50 and x < 60 else('60-70' if x >= 60 and x < 70 else ('70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

# %%
drop_col = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'street', 'city', 'state', 'lat',
            'long', 'dob', 'unix_time', 'cust_age', 'merch_lat',
            'merch_long', 'city_pop']
df.drop(drop_col, axis=1, inplace=True)

# %%
age_piv_2 = pd.pivot_table(data=df,
                           index='cust_age_groups',
                           columns='is_fraud',
                           values='amt',
                           aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# %%
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# %%
job_txn_piv_2 = pd.pivot_table(data=df,
                               index='job',
                               columns='is_fraud',
                               values='amt',
                               aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

# %%
df['trans_hour'] = df['trans_hour'].astype('int')
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# %%
# Train_test_split
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
# check phân bố lớp
print(df['is_fraud'].value_counts())
print(train.shape)
print(test.shape)

# Loại bỏ cột trans_num khỏi cả train và test
train.drop('trans_num', axis=1, inplace=True)
test.drop('trans_num', axis=1, inplace=True)

# Tách dữ liệu thành dependent và independent variables
y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)

y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data: ', (X_train.shape, y_train.shape))
print('Shape of testing data: ', (X_test.shape, y_test.shape))

# %%
# Scaling dữ liệu
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Convert về DataFrame
X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

# %%
# AdaBoost Model
# Sử dụng DecisionTreeClassifier với max_depth=1 làm weak learner (decision stump)
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
ada_model.fit(X_train_sc, y_train)

# Dự đoán xác suất
y_pred_train_proba = ada_model.predict_proba(X_train_sc)
y_pred_test_proba = ada_model.predict_proba(X_test_sc)

# Tạo DataFrame cho kết quả dự đoán
y_train_results = pd.DataFrame(y_pred_train_proba, columns=['pred_not_fraud', 'pred_fraud'])
y_test_results = pd.DataFrame(y_pred_test_proba, columns=['pred_not_fraud', 'pred_fraud'])

y_train_results['y_train_actual'] = y_train.values
y_test_results['y_test_actual'] = y_test.values

# %%
# Tính toán các chỉ số cho các ngưỡng (thresholds) khác nhau
numbers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for i in numbers:
    y_train_results[i] = y_train_results.pred_fraud.map(lambda x: 1 if x > i else 0)
    y_test_results[i] = y_test_results.pred_fraud.map(lambda x: 1 if x > i else 0)

cutoff_df = pd.DataFrame(columns=['Threshold', 'precision_score', 'recall_score', 'F1_score', 'Accuracy'])

# %%
# Tính toán chỉ số trên tập train
for i in numbers:
    cm1 = confusion_matrix(y_train_results['y_train_actual'], y_train_results[i])
    TP, FP, FN, TN = cm1[1, 1], cm1[0, 1], cm1[1, 0], cm1[0, 0]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    cutoff_df.loc[i] = [i, precision, recall, f1_score_value, accuracy]

print(cutoff_df)

best_idx = cutoff_df['F1_score'].idxmax()
best_threshold = cutoff_df.loc[best_idx, 'Threshold']
best_precision = cutoff_df.loc[best_idx, 'precision_score']
best_recall = cutoff_df.loc[best_idx, 'recall_score']
best_f1_score = cutoff_df.loc[best_idx, 'F1_score']
best_accuracy = cutoff_df.loc[best_idx, 'Accuracy']
best_auc = roc_auc_score(y_train_results['y_train_actual'], y_train_results.pred_fraud)  # ROC AUC thường dùng toàn bộ dự đoán xác suất

print(f'Best Threshold (Train): {best_threshold:.4f}')
print(f'Best Accuracy (Train): {best_accuracy:.4f}')
print(f'Best Precision (Train): {best_precision:.4f}')
print(f'Best Recall (Train): {best_recall:.4f}')
print(f'Best F1 Score (Train): {best_f1_score:.4f}')
print(f'Best ROC_AUC Score (Train): {best_auc:.4f}')

# %%
# Tính toán chỉ số trên tập test
cutoff_df = pd.DataFrame(columns=['Threshold', 'precision_score', 'recall_score', 'F1_score', 'Accuracy'])
for i in numbers:
    cm1 = confusion_matrix(y_test_results['y_test_actual'], y_test_results[i])
    TP, FP, FN, TN = cm1[1, 1], cm1[0, 1], cm1[1, 0], cm1[0, 0]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    cutoff_df.loc[i] = [i, precision, recall, f1_score_value, accuracy]

print(cutoff_df)

best_idx = cutoff_df['F1_score'].idxmax()
best_threshold = cutoff_df.loc[best_idx, 'Threshold']
best_precision = cutoff_df.loc[best_idx, 'precision_score']
best_recall = cutoff_df.loc[best_idx, 'recall_score']
best_f1_score = cutoff_df.loc[best_idx, 'F1_score']
best_accuracy = cutoff_df.loc[best_idx, 'Accuracy']
best_auc = roc_auc_score(y_test_results['y_test_actual'], y_test_results.pred_fraud)

print(f'Best Threshold (Test): {best_threshold:.4f}')
print(f'Best Accuracy (Test): {best_accuracy:.4f}')
print(f'Best Precision (Test): {best_precision:.4f}')
print(f'Best Recall (Test): {best_recall:.4f}')
print(f'Best F1 Score (Test): {best_f1_score:.4f}')
print(f'Best ROC_AUC Score (Test): {best_auc:.4f}')