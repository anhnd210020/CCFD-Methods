import os
# Comment out the following line to enable GPU usage
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ----------------------------
# 1. Data Preprocessing
# ----------------------------
file_path = '/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv'
df = pd.read_csv(file_path)

# Preprocessing steps
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_date_trans_time_numeric'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
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

age_piv_2 = pd.pivot_table(data=df, index='cust_age_groups', columns='is_fraud', values='amt', aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

job_txn_piv_2 = pd.pivot_table(data=df, index='job', columns='is_fraud', values='amt', aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

df['merchant_num'] = pd.factorize(df['merchant'])[0]
df['last_num'] = pd.factorize(df['last'])[0]
df['street_num'] = pd.factorize(df['street'])[0]
df['city_num'] = pd.factorize(df['city'])[0]
df['zip_num'] = pd.factorize(df['zip'])[0]
df['state_num'] = pd.factorize(df['state'])[0]

df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'street', 'city', 
             'state', 'lat', 'long', 'dob', 'unix_time', 'merch_lat', 'merch_long', 'city_pop']
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# Split the dataset
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train.shape)
print("Test shape:", test.shape)

if 'trans_num' in train.columns:
    train.drop('trans_num', axis=1, inplace=True)
if 'trans_num' in test.columns:
    test.drop('trans_num', axis=1, inplace=True)

y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)

y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data:', X_train.shape, y_train.shape)
print('Shape of testing data:', X_test.shape, y_test.shape)

# ----------------------------
# 2. Build and Train the Autoencoder on Fraud Samples (Using PyTorch)
# ----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Determine the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Isolate fraud samples (X_train_fraud vẫn là DataFrame với tên cột)
X_train_fraud = X_train[y_train == 1]
print("Fraud samples shape:", X_train_fraud.shape)

# Scale the fraud data
scaler_ae = StandardScaler()
X_train_fraud_scaled = scaler_ae.fit_transform(X_train_fraud)

input_dim = X_train_fraud_scaled.shape[1]
latent_dim = 8

# Định nghĩa model Autoencoder với PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Chuyển dữ liệu fraud sang tensor và chuyển lên device
X_train_fraud_tensor = torch.from_numpy(X_train_fraud_scaled).float().to(device)

# Tạo dataset (input và target giống nhau)
dataset = TensorDataset(X_train_fraud_tensor, X_train_fraud_tensor)
dataset_size = len(dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Tạo DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Khởi tạo autoencoder, hàm loss và optimizer
model = Autoencoder(input_dim, latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Vòng lặp training
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_data, _ in train_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_data.size(0)
    epoch_train_loss = running_loss / train_size
    train_losses.append(epoch_train_loss)
    
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_data, _ in val_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            running_val_loss += loss.item() * batch_data.size(0)
    epoch_val_loss = running_val_loss / val_size
    val_losses.append(epoch_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()

# ----------------------------
# 3. Train the Random Forest Classifier for Synthetic Sample Validation
# ----------------------------
scaler_rf = StandardScaler()
X_train_scaled_rf = scaler_rf.fit_transform(X_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled_rf, y_train)

# ----------------------------
# 4. Generate Synthetic Fraud Samples Using the Autoencoder and Validate with Random Forest
# ----------------------------
num_legit = X_train[y_train == 0].shape[0]
print("Number of legitimate samples:", num_legit)

# Khởi tạo danh sách mẫu fraud tổng hợp ban đầu từ dữ liệu fraud đã scale
synthetic_fraud_list = X_train_fraud_scaled.copy()

iteration = 0
while synthetic_fraud_list.shape[0] < num_legit:
    iteration += 1
    idx = np.random.randint(0, X_train_fraud_scaled.shape[0])
    sample = X_train_fraud_scaled[idx:idx+1]
    
    # Chuyển sample sang tensor và đưa lên device
    sample_tensor = torch.from_numpy(sample).float().to(device)
    model.eval()
    with torch.no_grad():
        synthetic_sample_tensor = model(sample_tensor)
    synthetic_sample = synthetic_sample_tensor.cpu().numpy()
    
    # Chuyển numpy array thành DataFrame có cùng tên cột như dữ liệu đã fit (X_train_fraud)
    synthetic_sample_df = pd.DataFrame(synthetic_sample, columns=X_train_fraud.columns)
    synthetic_sample_original_scale = scaler_ae.inverse_transform(synthetic_sample_df)
    
    # Để sử dụng scaler_rf, chuyển kết quả trên thành DataFrame với tên cột giống X_train
    synthetic_sample_original_df = pd.DataFrame(synthetic_sample_original_scale, columns=X_train.columns)
    synthetic_sample_rf_scale = scaler_rf.transform(synthetic_sample_original_df)
    
    prediction = rf_model.predict(synthetic_sample_rf_scale)
    
    if prediction[0] == 1:
        synthetic_fraud_list = np.vstack([synthetic_fraud_list, synthetic_sample])
    
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Synthetic fraud samples count = {synthetic_fraud_list.shape[0]}")

print("Final synthetic fraud samples shape:", synthetic_fraud_list.shape)

# ----------------------------
# 5. Create a New Balanced Dataset
# ----------------------------
synthetic_fraud_original = scaler_ae.inverse_transform(
    pd.DataFrame(synthetic_fraud_list, columns=X_train_fraud.columns)
)

legit_indices = y_train[y_train == 0].index
X_train_legit = X_train.loc[legit_indices]
X_train_legit_scaled = scaler_rf.transform(X_train_legit)
X_train_legit_original = scaler_rf.inverse_transform(X_train_legit_scaled)

df_synthetic_fraud = pd.DataFrame(synthetic_fraud_original, columns=X_train.columns)
df_legit = pd.DataFrame(X_train_legit_original, columns=X_train.columns)
df_legit['is_fraud'] = 0
df_synthetic_fraud['is_fraud'] = 1

balanced_df = pd.concat([df_legit, df_synthetic_fraud], axis=0).reset_index(drop=True)
print("Balanced dataset shape:", balanced_df.shape)

balanced_df.to_csv('/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/balanced_data.csv', index=False)
print("New balanced dataset saved.")