import numpy as np
import pandas as pd
import torch

# --- Preprocessing Data ---

# Load CSV and parse the datetime column
df = pd.read_csv('/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv',
                 parse_dates=['trans_date_trans_time'])

# Create a daily date column
df['date'] = df['trans_date_trans_time'].dt.date

# Function to compute spatial features for one day
def compute_spatial_features(df_day):
    spatial_features = {}
    # For each spatial level, compute aggregated features: total amount, average amount, transaction count.
    for level in ['state', 'city', 'zip', 'merchant']:
        if df_day.empty:
            spatial_features[level] = np.array([0, 0, 0])
        else:
            grouped = df_day.groupby(level).agg({
                'amt': ['sum', 'mean', 'count']
            })
            grouped.columns = ['total_amt', 'avg_amt', 'trans_count']
            if grouped.empty:
                spatial_features[level] = np.array([0, 0, 0])
            else:
                # For simplicity, choose the group with the maximum transaction count
                best = grouped.loc[grouped['trans_count'].idxmax()]
                spatial_features[level] = best.values  # shape: (3,)
    # Order: state, city, zip, merchant → shape: (4, 3)
    return np.array([
        spatial_features['state'],
        spatial_features['city'],
        spatial_features['zip'],
        spatial_features['merchant']
    ])

# Parameters for tensor dimensions
num_temporal_slices = 7    # Last 7 days per user
num_spatial_slices = 4     # state, city, zip, merchant
feature_dim = 3            # total_amt, avg_amt, trans_count

# Process each user's data
users = df['cc_num'].unique()
user_tensors = []
labels_list = []

for user in users:
    user_df = df[df['cc_num'] == user]
    dates = sorted(user_df['date'].unique())
    daily_features = []
    for d in dates:
        df_day = user_df[user_df['date'] == d]
        spatial_feat = compute_spatial_features(df_day)  # shape: (4, 3)
        daily_features.append(spatial_feat)
    daily_features = np.array(daily_features)  # shape: (num_days, 4, 3)
    
    # If fewer than 7 days, pad at the beginning with zeros
    if daily_features.shape[0] < num_temporal_slices:
        pad_shape = (num_temporal_slices - daily_features.shape[0], num_spatial_slices, feature_dim)
        padding = np.zeros(pad_shape)
        daily_features = np.concatenate([padding, daily_features], axis=0)
    else:
        daily_features = daily_features[-num_temporal_slices:, :, :]
    user_tensors.append(daily_features)
    
    # Define label for user: for example, if any transaction is fraud, label the user as fraud.
    user_label = user_df['is_fraud'].max()
    labels_list.append(user_label)

# Convert lists into tensors
input_tensor = torch.tensor(np.array(user_tensors), dtype=torch.float32)   # Shape: (num_users, 7, 4, 3)
labels_tensor = torch.tensor(np.array(labels_list), dtype=torch.float32)

print("Preprocessed input tensor shape:", input_tensor.shape)
print("Preprocessed labels tensor shape:", labels_tensor.shape)

# Save the preprocessed tensors to disk
preprocessed_data = {'input_tensor': input_tensor, 'labels_tensor': labels_tensor}
torch.save(preprocessed_data, 'preprocessed_data.pt')
print("Preprocessed data saved to 'preprocessed_data.pt'")