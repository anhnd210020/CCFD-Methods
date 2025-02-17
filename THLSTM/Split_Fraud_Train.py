#!/usr/bin/env python
import pandas as pd
import numpy as np

def split_by_transaction(input_file, train_file, test_file, train_ratio=0.8):
    """
    Splits the data based on transaction order.
    The data is sorted by the transaction time and the first `train_ratio` portion is used for training.
    """
    # Read the dataset
    df = pd.read_csv(input_file)
    # Convert the transaction time column to datetime for proper sorting
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    # Sort the dataframe by transaction time (oldest to newest)
    df_sorted = df.sort_values(by='trans_date_trans_time').reset_index(drop=True)
    # Determine the split index based on the desired training ratio
    split_index = int(train_ratio * len(df_sorted))
    # Split the dataframe into training and testing parts
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]
    # Save the splits to CSV files
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"Transaction-based split complete. Saved training data to '{train_file}' and testing data to '{test_file}'.")

def split_by_user(input_file, train_file, test_file, train_ratio=0.8, user_column='cc_num', random_state=42):
    """
    Splits the data based on unique users.
    80% of the unique users (and all of their transactions) are used for training,
    and the remaining 20% for testing.
    """
    # Read the dataset
    df = pd.read_csv(input_file)
    # Get the unique user IDs
    users = df[user_column].unique()
    # Shuffle the users for random splitting
    np.random.seed(random_state)
    np.random.shuffle(users)
    # Determine the split index for users
    split_index = int(train_ratio * len(users))
    train_users = users[:split_index]
    test_users = users[split_index:]
    # Filter the original dataframe based on user membership
    train_df = df[df[user_column].isin(train_users)]
    test_df = df[df[user_column].isin(test_users)]
    # Optionally, you can sort the splits (e.g., by transaction time)
    train_df = train_df.sort_values(by='trans_date_trans_time').reset_index(drop=True)
    test_df = test_df.sort_values(by='trans_date_trans_time').reset_index(drop=True)
    # Save the splits to CSV files
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"User-based split complete. Saved training data to '{train_file}' and testing data to '{test_file}'.")

if __name__ == "__main__":
    # Input dataset (update the path if necessary)
    input_file = "/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv"

    # --- Index 1: Transaction-based split ---
    split_by_transaction(
        input_file=input_file,
        train_file="/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/Fraud_Train/fraudTraintransactions.csv",
        test_file="/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/Fraud_Train/fraudTesttransactions.csv",
        train_ratio=0.8
    )

    # --- Index 2: User-based split (based on 'cc_num') ---
    split_by_user(
        input_file=input_file,
        train_file="/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/Fraud_Train/fraudTraincc_num.csv",
        test_file="/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/Fraud_Train/fraudTestcc_num.csv",
        train_ratio=0.8,
        user_column='cc_num',
        random_state=42
    )
