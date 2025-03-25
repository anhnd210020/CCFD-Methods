# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Set a default seaborn style for consistency
sns.set(style="whitegrid")


def load_data(train_path: str, test_path: str) -> pd.DataFrame:
    """
    Load and concatenate training and test datasets.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test])
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe by converting dates, 
    extracting day names, hour and calculating age.
    """
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()
    df['hour'] = df['trans_date_trans_time'].dt.hour
    # Calculate age as of 2025; adjust if needed
    df['age'] = 2025 - pd.to_datetime(df['dob']).dt.year
    return df


def transaction_analysis(df: pd.DataFrame) -> None:
    """
    Analyze the number of transactions per credit card number.
    """
    trans_count = df.groupby('cc_num').size()
    max_trans = trans_count.max()
    min_trans = trans_count.min()
    max_cc = trans_count.idxmax()
    min_cc = trans_count.idxmin()

    print("Max transactions:", max_trans, "for cc_num:", max_cc)
    print("Min transactions:", min_trans, "for cc_num:", min_cc)


def fraud_analysis(df: pd.DataFrame) -> None:
    """
    Analyze fraudulent transactions by day of week and hour,
    and calculate fraud-related percentages.
    """
    fraud_df = df[df['is_fraud'] == 1]
    
    # Fraud analysis by day
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fraud_day_count = fraud_df['day_of_week'].value_counts().reindex(days_order)
    
    # Fraud analysis by hour
    fraud_hour_count = fraud_df['hour'].value_counts().sort_index()
    
    print("Fraud transactions by day:\n", fraud_day_count)
    print("\nFraud transactions by hour:\n", fraud_hour_count)
    
    # Plot fraud transactions by day
    plt.figure(figsize=(10, 4))
    sns.barplot(x=fraud_day_count.index, y=fraud_day_count.values, palette="viridis")
    plt.title('Fraud Transactions by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Fraud Transactions')
    plt.tight_layout()
    plt.show()

    # Plot fraud transactions by hour
    plt.figure(figsize=(10, 4))
    sns.barplot(x=fraud_hour_count.index, y=fraud_hour_count.values, palette="magma")
    plt.title('Fraud Transactions by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Fraud Transactions')
    plt.tight_layout()
    plt.show()
    
    # Fraud percentages
    fraudulent_cc = fraud_df['cc_num'].nunique()
    total_cc = df['cc_num'].nunique()
    fraud_cc_percent = (fraudulent_cc / total_cc) * 100
    
    fraud_transactions = fraud_df.shape[0]
    total_transactions = df.shape[0]
    fraud_transactions_percent = (fraud_transactions / total_transactions) * 100
    
    print("Unique cc_num with fraud transactions: {} ({:.2f}% of all cc_num)".format(
        fraudulent_cc, fraud_cc_percent))
    print("Number of fraud transactions: {} ({:.2f}% of all transactions)".format(
        fraud_transactions, fraud_transactions_percent))


def overall_fraud_distribution(df: pd.DataFrame) -> None:
    """
    Plot the overall distribution of fraud versus legitimate transactions.
    """
    fraud_counts = df['is_fraud'].value_counts(normalize=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette='coolwarm')
    plt.xticks([0, 1], ['Legitimate', 'Fraud'])
    plt.xlabel("Transaction Type")
    plt.ylabel("Proportion")
    plt.title("Transaction Fraud Ratio")
    plt.tight_layout()
    plt.show()


def top_fraudulent_users(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Display and plot the top fraudulent users by transaction count.
    """
    fraud_users = df[df['is_fraud'] == 1]['cc_num'].value_counts().head(top_n)
    plt.figure(figsize=(10, 5))
    fraud_users.plot(kind='bar', color='red')
    plt.xlabel("User ID (cc_num)")
    plt.ylabel("Number of Fraudulent Transactions")
    plt.title(f"Top {top_n} Fraudulent Users")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    total_fraud_users = df[df['is_fraud'] == 1]['cc_num'].nunique()
    print("Total users with at least one fraud transaction:", total_fraud_users)


def boxplot_transaction_amount(df: pd.DataFrame) -> None:
    """
    Create a boxplot comparing transaction amounts for fraudulent and legitimate transactions.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='is_fraud', y='amt', data=df, palette="coolwarm")
    plt.xlabel("Transaction Type (0: Legitimate, 1: Fraud)")
    plt.ylabel("Transaction Amount")
    plt.title("Transaction Amount Distribution")
    plt.ylim(0, 5000)  # Adjust the limit to reduce the influence of outliers
    plt.tight_layout()
    plt.show()


def fraud_by_merchant_and_category(df: pd.DataFrame) -> None:
    """
    Analyze fraudulent transactions by merchant and transaction category.
    """
    fraud_df = df[df['is_fraud'] == 1]
    
    # Fraud by merchant
    fraud_by_merchant = fraud_df['merchant'].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    fraud_by_merchant.plot(kind='bar', color='red')
    plt.xlabel("Merchant")
    plt.ylabel("Fraudulent Transaction Count")
    plt.title("Top 10 Merchants by Fraudulent Transactions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Fraud by category
    fraud_by_category = fraud_df['category'].value_counts()
    plt.figure(figsize=(10, 5))
    fraud_by_category.plot(kind='bar', color='red')
    plt.xlabel("Category")
    plt.ylabel("Fraudulent Transaction Count")
    plt.title("Fraud Distribution by Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def fraud_by_state(df: pd.DataFrame) -> None:
    """
    Plot the top 10 states with the highest number of fraudulent transactions.
    """
    fraud_state = df[df['is_fraud'] == 1]['state'].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    fraud_state.plot(kind='bar', color='red')
    plt.xlabel("State")
    plt.ylabel("Fraudulent Transaction Count")
    plt.title("Top 10 States by Fraudulent Transactions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def age_distribution(df: pd.DataFrame) -> None:
    """
    Plot the age distribution for fraudulent and legitimate transactions.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df[df['is_fraud'] == 1]['age'], bins=20, kde=True, color='red', label="Fraud")
    sns.histplot(df[df['is_fraud'] == 0]['age'], bins=20, kde=True, color='blue', label="Legitimate")
    plt.legend()
    plt.xlabel("Age")
    plt.ylabel("Number of Transactions")
    plt.title("Age Distribution for Fraudulent vs. Legitimate Transactions")
    plt.tight_layout()
    plt.show()


def fraud_map_visualization(df: pd.DataFrame) -> folium.Map:
    """
    Create a folium map marking the locations of fraudulent transactions.
    """
    fraud_df = df[df['is_fraud'] == 1]
    center_lat = df['merch_lat'].mean()
    center_long = df['merch_long'].mean()
    fraud_map = folium.Map(location=[center_lat, center_long], zoom_start=5)
    
    fraud_locations = fraud_df[['merch_lat', 'merch_long']].dropna()
    for _, row in fraud_locations.iterrows():
        folium.CircleMarker(
            location=[row['merch_lat'], row['merch_long']],
            radius=3,
            color="red",
            fill=True,
            fill_color="red"
        ).add_to(fraud_map)
    return fraud_map


def main() -> folium.Map:
    # Define file paths
    train_path = '/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTrain.csv'
    test_path = '/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/fraudTest.csv'
    
    # Load and preprocess data
    df = load_data(train_path, test_path)
    df = preprocess_data(df)
    
    # Run analyses
    transaction_analysis(df)
    fraud_analysis(df)
    overall_fraud_distribution(df)
    top_fraudulent_users(df)
    boxplot_transaction_amount(df)
    fraud_by_merchant_and_category(df)
    fraud_by_state(df)
    age_distribution(df)
    
    # Save fraudulent transactions to CSV
    df[df['is_fraud'] == 1].to_csv(
        '/home/ducanh/Credit Card Transactions Fraud Detection/EDA/fraud_transactions.csv', index=False)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Columns with missing values:\n", missing_values[missing_values > 0])
    
    # Generate fraud map visualization
    fraud_map = fraud_map_visualization(df)
    
    # Return the map object so it can be displayed in a notebook
    return fraud_map


if __name__ == '__main__':
    fraud_map = main()
    # In a Jupyter notebook, display the map by evaluating `fraud_map`