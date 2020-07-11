PROJECT_DIR = r"C:\Users\Olks\Desktop\churn_prediction"

train_path = join(PROJECT_DIR, "train.csv")
train_v2_path = join(PROJECT_DIR, "train_v2.csv")
user_logs_path = join(PROJECT_DIR, "user_logs.csv")
user_logs_v2_path = join(PROJECT_DIR, "user_logs_v2.csv")
sample_submission_zero_path = join(PROJECT_DIR, "sample_submission_zero.csv")
sample_submission_v2_path = join(PROJECT_DIR, "sample_submission_v2.csv")
transactions_path = join(PROJECT_DIR, "transactions.csv")
transactions_v2_path = join(PROJECT_DIR, "transactions_v2.csv")
transactions_v3_path = join(PROJECT_DIR, "transactions_v3.csv")
members_v3_path = join(PROJECT_DIR, "members_v3.csv")

# ----- Load files ----- #
# Transactional data
transactions_v3 = pd.read_csv(transactions_v3_path)

# Test set for Kaggle submission
sample_submission_v2 = pd.read_csv(sample_submission_v2_path)
