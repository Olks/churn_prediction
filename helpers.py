import random
import pandas as pd
import xgboost as xgb
from os.path import join

import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt


# ------- helper methods ------ #
def get_labels(df, cutoff_day):
    """Calculates lables for the month before cutoff_day.

    Parameters:
    df (pandas.DataFrame): Transactions table
    cutoff_daye (str): First day of the month in format %Y%m%d e.g. 20170301

    Returns:
    pandas.DataFrame: Two columns data frame; columns: "msno", "is_churn"
    """

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format='%Y%m%d')
    df["membership_expire_date"] = pd.to_datetime(df["membership_expire_date"], format='%Y%m%d')
    
    cutoff_day = pd.to_datetime(cutoff_day, format='%Y%m%d') 
    end = cutoff_day - pd.DateOffset(1)
    start = (end - pd.offsets.MonthBegin(1)).floor('d')
    
    trans = df.loc[df.is_cancel == 0].drop(columns=["is_cancel"])
    
    potenctial_churn = trans.loc[(trans.membership_expire_date <= end) & 
                                 (trans.membership_expire_date >= start)]
    last_trans = trans.groupby("msno")["transaction_date"].max().reset_index()
    potential_merged = potenctial_churn.merge(last_trans, on="msno")
    potential_merged
    
    potential_merged["is_churn"] = potential_merged.apply(lambda row: 0 
                            if row["transaction_date_y"] > row["transaction_date_x"] 
                            else 1, axis=1)
    return  potential_merged[["msno", "is_churn"]]


def calculate_last_subsciption_features(df, cutoff_day):
    """Calculates features based on tranactonal data.

    Parameters:
    df (pandas.DataFrame): Transactions table
    cutoff_daye (str): First day of the month in format %Y%m%d e.g. 20170301

    Returns:
    pandas.DataFrame: Tables with features; columns: "msno", ...
    """
    
     # change date format from int to pandas datetime
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format='%Y%m%d')
    df["membership_expire_date"] = pd.to_datetime(df["membership_expire_date"], format='%Y%m%d')
    
     # set cutoff day and filter data
    train_cutoff_day = pd.to_datetime(cutoff_day, format='%Y%m%d') 
    df = df.loc[df["transaction_date"] < train_cutoff_day]
    
    # filter out cancelations
    df = df.loc[df.is_cancel == 0].drop(columns=["is_cancel"])
    
    # change date format from int to pandas datetime
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format='%Y%m%d')
    df["membership_expire_date"] = pd.to_datetime(df["membership_expire_date"], format='%Y%m%d')
    
    # find which transaction is the last one and merge with transactional data
    last_transaction = df.groupby("msno")["transaction_date"].max().reset_index()
    
    # here we get duplicates
    last_subscription = df.merge(last_transaction)
    
    columns = ["msno", 
               "payment_method_id",
               "payment_plan_days",
               "plan_list_price",
               "actual_amount_paid",
               "is_auto_renew",
               "transaction_date",
               "membership_expire_date"]
    
    final_table = last_subscription[columns].rename(columns={
                                                 "is_auto_renew": "last_is_auto_renew",
                                                 "payment_method_id": "last_payment_method_id",
                                                 "payment_plan_days": "last_payment_plan_days",
                                                 "plan_list_price": "last_plan_list_price",
                                                 "actual_amount_paid": "last_actual_amount_paid",
                                                 "membership_expire_date": "last_membership_expire_date"
                                                            })
    
    last_trans_num = final_table.groupby("msno").transaction_date.count().reset_index().rename(columns=
                                                            {"transaction_date": "last_transactions_num"})
    
    final_table = final_table.drop_duplicates(["msno", "transaction_date"])
    
    final_table = final_table.merge(last_trans_num, on="msno")
    
    # get dummies for  paymenet_method_id
    payment_dummies = pd.get_dummies(final_table["last_payment_method_id"])
    column_names = [f"p_{id}" for id in payment_dummies.columns]
    final_table[column_names] = payment_dummies
    
    return final_table


def calculate_transactional_features(df, cutoff_day):
    """Calculates features based on tranactonal data.

    Parameters:
    df (pandas.DataFrame): Transactions table
    cutoff_daye (str): First day of the month in format %Y%m%d e.g. 20170301

    Returns:
    pandas.DataFrame: Tables with features; columns: "msno", ...
    """
    # change date format from int to pandas datetime
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format='%Y%m%d')
    df["membership_expire_date"] = pd.to_datetime(df["membership_expire_date"], format='%Y%m%d')
    
    # set cutoff day and filter data
    train_cutoff_day = pd.to_datetime(cutoff_day, format='%Y%m%d') 
    df = df.loc[df["transaction_date"] < train_cutoff_day]
    
    four_weeks_ago = train_cutoff_day - pd.DateOffset(28)
    two_weeks_ago = train_cutoff_day - pd.DateOffset(14)
    
     # lets filter last 4 weeks data 
    last_4_weeks = df.loc[df["transaction_date"] >= four_weeks_ago]
    cancelations_4weeks = last_4_weeks.groupby("msno").agg({
                                "is_cancel": sum}).reset_index().rename(columns={"is_cancel": 
                                                                                 "cancel_sum_4weeks"})
        
    total_sums = df.groupby("msno").agg({
                        "is_cancel": sum,
                        "actual_amount_paid": sum
                                  }).reset_index().rename(columns={"is_cancel": "cancel_sum",
                                                                   "actual_amount_paid": 
                                                                   "actual_amount_paid_sum"
                                                                  })
    
    # filter out cancelations
    df = df.loc[df.is_cancel == 0]
    
    # find the fist transaction per user
    users_trans_features = df.groupby("msno").agg({"transaction_date": 
                                                   ["min", "count"]})
    
    # flatten columns' names
    users_trans_features.columns = [a + "_" + b for a,b in users_trans_features.columns]
    
    users_trans_features = users_trans_features.rename(columns={
                                                "transaction_date_count": "transactions_num"}).reset_index()
    
    # change date format from int to pandas datetime
    users_trans_features["first_transaction"] = pd.to_datetime(users_trans_features.transaction_date_min, 
                                                                  format='%Y%m%d')
    
    # calculate number of days passed from the first user transaction
    users_trans_features["days_from_start"] = (train_cutoff_day - 
                                               users_trans_features["first_transaction"]).dt.days
    
    users_trans_features = users_trans_features.merge(total_sums, on="msno", how="left").fillna(0)
    users_trans_features = users_trans_features.merge(cancelations_4weeks, on="msno", how="left").fillna(0)
    
    return users_trans_features


def calculate_logs_features(logs_data, cutoff_day):
    """Calculates features based on user logging data.

    Parameters:
    logs_data (pandas.DataFrame): User logs data table
    cutoff_daye (str): First day of the month in format %Y%m%d e.g. 20170301

    Returns:
    pandas.DataFrame: Tables with features; columns: "msno", ...
    """
    # Set cutoff day
    train_cutoff_day = pd.to_datetime(cutoff_day, format='%Y%m%d') 
    
    # Change format to date 
    logs_data["date"] = pd.to_datetime(logs_data["date"], format='%Y%m%d')
    four_weeks_ago = train_cutoff_day - pd.DateOffset(28)
    two_weeks_ago = train_cutoff_day - pd.DateOffset(14)
    
    # lets filter last 4 weeks data and 2 weeks data
    last_4_weeks = logs_data.loc[logs_data.date >= four_weeks_ago]
    last_2_weeks = logs_data.loc[logs_data.date >= two_weeks_ago]
    
    last_2_weeks_stats = last_2_weeks.groupby("msno").agg({
                                        "date": ["max", "count"],
                                        "total_secs": ["sum", "mean"],
                                        "num_unq": ["sum", "mean"],
                                        "num_25": ["sum", "mean"],
                                        "num_50": ["sum", "mean"],
                                        "num_75": ["sum", "mean"],
                                        "num_985": ["sum", "mean"],
                                        "num_100": ["sum", "mean"],
                                              })
    last_2_weeks_stats.columns = [a + "_" + b + "_2weeks" for a,b in last_2_weeks_stats.columns]
    
    last_user_log = last_4_weeks.groupby("msno").agg({
                                        "date": ["max", "count"],
                                        "total_secs": ["sum", "mean"],
                                        "num_unq": ["sum", "mean"],
                                        "num_25": ["sum", "mean"],
                                        "num_50": ["sum", "mean"],
                                        "num_75": ["sum", "mean"],
                                        "num_985": ["sum", "mean"],
                                        "num_100": ["sum", "mean"],
                                              })

    # flatten columns' names
    last_user_log.columns = [a + "_" + b for a,b in last_user_log.columns]
    last_user_log["days_from_last_log"] =  (train_cutoff_day - last_user_log["date_max"]).dt.days
    final_table = last_user_log.rename(columns={"date_count": "last_month_logs_num"}).reset_index()
    
    # merge 4 weeks stats with 2 weeks stats
    final_table = final_table.merge(last_2_weeks_stats.reset_index(), on="msno", how="left")
    final_table[last_2_weeks_stats.columns] = final_table[last_2_weeks_stats.columns].fillna(0)
    return final_table
