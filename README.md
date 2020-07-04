# Churn prediction

This is a notebook and helper code for the Kaggle competition: 
<br>
**WSDM - KKBox's Churn Prediction Challenge
Can you predict when subscribers will churn?**
<br>
https://www.kaggle.com/c/kkbox-churn-prediction-challenge/

Submissions:
1. "Basic start" [Log loss calculated by Kaggle -> 0.15289 (336 place on Public Leaderboard/ 576)]
  - Features:
      * days_from_start = Number of days from first transaction
      * transactions_num = Number or transactions
      * calnceletions_num = Number of cancelations
  - Training set and validation set chosen randomly 
  - XGBoost Classification - objective= 'binary:logistic' - probabilities predicted


2. ...

