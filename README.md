# Churn prediction

This is a notebook and helper code for the Kaggle competition: 
<br>
**WSDM - KKBox's Churn Prediction Challenge
Can you predict when subscribers will churn?**
<br>
https://www.kaggle.com/c/kkbox-churn-prediction-challenge/

This competition is over but one can still upload the results and have them validated and learn!


Submissions:
1. "Basic start" [Log loss calculated by Kaggle -> 0.15289 (336th place on Public Leaderboard/ 576)]
  - Features:
      * days_from_start = Number of days from first transaction
      * transactions_num = Number or transactions
      * cancelations_num = Number of cancelations
  - Training set and validation set chosen randomly 
  - XGBoost Classification - objective= 'binary:logistic' - probabilities predicted


2. "5 features" [Log loss calculated by Kaggle -> 0.13153 (115th place on Public Leaderboard/ 576)] 
  <br>!! in best 20% !!
  - same as previously but 2 additional features added:
	* last_is_auto_renew = If last subscription is renewed automatically
	* subscription_len = Length of last subscription
	
3. "with some logs features" [Log loss calculated by Kaggle -> 0.13066 (113th place on Public Leaderboard/ 576)] 
  <br>!! in best 20% !!
  - same as previously but more additional features added:
	* num_100_sum = Number of songs listened till the end during last month
	* days_from_last_log = Number of days from the last log
    * last_month_logs_num = Number of days with log
    * total_secs_sum = Sum of total_secs
    * total_secs_mean = Mean of total_secs
    * num_unq_sum = Sum of num_unq
    * num_unq_mean = Mean of num_unq

4. "selection of transactional and logs features"
<br> [Log loss calculated by Kaggle -> 0.12343 (75th place on Public Leaderboard/ 576)] 
<br> !! best 15% !!
	* transactions_num, 
    * is_cancel, 
    * days_from_start, 
    * last_is_auto_renew, 
    * last_payment_plan_days,
    * price_diff,
    * last_plan_list_price,
    * last_actual_amount_paid,
    * last_transactions_num,
	* p_x (many features - dummies for all payment methods),
	* last_month_logs_num,
	* num_100_sum,
	* num_25_sum

5. "careful features selection"  
<br> [Log loss calculated by Kaggle -> 0.11637 (41th place on Public Leaderboard/ 576)] 
<br> !! best 8% !!
	* days_from_start, 
	* cancel_sum_4weeks,
    * last_is_auto_renew,
	* last_payment_plan_days,
	* last_month_logs_num,
	* num_100_sum.
	
