# 1. Introduction
In the lending industry, debtors can apply for loans from lenders, and investors can buy the loans from lenders in trade for the promise of compensation with interest. The investors will make income from the interest if the borrower repays the loan. However, failure to repay the loan will cause losses to lenders and investors. Therefore, lenders face the hassle of predicting the threat of a borrower being unable to repay a loan. In this project, the dataset from Lending Club is used to train Machine Learning models to understand the applicant's profile, to decide if the borrower has the ability to repay the loan, and to minimize the risk of future loan defaults. 

# 2. Dataset
The dataset is download from Kaggle, it covers all approved loan information and loan status (whether default or not) from 2007 to the third quarter of 2020. 
https://www.kaggle.com/ethon0426/lending-club-20072020q1/discussion/207082

# 3. Feature Selection
The full dataset has 2.9 million loan records and 151 features for each loan. To select appropriate predictors, firstly I chose features that would be available to an investor before deciding to fund the loan. Then checked the correlation between other features and the target value 'loan_status'. After that, we got 21 selected features: 
'loan_amnt'
'term'
'int_rate'
'sub_grade'
'home_ownership'
'annual_inc'
'verification_status'
'purpose'
'addr_state'
'dti'
'open_acc'
'pub_rec'
'revol_bal'
'revol_util'
'initial_list_status'
'application_type'
'mort_acc'
'pub_rec_bankruptcies'
'loan_status_flag'
'fico'
'earliest_cr_line_y'
 The dataset is filtered without duplicates.

# 4. Prediction
While ‘loan_status_flag’ was set as the dependent variable, the other 21 selected features were set as the independent variable. The positive label is “charge off”, and the negative label is “fully paid”. The ratio between the number of positive labels and negative labels is about 1:4. The imbalance may cause the model to favor the negative labels. I applied downsampling on the negative samples to make them have the same size as the positive samples. Meanwhile, I set the proportion of the training group to 20% and fixed the split data by setting the ‘random_state’ parameter.

After splitting the data set, we tried to train the data with different methods:
  1. Logistic Regression: which is used the classification problems with discrete output;
  2. Random Forest Classifier: which utilizes ensemble learning and consists of many decision trees, could get higher accuracy for a huge sample dataset;
  3. XGBoost: which is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework.
  
The XGBoost models yield better performance, with accuracy equal to 66%, on our dataset in most of our initial experiments. We decided to use the XGBoost model and further tune it to achieve better performance. The hyperparameter tuning used randomized grid search based on the 3-fold cross-validation. Some important parameters of the model are listed below:
  learning_rate = 0.69472, 
  max_depth = 2,
  n_estimators = 400, 
  subsample = 0.996.

# 4. Web Application
https://share.streamlit.io/melody1745/predicting-loan-default/main/loan_app.py
