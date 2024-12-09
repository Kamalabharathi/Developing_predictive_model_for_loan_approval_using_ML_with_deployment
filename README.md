# Developing a Predictive Model for Loan Approval Using ML with Deployment process
## ABSTRACT
Banks are making major part of profits through loans. Loan approval is a very 
important process for banking organizations. It is very difficult to predict the 
possibility of payment of loan by the customers because there is an increasing rate of 
loan defaults and the banking authorities are finding it more difficult to correctly 
access loan requests and tackle the risks of people defaulting on loans. In the recent 
years, many researchers have worked on prediction of loan approval systems. Machine 
learning technique is very useful in predicting outcomes for large amount of data. This 
project proposes to use Machine Learning algorithms such as Decision Tree and 
Logistic Regression algorithm to predict the loan approval of customers. The two 
algorithms are going to be used on the loan approval dataset and going to find the 
algorithm with maximum accuracy to deploy the model. 


## INTRODUCTION
A loan is a financial arrangement where one party, typically a bank or financial institution, provides money to another party with the agreement that the money will be repaid over time, usually with interest. Loans are commonly used for various purposes, such as purchasing a home, financing education, or starting a business. The borrower agrees to the terms set by the lender, including the repayment schedule and interest rate. This arrangement helps individuals and businesses access funds they might not have readily available, enabling them to achieve their financial goals.

This is also a core business part of banks.  The main portion the bank’s profit is directly come from the profit earned from the loans. 

Though bank approves loan after a regress process of verification and testimonial but still there’s no surely whether the chosen hopeful is the right hopeful or not.

In the modern financial landscape, the ability to accurately assess loan applications is crucial for minimizing risk and ensuring sustainable lending practices. 

This project leverages machine learning techniques to develop a robust model for predicting loan approval based on a diverse set of applicant attributes.

## LITERATURE SURVEY
The chapter reviews are a papers in the domain of predicting loan approval. 
1. Loan Approval Prediction Using Deep Neural Networks.
AUTHOR - Nguyen, T., Le, H., & Tran, K. 
YEAR: 2020

This study leverages Deep Neural Networks (DNNs) to enhance loan approval prediction accuracy, achieving 69% accuracy by optimizing architectures and hyperparameters. It provides valuable insights for financial institutions in decision-making and risk management.

2. Loan Default Prediction with Extreme Gradient Boosting (XGBoost). 
AUTHOR - Chen, Y., Liu, M., & Zhang, H. 
YEAR: 2021

This study evaluates XGBoost for predicting loan defaults, achieving 66% accuracy through optimized preprocessing and hyperparameter tuning. It offers valuable insights for financial institutions to improve risk management and decision-making.

3. Comparison of Random Forest and Gradient Boosting in Predicting Loan 
Defaults.
AUTHOR - Patel, A., Kumar, R., & Sharma, M. 
YEAR – 2018 

This study compares Random Forest and Gradient Boosting for loan default prediction, analyzing applicant data like income and credit history. Random Forest achieved 68% accuracy, highlighting its effectiveness in identifying high-risk borrowers for financial institutions.

Following the papers in literature, this project proposes to use the simple ML 
algorithms Logistic Regression and Decision Tree for predicting approval of loan.

## PROBLEM STATEMENT
Checking details of all applicants consumes lot of time and efforts.  There is chances of human error may occur due checking all details manually.  There is possibility of assigning loan to ineligible applicant.

So, developing a automatic loan prediction using machine learning techniques. Train the machine with previous dataset, so machine can analyse and understand the process.  Then machine will check for eligible applicant and give us result.

## OBJECTIVES

1. Data Exploration and Preprocessing: To Conduct thorough data exploration to understand the dataset's structure, identify missing values, and perform necessary preprocessing steps.

2. Feature Engineering: To Generate relevant features that can enhance the predictive power of the machine learning model.

3. Model Development: To Implement and evaluate suitable machine learning algorithm to identify the most effective model for predicting loan approval.

4. Model Evaluation and Optimization: To Assess the performance of the developed models using appropriate metrics and optimize the selected model for better accuracy and generalization.

5. Deployment: To Develop a user-friendly interface for deploying the predictive model, enabling financial institutions to input applicant data and receive loan approval predictions.

## DATA SOURCE
The dataset that is used in this project is taken from the Kaggle https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

The Kaggle Datasets website is a hub for finding, exploring, and sharing datasets for data science and machine learning projects. Here’s a detailed guide to help you navigate and make the most of the Kaggle Datasets website.

## DESCRIPTION
The dataset used in this project contains 599 records with 12 attributes, each representing different aspects of loan applicants' profiles. 

* Loan_ID: A unique identifier for each loan application.
* Gender: The gender of the applicant (e.g., Male or Female).
* Married: Marital status of the applicant (e.g., Yes or No).
* Dependents: The number of dependents the applicant has (e.g., 0, 1, 2, or 3+).
* Education: The educational qualification of the applicant (e.g., Graduate or Not Graduate).
* Self_Employed: Whether the applicant is self-employed (e.g., Yes or No).
* ApplicantIncome: The income of the primary applicant.
* CoapplicantIncome: The income of the co-applicant, if any.
* LoanAmount: The loan amount requested by the applicant (may include some missing values in the dataset).
* Loan_Amount_Term: The term of the loan (in months).
* Credit_History: A binary indicator of the applicant's credit history (1 for good credit history, 0 for bad).
* Property_Area: The type of property area where the applicant resides (e.g., Urban, Rural, or Semiurban).
* Loan_Status: The outcome of the loan application (e.g., Y for approved, N for not approved).

## METHODOLOGY
This project involves developing and deploying Machine Learning models for predicting loan approvals.

1. Data Discovery & Collection: Analyze loan application data, focusing on applicant demographics, financial details, and loan history from sources like banks or Kaggle.
2. Data Preparation: Handle missing values, perform feature engineering, encode categorical variables, and normalize numeric features.
3. Model Selection: Build Logistic Regression and Decision Tree models using the One-vs-All approach to classify loans as approved or denied.
4. Training & Evaluation: Train models iteratively, assess performance using accuracy, precision, recall, F1 score, and K-fold cross-validation.
5. Parameter Tuning: Optimize Decision Tree depth and pruning; adjust Logistic Regression regularization and learning rate.
6. Predictions: Predict loan approval outcomes based on testing data.
7. Deployment: Deploy models via Flask/Django web applications for real-world loan approval predictions.

## CONCLUSION: 
The Project developed a Machine Learning models, specifically Decision Tree 
and Logistic Regression, to predict loan approval. Logistic Regression proved to be 
the best-performing model in terms of accuracy. It successfully captured the 
relationships between applicant features such as demographics, income, and credit 
score, which contributed to its strong predictive ability. The deployment process was 
implemented to allow users to check their loan approval status in real-time by 
inputting their details through a web-based interface. This system improved the 
decision-making process for financial institutions by streamlining loan approvals with 
high precision. 

## FUTURE WORK: 
* Bank Loan Approval prediction to connect with cloud. 
* To optimize the work to implement in Artificial Intelligence environment.
