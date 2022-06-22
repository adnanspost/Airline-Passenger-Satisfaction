# Airline-Passenger-Satisfaction

## Introdaction:

This project involves solving a binary classification machine learing problem in the airline industry.

## Aim:

The aim of this project is to make accurate predictions from historical data in regards to whether a customer is satisfied with the service received or not.

## Methodolgy:

While the fine details of the methdology followed in this project will be elaborated further in the relevant sections to come, it is to be mentioned that 16 machine learning models have been implemented, compared, and contrasted to identify the best performing one. 

### 01. Import CPU Python Libraries

### 02. Function Helper

### 03. Import Dataset & Data Description
- Import CSV File
- Data Description

### 04. Data Understanding
- Data Information
- Data Summary Statistic
- Data Variance

### 05. Select the Featurs

### 06. Data Pre-Processing
- Drop Variables
- Convert Data Type
- Missing Value

### 07. Exploratory Data Analysis
- DV Visualization
- Categorical IDV
- Categorical IDV With DV
- Numerical IDV
- Numerical IDV With DV

### 08. Data Transformation
- Standard Scaler

### 09. Feature Selection
- Wrapper - Forward

### 10. Feature Engineering
- LableEncoder

### 11. Statistics
- Correlations IDV with DV
- Correlation between all the Variables

### 12. Resampling Data
- SMOTE

### 13. Data Splitting

### 14. Standard Machine Learning Models
- Fit the Machine Learning Models 'Train the Models'
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - Histogram-based Gradient Boosting Classification Tree
    - AdaBoost Classifier
    - Extra Trees Classifier
    - K Neighbors Classifier
    - Naive Bayes Classifiers
    - Naive Bayes Classifier for Multivariate Bernoulli
    - Decision Tree Classifier
    - Logistic Regression Classifier
    - Logistic Regression CV Classifier
    - Stochastic Gradient Descent Classifier
    - Linear Perceptron Classifier
    - XGBoost Classifiers
    - Support Vector Machines Classifiers
    - Linear Support Vector Classification
    - Multilayer Perceptron Classifier
- Predicat X_test
- Models Evaluation
    - Accuracy Score
    - Classification Report
    - Confusion Matrix

### 15. Accuracy Score Summary
 

## Results:

For this particular classiication problem, the highest performing model is that of Extra Trees Classifier which achieved an accuracy score of 93.8% with a F1 Score of xx, Recall of yy, and xyz score of zz.


![results](https://user-images.githubusercontent.com/108016592/175097013-6b300503-f4c9-40dc-a278-aac369d4df19.png)

![MicrosoftTeams-image (13)](https://user-images.githubusercontent.com/108016592/175095830-43d632fc-714c-4562-84cc-bb9d6e4d9666.png)

## Discussion:

Although the Extra Tree Classifier has been observed to display the highest accuracy in comparison to all the other models, this score alone is not the sole determinant for choosing the Extra Tree Classifier model.

Upon closer inspection it is observed that the Extra Tree Classifier outperforms all the other models in F1 Score, Recall, and xyz. Should the latter scores have been observed to be higher in the other models, perhaps another model would have been determined to be the best performer as opposed to the Extra Tree Classifier.

## Conclusion:

The results achieved through most of the machine learning models in this project have achieved accuracy scores of above 80% and the best performer of all the 16 models that have been compared and contrasted has been determined to be the Extra Tree Classifier with an accuracy.

Having that said, it is to be mentioned that this performance can be expected to improve further by optimizing the models utilzed. This can be done by experimenting further in terms of changing the hyper-parameters of the models, changing the method of feature selection, or using a different method of data transformation before deploying the machine learning models.
