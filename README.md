# Credit Risk Classification and Resampling

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this project, various techniques are used to train and evaluate models with imbalanced classes. The dataset consists of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

## Overview of the Analysis

This section describes the analysis completed for the machine learning models used in this project.  

* The Purpose of the analysis was to compare two different regression methods in Supervised Machine Learning.

* The financial information analyzed was over 77,000 records of loan from datalending_data.csv; a value of `0` in the “loan_status” column means that the loan is healthy. A value of `1` means that the loan has a high risk of defaulting.  

* The prediction used dataset samples that was split into training and testing variables using function `train_test_split`; (`y`) from the “loan_status” column, and then the features (`X`) DataFrame were created from the remaining columns.  
  
* The stages of the machine learning process modeling were Preprocess, Train, Validate, and Predict.
  
* Machine learning methods utilized included `value_counts`, `RandomOverSampler` module from the imbalanced-learn library to resample the data, confirming that the labels had an equal number of data points. Used the `LogisticRegression` classifier and then resampled data to fit the model and make predictions.


## Results

The models performance was evaluated by: 1) Calculating the accuracy score of the model. 2) Generating a confusion matrix. 3) Printing the classification report.
 
* Machine Learning Model 1: Original Dataset
Logistic regression was applied to complete the following steps: 1. Fitting of a logistic regression model by using the training data (`X_train` and `y_train`). 2. Saved the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.
      
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56271
           1       0.86      0.90      0.88      1881

    accuracy                           0.99     58152
   macro avg       0.93      0.95      0.94     58152
weighted avg       0.99      0.99      0.99     58152



* Machine Learning Model 2: Resampled Data
A small number of high-risk loans were identified with the first model, the data was resampled as follows:  
  
                    precision    recall  f1-score   support

  Healthy Loan '0'       0.99      0.99      0.99     56271
High-risk loan '1'       0.99      0.99      0.99     56271

          accuracy                           0.99    112542
         macro avg       0.99      0.99      0.99    112542
      weighted avg       0.99      0.99      0.99    112542

  
## Summary

In summary, the recommendation for the model to use is the original which has overall very good classification results vs. the resampled data, which in theory may have skewed results. 

Precision is the ratio of correctly predicted posibive observations vs. the total predicted positive observations; therefore, the original dataset model has 99% accuracy; predicting healthy loans with 100% precision and high-risk loans at 86% precision. 
This also means that the Classification rate or Recall, is zero for False Negative and 10% for False Positives.

Because logistic regression is used, performance does not depend on the problem type to solve (importance to predict the `1`'s, or predict the `0`'s ) however high results in the classification metrics can provide better confidence.  

## Usage
The code is in Jupyter notebook visual_data_analysis.ipynb

## Contributors
Ana Cano - Author 

## License
Copyright (c) 2011-2017 GitHub Inc. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

