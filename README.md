# Credit Risk Resampling

# Credit Risk Classification

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this project, various techniques are used to train and evaluate models with imbalanced classes. The dataset consists of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.


### 1. Splitting the Data into Training and Testing Sets

1. The lending_data.csv data is read into a Pandas DataFrame.

2.Labels are set; (`y`) from the “loan_status” column, and then the features (`X`) DataFrame are created from the remaining columns.

**Note** A value of `0` in the “loan_status” column means that the loan is healthy. A value of `1` means that the loan has a high risk of defaulting.  

3. The balance of the labels variable (`y`) is checked by using the `value_counts` function.

4. Data is split into training and testing datasets by using `train_test_split`.

### 2 A Logistic Regression Model is created with the Original Data

Logistic regression is applied to complete the following steps:

1. Fitting of a logistic regression model by using the training data (`X_train` and `y_train`).

2. Saved the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.

3. Evaluated the model’s performance by doing the following:

    * Calculating the accuracy score of the model.

    * Generating a confusion matrix.

    * Printing the classification report.

4. Answered the question: How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

### 3 A Logistic Regression Model is predicted with Resampled Training Data 

A small number of high-risk loan labels was noticed; the training data was resampled and the model was reevaluated as follows:

1. The `RandomOverSampler` module was used, from the imbalanced-learn library to resample the data, confirming that the labels had an equal number of data points. 

2. Used the `LogisticRegression` classifier and then resampled data to fit the model and make predictions.

3. Evaluated the model’s performance by:

    * Calculating the accuracy score of the model.

    * Generating a confusion matrix.

    * Printing the classification report.
    
4. Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

### Write a Credit Risk Analysis Report

For this section, you’ll write a brief report that includes a summary and an analysis of the performance of both machine learning models that you used in this challenge. You should write this report as the `README.md` file included in your GitHub repository.

Structure your report by using the report template that `Starter_Code.zip` includes, and make sure that it contains the following:
   
## Overview of the Analysis

This section describes the analysis completed for the machine learning models used in this project.  

* Purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.
 
* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

3. A summary: Summarize the results from the machine learning models. Compare the two versions of the dataset predictions. Include your recommendation for the model to use, if any, on the original vs. the resampled data. If you don’t recommend either model, justify your reasoning.

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
