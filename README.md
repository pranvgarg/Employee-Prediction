# Employee Retention Prediction

## Project Overview

This project aims to develop a model to predict employee churn based on past professional, demographic, and attrition data. The goal is to help leadership allocate resources efficiently for employee retention, forecast headcount gaps, and develop succession plans.

## Agenda

1. Introduction to Problem
2. Feature Engineering
3. Model Overview
4. Model Accuracy
5. Business Implications
6. Implementation Plan

## Problem

Leadership wants to know an efficient way to allocate resources for employee retention. Teams need to forecast headcount gaps and develop succession plans. Management wants to implement proactive measures to reduce turnover costs.

## Potential Solution

Develop a model based on past employee professional, demographic, and attrition data to forecast churn in a given year.

## Data

The dataset includes the following features:

- Education Level
- Joining Year
- City
- Payment Tier
- Age
- Gender
- Benched
- Experience in Current Field
- Leave or Not

## Feature Engineering

### One-Hot Encoding
### New Variables
### Original Predictors
### Dropping Variables

### Correlation Matrix

- No significant correlation suggesting collinearity.
- Moderate correlation between office city and education level.
- Non-perfect correlation between one-hot encoded variables with more than two possibilities.

## Model Overview

Four models were developed and compared:

- Decision Tree Classifier
- AdaBoost Classifier
- Random Forest
- Tuned Random Forest

### Model Parameters

**Decision Tree Classifier**

- Criterion: `gini`
- Max Depth: 3

**AdaBoost Classifier**

- Number of Estimators: 100

**Random Forest**

- Criterion: `gini`
- Number of Estimators: 100
- Max Depth: 5
- Max Features: `sqrt`
- Min Samples Leaf: 2
- Min Samples Split: 10

**Tuned Random Forest**

- Criterion: `gini`
- Number of Estimators: 250
- Max Depth: 7
- Max Features: `sqrt`
- Min Samples Leaf: 2
- Min Samples Split: 5

## Model Accuracy

| Metric      | 0    | 1    | Avg   |
|-------------|------|------|-------|
| Precision   | 0.84 | 0.94 | 0.89  |
| Recall      | 0.98 | 0.65 | 0.81  |
| F1-Score    | 0.91 | 0.77 | 0.84  |
| Support     | 610  | 321  | 931   |

### Calibration Plot for Tuned Random Forest

### Model Calibration Takeaways

- Insights into variables leading to a drop in Gini impurity.
- Dependent on RandomForest model.

### Feature Importance and Shapley Values

## Conclusion and Takeaways

- Low tenure employees predicted to leave are likely to leave.
- PhD holders predicted to leave are likely to leave.
- Employees who are benched are likely to leave.

### Implications

- Invest in professional development and favorable pay packages for low tenure group.
- Engage benched employees with teams.
- Recognize that PhD holders have options and may require special attention.

### Suggestions

- Add quantitative variables such as hours per week worked and years since last promotion to improve model performance.

## Implementation Plan

- Implement proactive measures based on model insights.
- Monitor and adjust resource allocation strategies.
- Develop a comprehensive succession plan based on forecasted headcount gaps.

## Files

- `data/Employee.csv`: The dataset used for model development.
- `notebooks/Employee_Prediction.ipynb`: Jupyter notebook with model development and analysis.
- `scripts/Employee_Prediction.R`: R script for data processing and feature engineering.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Employee-Prediction.git
cd Employee-Prediction