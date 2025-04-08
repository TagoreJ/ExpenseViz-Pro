# 💰 Personal Expense Tracker Analysis

A complete data analysis and modeling project focused on understanding and predicting personal income, expenses, and savings behavior using R. This project includes data cleaning, exploratory data analysis (EDA), regression and classification models, time series forecasting, clustering, and anomaly detection.

---

## 📌 Project Overview

This project investigates how individuals spend, save, and manage their income. By analyzing patterns across demographics, job types, and city tiers, we extract key insights and build models to predict financial behavior.

---

## 🧰 Tools & Libraries

- **Language**: R
- **Libraries**: `tidyverse`, `ggplot2`, `dplyr`, `lubridate`, `caret`, `randomForest`, `cluster`, `forecast`, `factoextra`, `zoo`, `ggthemes`, `corrplot`

---

## 📊 Features Engineered

- `Total_Expenses`: Sum of all expense categories
- `Expense_to_Income_Ratio`: Indicator of financial strain
- `Disposable_Income`: Income minus total expenses
- `High_Saver`: Logical flag for those whose disposable income ≥ desired savings
- `Age_Group`: Categorized age ranges

---

## 🔍 Exploratory Data Analysis (EDA)

- **Distribution Analysis**: Histograms for income, expenses, savings
- **Boxplots**: City tier vs expenses/income/savings
- **Correlation Matrix**: Relationship strength between variables
- **Visualizations**: ggplot2-based bar, line, and scatter plots

### 📝 Key Insights:
- Tier 1 cities have higher incomes but also higher expenses.
- Occupation significantly affects saving/spending behavior.
- Middle-aged individuals tend to save more consistently.
- Strong correlation between income and desired savings.

---

## 📈 Predictive Modeling

### 1. **Regression (Predicting Disposable Income)**

- Algorithms: Linear Regression, Random Forest
- Evaluation: RMSE, R²
- ✅ **Random Forest outperformed linear regression** with better accuracy and lower error.

### 2. **Classification (High Saver or Not)**

- Algorithm: Logistic Regression
- Metrics: Accuracy, Precision, Recall, F1 Score

> Models successfully identify high savers based on income, expenses, occupation, and age.

---

## 📅 Time Series Forecasting

- Goal: Predict future **Groceries expenses** over the next 6 months
- Method: ARIMA model using historical expense data

> Forecast shows consistent monthly trends useful for budget planning.

---

## 🚨 Anomaly Detection

- Method: Z-score based
- Focus: Detect unusually high grocery spenders

> Several users were flagged with outlier behavior — useful for identifying overspending or fraud.

---

## 📊 Clustering Analysis

### A. Income + Savings Clustering:
- Grouped users based on financial profile
- Result: 3 distinct financial behavior clusters

### B. Expense Category Clustering:
- Grouped based on spending style: groceries, entertainment, transport, etc.
- Visualized with `fviz_cluster`

> Clusters help identify user personas: Budgeters, Balanced Spenders, and Free Spenders

---

## 🌆 City Tier Analysis

- Compared Tier 1, 2, and 3 cities
- Insights:
  - Tier 1: High income, high expense, low savings
  - Tier 2/3: Modest income, controlled spending, better savings ratio
  - Insurance cost and savings vary significantly by tier

---

## 📦 Project Structure

```bash
📁 Personal_Expense_Tracker/
├── data/                      # Raw and cleaned datasets
├── scripts/                   # R scripts for analysis and modeling
├── plots/                     # Exported charts and visualizations
├── models/                    # Saved model objects (if applicable)
└── README.md                  # Project documentation
