library(ggplot2)
library(dplyr)
library(reshape2)
library(corrplot)
library(dplyr)
library(lubridate)




per = Personal_Exp_Tracker
head(per)
str(per)


# Basic statistics for numeric columns
summary(per)

# Group-wise average income by occupation
aggregate(Income ~ Occupation, per, mean)

# Group-wise desired savings by city tier
aggregate(Desired_Savings ~ City_Tier, per, mean)

# Correlation matrix of key numeric variables
num_cols <- per[, sapply(per, is.numeric)]
cor_matrix <- cor(num_cols)
print(cor_matrix)



# Convert categorical columns to factors
per$Occupation <- as.factor(per$Occupation)
per$City_Tier <- as.factor(per$City_Tier)

# Expense to income ratio
per$Total_Expenses <- rowSums(per[, grep("Rent|Repayment|Insurance|Groceries|Transport|Eating_Out|Entertainment|Utilities|Healthcare|Education|Miscellaneous", names(per))], na.rm = TRUE)
per$Expense_to_Income <- per$Total_Expenses / per$Income
per$Expense_to_Income

# Age group
per$Age_Group <- cut(per$Age, breaks = c(0, 25, 40, 60, Inf), labels = c("Young", "Adult", "Middle-aged", "Senior"))

# High saver flag
per$High_Saver <- ifelse(per$Disposable_Income > per$Desired_Savings, 1, 0)




# For Graphs
library(ggplot2)

# Histogram of Income
ggplot(per, aes(x = Income)) + geom_histogram(binwidth = 5000, fill = "skyblue", color = "black")

# Boxplot: Disposable Income by City Tier
ggplot(per, aes(x = City_Tier, y = Disposable_Income)) + geom_boxplot(fill = "lightgreen")

# Scatter plot: Income vs Desired Savings
ggplot(per, aes(x = Income, y = Desired_Savings)) + geom_point(alpha = 0.5) + geom_smooth(method = "lm")

# Bar plot: Average Entertainment expense by Occupation
agg <- aggregate(Entertainment ~ Occupation, per, mean)
ggplot(agg, aes(x = reorder(Occupation, -Entertainment), y = Entertainment)) +
  geom_bar(stat = "identity", fill = "coral") + coord_flip()



ggplot(data_clean, aes(x = Income)) +
  geom_histogram(binwidth = 100, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Expenses", x = "Income", y = "Frequency")


ggplot(data_clean, aes(x = Occupation, y = Income, fill = Occupation)) +
  geom_boxplot() +
  labs(title = "Expenses by Category", x = "Occupation", y = "Income") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(data_clean, aes(x = City_Tier, y = Income)) +
  geom_boxplot() +
  labs(title = "Income by City Tier", x = "City Tier", y = "Income") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# mORE Graphs

#Histogram

vars <- c("Income", "Age", "Desired_Savings", "Disposable_Income", 
          "Groceries", "Transport", "Entertainment", "Healthcare")

for (v in vars) {
  print(
    ggplot(per, aes_string(x = v)) +
      geom_histogram(fill = "skyblue", color = "black", bins = 30) +
      ggtitle(paste("Histogram of", v))
  )
}



# Boxplot by occupation

expense_vars <- c("Groceries", "Transport", "Entertainment")

for (v in expense_vars) {
  print(
    ggplot(per, aes_string(x = "Occupation", y = v)) +
      geom_boxplot(fill = "lightgreen") +
      ggtitle(paste(v, "by Occupation")) +
      theme(axis.text.x = element_text(angle = 90, hjust = 1))
  )
}


# Scatter plot


pairs <- list(
  c("Income", "Desired_Savings"),
  c("Income", "Disposable_Income"),
  c("Disposable_Income", "Desired_Savings"),
  c("Income", "Rent")
)

for (p in pairs) {
  print(
    ggplot(per, aes_string(x = p[1], y = p[2])) +
      geom_point(alpha = 0.5, color = "tomato") +
      ggtitle(paste(p[1], "vs", p[2]))
  )
}

# Correaltion heatmap

# Select numeric columns
num_cols <- sapply(per, is.numeric)
cor_per <- cor(per[, num_cols], use = "complete.obs")

# Plot
corrplot(cor_per, method = "color", tl.cex = 0.7, type = "upper")




#All Graphs

ggplot(per, aes(x = Income)) + geom_histogram(bins = 30, fill = "blue", color = "white") + ggtitle("Income Distribution")
ggplot(per, aes(x = Disposable_Income)) + geom_histogram(bins = 30, fill = "green", color = "white") + ggtitle("Disposable Income Distribution")
ggplot(per, aes(x = Desired_Savings)) + geom_histogram(bins = 30, fill = "purple", color = "white") + ggtitle("Desired Savings")
ggplot(per, aes(x = Groceries)) + geom_histogram(bins = 30, fill = "orange", color = "white") + ggtitle("Groceries Expense")
ggplot(per, aes(x = Transport)) + geom_histogram(bins = 30, fill = "red", color = "white") + ggtitle("Transport Expense")
ggplot(per, aes(x = Entertainment)) + geom_histogram(bins = 30, fill = "skyblue", color = "white") + ggtitle("Entertainment Expense")
ggplot(per, aes(x = Utilities)) + geom_histogram(bins = 30, fill = "darkgreen", color = "white") + ggtitle("Utilities Expense")
ggplot(per, aes(x = Healthcare)) + geom_histogram(bins = 30, fill = "magenta", color = "white") + ggtitle("Healthcare Expense")




# BOXPLOT
ggplot(per, aes(x = Occupation, y = Transport)) + geom_boxplot() + ggtitle("Transport by Occupation") + theme(axis.text.x = element_text(angle = 45))
ggplot(per, aes(x = Occupation, y = Groceries)) + geom_boxplot() + ggtitle("Groceries by Occupation") + theme(axis.text.x = element_text(angle = 45))
ggplot(per, aes(x = Occupation, y = Entertainment)) + geom_boxplot() + ggtitle("Entertainment by Occupation") + theme(axis.text.x = element_text(angle = 45))
ggplot(per, aes(x = Occupation, y = Utilities)) + geom_boxplot() + ggtitle("Utilities by Occupation") + theme(axis.text.x = element_text(angle = 45))
ggplot(per, aes(x = Occupation, y = Healthcare)) + geom_boxplot() + ggtitle("Healthcare by Occupation") + theme(axis.text.x = element_text(angle = 45))

# SCatter 
ggplot(per, aes(x = Income, y = Desired_Savings)) + geom_point() + ggtitle("Income vs Desired Savings")
ggplot(per, aes(x = Income, y = Disposable_Income)) + geom_point() + ggtitle("Income vs Disposable Income")
ggplot(per, aes(x = Disposable_Income, y = Desired_Savings)) + geom_point() + ggtitle("Disposable Income vs Desired Savings")
ggplot(per, aes(x = Income, y = Entertainment)) + geom_point() + ggtitle("Income vs Entertainment")
ggplot(per, aes(x = Disposable_Income, y = Education)) + geom_point() + ggtitle("Disposable Income vs Education")



ggplot(per, aes(x = Occupation)) + geom_bar(fill = "coral") + ggtitle("Occupation Distribution") + theme(axis.text.x = element_text(angle = 45))
ggplot(per, aes(x = Gender)) + geom_bar(fill = "turquoise") + ggtitle("Gender Distribution")
ggplot(per, aes(x = City_Tier)) + geom_bar(fill = "gold") + ggtitle("City Tier Distribution")
ggplot(per, aes(x = Education)) + geom_bar(fill = "plum") + ggtitle("Education Distribution")
ggplot(per, aes(x = Age)) + geom_bar(fill = "steelblue") + ggtitle("Age Distribution")
ggplot(per, aes(x = Marital_Status)) + geom_bar(fill = "seagreen") + ggtitle("Marital Status Distribution")





# Train Set for Regression

set.seed(123)
library(caret)

# Split data: 80% train, 20% test
trainIndex <- createDataPartition(df$Disposable_Income, p = 0.8, list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]


# Fit linear regression model
lm_model <- lm(Disposable_Income ~ ., data = train)

# Summary
summary(lm_model)

# Predict on test set
pred_lm <- predict(lm_model, newdata = test)
pred_lm



# RMSE and R² ( for checking model performance)
library(Metrics)
rmse_value <- rmse(test$Disposable_Income, pred_lm)
r2_value <- R2(pred_lm, test$Disposable_Income)

cat("Linear Regression:\nRMSE:", rmse_value, "\nR²:", r2_value, "\n")


# Using Random Forest

library(randomForest)

rf_model <- randomForest(Disposable_Income ~ ., data = train, ntree = 100)
pred_rf <- predict(rf_model, newdata = test)

rmse_rf <- rmse(test$Disposable_Income, pred_rf)
r2_rf <- R2(pred_rf, test$Disposable_Income)

cat("Random Forest:\nRMSE:", rmse_rf, "\nR²:", r2_rf, "\n")



# Hyper pariamte tuning

# Tune ntree and mtry using caret
control <- trainControl(method = "cv", number = 5)
tunegrid <- expand.grid(.mtry = c(2, 5, 8))

tuned_rf <- train(
  Disposable_Income ~ ., data = train,
  method = "rf",
  trControl = control,
  tuneGrid = tunegrid,
  ntree = 100
)

# Best tuned model
print(tuned_rf)




# logiestic regression

# Make sure High_Saver is a factor
df$High_Saver <- as.factor(df$High_Saver)

# Split again if needed
trainIndex <- createDataPartition(df$High_Saver, p = 0.8, list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]

# Logistic Regression
log_model <- glm(High_Saver ~ ., data = train, family = "binomial")

# Predict
pred_prob <- predict(log_model, newdata = test, type = "response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# Confusion matrix
confusionMatrix(factor(pred_class), test$High_Saver)

# Accuracy, Precision, Recall, F1
library(MLmetrics)
accuracy <- Accuracy(pred_class, as.numeric(as.character(test$High_Saver)))
precision <- Precision(pred_class, as.numeric(as.character(test$High_Saver)))
recall <- Recall(pred_class, as.numeric(as.character(test$High_Saver)))
f1 <- F1_Score(pred_class, as.numeric(as.character(test$High_Saver)))

cat("Logistic Regression:\nAccuracy:", accuracy, "\nPrecision:", precision,
    "\nRecall:", recall, "\nF1 Score:", f1, "\n")




# Aggregate monthly groceries spend 

monthly_groceries <- per %>%
  mutate(Month = floor_date(as.Date(Date), "month")) %>%
  group_by(Month) %>%
  summarise(Groceries = mean(Groceries))

# Convert to ts object
groceries_ts <- ts(monthly_groceries$Groceries, frequency = 12)

# Fit ARIMA model
fit <- auto.arima(groceries_ts)

# Forecast next 6 months
forecast_data <- forecast(fit, h = 6)
plot(forecast_data)




# Anomaly Detection – Unusual Expenses


#Basic Z-Score Method:

# Flag rows where a category expense is unusually high (e.g., Groceries)
df$Groceries_z <- scale(df$Groceries)

# Threshold at ±3 standard deviations
df$Groceries_Anomaly <- ifelse(abs(df$Groceries_z) > 3, 1, 0)

# View anomalies
anomalies <- df %>% filter(Groceries_Anomaly == 1)
print(anomalies)






# Clustering (Income)

# K-means clustering based on income and savings
income_data <- per[, c("Income", "Desired_Savings")]
income_data_scaled <- scale(income_data)
income_data_scaled
set.seed(123)
km_income <- kmeans(income_data_scaled, centers = 3)
per$Income_Cluster <- factor(km_income$cluster)

# Plot cluster centers
library(factoextra)
fviz_cluster(km_income, data = income_data_scaled, geom = "point", 
             ellipse.type = "convex", main = "K-means Clustering of Income and Savings",
             xlab = "Income", ylab = "Desired Savings")



# K-means clustering based on expense categories
expenses <- per[, c("Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities")]
expenses_scaled <- scale(expenses)
expenses_scaled

set.seed(123)
km <- kmeans(expenses_scaled, centers = 3)
per$Cluster <- factor(km$cluster)

# Plot cluster centers

fviz_cluster(km, per = expenses_scaled)

# Plot clusters
ggplot(per, aes(x = Groceries, y = Transport, color = Cluster)) +
  geom_point() +
  labs(title = "K-means Clustering of Expenses", x = "Groceries", y = "Transport") +
  theme_minimal()




# Predictive Modeling: Predict Insurance or Disposable Income

# Encode City_Tier as factor
per$City_Tier <- as.factor(per$City_Tier)

# Split data
set.seed(123)
library(caret)
trainIndex <- createDataPartition(per$Insurance, p = 0.8, list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]

# Linear regression model
lm_insurance <- lm(Insurance ~ Income + Age + Dependents + City_Tier + Occupation, data = train)
summary(lm_insurance)

# Predict on test
pred_ins <- predict(lm_insurance, newdata = test)

# Evaluate
library(Metrics)
cat("Insurance Prediction RMSE:", rmse(test$Insurance, pred_ins), "\n")



