suppressWarnings({
  library(MASS)
  library(caret)
  library(randomForest)
  library(gbm)
  library(rpart)
  library(rpart.plot)
  library(tidyverse)
  library(ISLR2)
  library(MLmetrics)
  library(MLeval)
  library(pROC)
  library(PRROC)
  })

#load data
Employee <- read_csv("Employee.csv")

#set seed
set.seed(18)
# If there is a class imbalance, consider upsampling the minority class
if (sum(Employee$LeaveOrNot == 1) == 0 || sum(Employee$LeaveOrNot == 0) == 0) {
  stop("One of the classes is not represented in the data.")
}

###########################################################################
# Feature Engineering
###########################################################################
Employee$Tenure <- as.integer(2018 - Employee$JoiningYear)

#remove joining year variable by using subset
Employee <- subset(Employee, select = -(JoiningYear))

#casting categorical variables as factors

#Education: bachelors, masters, & PHD
Employee$Education = (factor(Employee$Education, levels=c("Bachelors", "Masters", "PHD")))

#City: Pune, Bangalore, New Delhi
Employee$City = (factor(Employee$City, levels=c("Pune", "Bangalore", "New Delhi")))

#Payment Tier: 1, 2, 3
Employee$PaymentTier = (factor(Employee$PaymentTier, levels=c(1, 2, 3)))

#Gender: Female, Male
Employee$Gender = (factor(Employee$Gender, levels=c("Female", "Male")))

#Ever benched: Yes, No
Employee$EverBenched = (factor(Employee$EverBenched, levels=c("Yes", "No"), labels = c("Yes", "No")))

#target variable -- Leave or Not
Employee$LeaveOrNot <- ifelse(Employee$LeaveOrNot == 1, "Yes", "No")
#unique_values <- unique(Employee$LeaveOrNot)
#print(unique_values)
Employee$LeaveOrNot = factor(Employee$LeaveOrNot, levels=c("Yes", "No"))

#check the data types to ensure that we now have the correct data types for our model
str(Employee)

#create data sets for training and testing
# Hold out 20% of the data as a final validation set
train_ix = createDataPartition(Employee$LeaveOrNot,
                               p = 0.8)

#create train-test split
Employee_train = Employee[train_ix$Resample1,]
Employee_test  = Employee[-train_ix$Resample1,]

# Note that caret used stratified sampling to preserve
# the balance of Y/N:
table(Employee$LeaveOrNot[train_ix$Resample1]) %>% 
  prop.table
table(Employee$LeaveOrNot[-train_ix$Resample1]) %>% 
  prop.table


###########################################################################
# Setup cross-validation
###########################################################################

# Define how we're going to estimate OOS error using cross-validation

# Number of folds
kcv = 10

# I'm manually making the folds here so we can look at them, and so
# they're the same when we evaluate each method below. If you omit
# the indexOut argument below caret with make the folds behind the scenes.

cv_folds = createFolds(Employee_train$LeaveOrNot,
                       k = kcv)

# This function sets up how we're going to do our training: The method for
# estimating OOS error (CV) and associated settings (here the folds we created 
# above). I'm also going to request that our final fit is determined not by
# the minimum estimated OOS RMSE but using the one standard deviation (aka
# one standard error) rule instead by specifying selectionFunction="oneSE"

# Defining a new summary function that computes a few different 
# error metrics using pre-defined summaries

my_summary = function(data, lev = NULL, model = NULL) {
  default = defaultSummary(data, lev, model)
  twoclass = twoClassSummary(data, lev, model)
  # Converting to TPR and FPR instead of sens/spec
  twoclass[3] = 1-twoclass[3]
  names(twoclass) = c("AUC_ROC", "TPR", "FPR")
  logloss = mnLogLoss(data, lev, model)
  c(default,twoclass, logloss)
}

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds,
  # Save predicted probabilities, not just classifications
  classProbs = TRUE,
  # Save all the holdout predictions, to summarize and plot
  savePredictions = TRUE,
  summaryFunction = my_summary,
  selectionFunction="oneSE")


###########################################################################
# Boosting
###########################################################################

gbm_grid <-  expand.grid(interaction.depth = c(1, 3, 5, 10), 
                         n.trees = c(100, 500, 750, 1000, 1500), 
                         shrinkage = c(0.1),
                         n.minobsinnode = 10)

# I'm going to exclude some combinations to make CV faster
gbm_grid = gbm_grid %>% 
  filter(!(n.trees>1000 & interaction.depth>5))

gbmfit <- train(LeaveOrNot ~ ., data = Employee_train, 
                method = "gbm", 
                trControl = fit_control,
                tuneGrid = gbm_grid,
                metric = "logLoss",
                verbose = FALSE)

print(gbmfit)
plot(gbmfit)


# Extracting performance summaries
# Confusion matrix as proportions, not counts, since 
# the test dataset varies across folds
# These are CV estimates of error rates/accuracy using a *default* cutoff
# to classify cases

confusionMatrix(gbmfit)

thresholder(gbmfit, 
            threshold = 0.5, 
            final = TRUE,
            statistics = c("Sensitivity",
                           "Specificity"))


gbmfit_res = thresholder(gbmfit, 
                         threshold = seq(0.1, 0.99, by = 0.01), 
                         final = TRUE) # it is throwing NA values because the end points are from 0 to 1 changes to 0.12 0.99

# plot(J~prob_threshold, data=gbmfit_res, type='l')

optim_J = gbmfit_res[which.max(gbmfit_res$J),]

# ROC curve
ggplot(aes(x=1-Specificity, y=Sensitivity), data=gbmfit_res) + 
  geom_line() + 
  ylab("TPR (Sensitivity)") + 
  xlab("FPR (1-Specificity)") + 
  geom_abline(intercept=0, slope=1, linetype='dotted') +
  geom_segment(aes(x=1-Specificity, xend=1-Specificity, y=1-Specificity, yend=Sensitivity), color='darkred', data=optim_J) + 
  theme_bw() #error graph from 0 to 1

# PR curve
ggplot(aes(x=Recall, y=Precision), data=gbmfit_res) + 
  geom_point() + 
  geom_line() + 
  ylab("Precision") + 
  xlab("Recall (TPR)") + 
  geom_point(aes(x=Recall, y=Precision), color='darkred', data=optim_J) + 
  theme_bw()

# Lift curve

# Extract predicted probs for best-fitting model
# For each observation it's predicted prob is computed when its
# fold is the testing/holdout dataset during CV
best_pars = gbmfit$bestTune
best_preds = gbmfit$pred %>% filter(n.trees==best_pars$n.trees, 
                                    interaction.depth==best_pars$interaction.depth)

gbm_lift = caret::lift(pred~Yes, data=best_preds)

ggplot(gbm_lift) + 
  geom_abline(slope=1, linetype='dotted') +
  xlim(c(0, 10)) + 
  theme_bw()

# Calibration plot
gbm_cal = caret::calibration(obs~Yes, data=best_preds, cuts=7)
ggplot(gbm_cal) + theme_bw()

#### Holdout set results TEST
test_probs = predict(gbmfit, newdata=Employee_test, type="prob")

get_metrics = function(threshold, test_probs, true_class, 
                       pos_label, neg_label) {
  # Get class predictions
  pc = factor(ifelse(test_probs[pos_label]>threshold, pos_label, neg_label), levels=c(pos_label, neg_label))
  test_set = data.frame(obs = true_class, pred = pc, test_probs)
  my_summary(test_set, lev=c(pos_label, neg_label))
}

# Get metrics for a given threshold
get_metrics(0.75, test_probs, Employee_test$LeaveOrNot, "Yes", "No")

# Compute metrics on test data using a grid of thresholds
thr_seq = seq(0, 1, length.out=500)
metrics = lapply(thr_seq, function(x) get_metrics(x, test_probs, Employee_test$LeaveOrNot, "Yes", "No"))
metrics_df = data.frame(do.call(rbind, metrics))

# ROC curve
ggplot(aes(x=FPR, y=TPR), data=metrics_df) + 
  geom_line() +
  ylab("TPR (Sensitivity)") + 
  xlab("FPR (1-Specificity)") + 
  geom_abline(intercept=0, slope=1, linetype='dotted') +
  annotate("text", x=0.75, y=0.25, 
           label=paste("AUC:",round(metrics_df$AUC_ROC[1], 2))) +
  theme_bw()

# Lift
gbm_oos_lift = caret::lift(Employee_test$LeaveOrNot~test_probs[,1])
ggplot(gbm_oos_lift) + 
  geom_abline(slope=1, linetype='dotted') +
  xlim(c(0, 100)) + 
  theme_bw()

# Calibration
gbm_cal = caret::calibration(Employee_test$LeaveOrNot~test_probs[,1], 
                             data=best_preds, cuts=11)
ggplot(gbm_cal) + theme_bw()



# Extract out-of-sample predicted probs for the optimal model
best_pars = gbmfit$bestTune
best_preds = gbmfit$pred %>% filter(n.trees==best_pars$n.trees, 
                                       interaction.depth==best_pars$interaction.depth)

# We can extract more information using the MLeval package
gbm_perf = evalm(gbmfit, showplots=FALSE) #ERROR

# The return object contains ggplot objects you can customize
gbm_perf$roc + 
  ggtitle("ROC curve for GBM") +
  theme_bw() 

gbm_perf$proc + 
  ggtitle("Precision/recall curve for GBM") +
  theme_bw() 

gbm_perf$cc + 
  ggtitle("Calibration curve for GBM")

###########################################################################
# Random Forest
###########################################################################
# Load necessary libraries
library(randomForest)
library(caret)
library(pROC)

# Setting up cross-validation
fit_control <- trainControl(method = "cv", 
                            number = 10,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary,
                            savePredictions = TRUE)

# Define the grid of hyperparameters
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5, 6),
                       ntree = c(100, 500, 1000),
                       nodesize = c(1, 5, 10),
                       maxnodes = c(10, 15, 20))

# Train the Random Forest model
rffit <- train(LeaveOrNot ~ ., data = Employee_train, 
               method = "rf", 
               trControl = fit_control,
               tuneGrid = rf_grid,
               metric = "ROC",
               importance = TRUE)

print(rffit)

plot(rffit)

# Extracting performance summaries
confusionMatrix(rffit)

thresholder(rffit, 
            threshold = 0.5, 
            final = TRUE,
            statistics = c("Sensitivity",
                           "Specificity"))

rffit_res = thresholder(rffit, 
                        threshold = seq(0.01, 0.999, by = 0.01), 
                        final = TRUE)

optim_J = rffit_res[which.max(rffit_res$J),]

# ROC curve
ggplot(aes(x = 1 - Specificity, y = Sensitivity), data = rffit_res) + 
  geom_line() + 
  ylab("TPR (Sensitivity)") + 
  xlab("FPR (1-Specificity)") + 
  geom_abline(intercept = 0, slope = 1, linetype = 'dotted') +
  geom_segment(aes(x = 1 - Specificity, xend = 1 - Specificity, y = 1 - Specificity, yend = Sensitivity), color = 'darkred', data = optim_J) + 
  theme_bw()

# PR curve
ggplot(aes(x = Recall, y = Precision), data = rffit_res) + 
  geom_point() + 
  geom_line() + 
  ylab("Precision") + 
  xlab("Recall (TPR)") + 
  geom_point(aes(x = Recall, y = Precision), color = 'darkred', data = optim_J) + 
  theme_bw()

# Lift curve
best_pars = rffit$bestTune
best_preds = rffit$pred %>% filter(mtry == best_pars$mtry)

rf_lift = caret::lift(pred ~ Yes, data = best_preds)

ggplot(rf_lift) + 
  geom_abline(slope = 1, linetype = 'dotted') +
  xlim(c(0, 10)) + 
  theme_bw()

# Calibration plot
rf_cal = caret::calibration(obs ~ Yes, data = best_preds, cuts = 7)
ggplot(rf_cal) + theme_bw()

#### Holdout set results TEST
test_probs = predict(rffit, newdata = Employee_test, type = "prob")

get_metrics = function(threshold, test_probs, true_class, 
                       pos_label, neg_label) {
  pc = factor(ifelse(test_probs[pos_label] > threshold, pos_label, neg_label), levels = c(pos_label, neg_label))
  test_set = data.frame(obs = true_class, pred = pc, test_probs)
  my_summary(test_set, lev = c(pos_label, neg_label))
}

get_metrics(0.75, test_probs, Employee_test$LeaveOrNot, "Yes", "No")

thr_seq = seq(0, 1, length.out = 500)
metrics = lapply(thr_seq, function(x) get_metrics(x, test_probs, Employee_test$LeaveOrNot, "Yes", "No"))
metrics_df = data.frame(do.call(rbind, metrics))

# ROC curve
ggplot(aes(x = FPR, y = TPR), data = metrics_df) + 
  geom_line() +
  ylab("TPR (Sensitivity)") + 
  xlab("FPR (1-Specificity)") + 
  geom_abline(intercept = 0, slope = 1, linetype = 'dotted') +
  annotate("text", x = 0.75, y = 0.25, 
           label = paste("AUC:", round(metrics_df$AUC_ROC[1], 2))) +
  theme_bw()

# Lift
rf_oos_lift = caret::lift(Employee_test$LeaveOrNot ~ test_probs[, 1])

ggplot(rf_oos_lift) + 
  geom_abline(slope = 1, linetype = 'dotted') +
  xlim(c(0, 100)) + 
  theme_bw()

# Calibration
rf_cal = caret::calibration(Employee_test$LeaveOrNot ~ test_probs[, 1], 
                            data = best_preds, cuts = 11)
ggplot(rf_cal) + theme_bw()

# Extract out-of-sample predicted probs for the optimal model
rf_perf = evalm(rffit, showplots = FALSE) 

rf_perf$roc + 
  ggtitle("ROC curve for Random Forest") +
  theme_bw() 

rf_perf$proc + 
  ggtitle("Precision/recall curve for Random Forest") +
  theme_bw() 

rf_perf$cc + 
  ggtitle("Calibration curve for Random Forest")


###########################################################################
# bagging using random forest
###########################################################################
################################################################################
## Plot: fit from random forests for three different number of trees in forest.
################################################################################

#--------------------------------------------------
# get rf fits for different number of trees
# note: to get this to work I had to use maxnodes parameter of randomForest!!!

# Define the grid of hyperparameters for tuning
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5, 6))

# Define the control for cross-validation
fit_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Get rf fits for different number of trees and save the predictions
ntreev <- c(100, 1000, 2000, 5000)
nset <- length(ntreev)
fmat <- matrix(0, nrow(Employee_train), nset)

# Loop through the number of trees
for (i in 1:nset) {
  cat("Fitting Random Forest with ntree = ", ntreev[i], "\n")
  
  # Train the model using caret
  rffit <- train(LeaveOrNot ~ ., data = Employee_train, 
                 method = "rf", 
                 trControl = fit_control,
                 tuneGrid = rf_grid,
                 ntree = ntreev[i], 
                 maxnodes = 300,
                 metric = "ROC",
                 verbose = FALSE)
  
  # Save predictions
  fmat[, i] <- predict(rffit, Employee_train)
}

# Plot OOB error using the last fitted model
par(mfrow = c(1, 1))
plot(rffit$finalModel)

# Variable importance plot
varImpPlot(rffit$finalModel)

# Predictions on test data
predictions_bagging <- predict(rffit, newdata = Employee_test)

# Confusion Matrix
confusion <- confusionMatrix(predictions_bagging, Employee_test$LeaveOrNot)
print(confusion)

# Plot Confusion Matrix
ggplot(as.data.frame(confusion$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "blue") +
  ggtitle("Confusion Matrix") +
  theme_minimal()

# ROC Curve
pred_probs <- predict(rffit, newdata = Employee_test, type = "prob")[,2]
roc_curve <- roc(Employee_test$LeaveOrNot, pred_probs)
plot(roc_curve, col = "blue")
cat("AUC: ", auc(roc_curve), "\n")

# Precision-Recall Curve
pr_curve <- pr.curve(scores.class0 = pred_probs, weights.class0 = as.numeric(Employee_test$LeaveOrNot) - 1, curve = TRUE)
plot(pr_curve, main = "Precision-Recall Curve")

