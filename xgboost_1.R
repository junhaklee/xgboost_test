# Load packages
library(readxl)
library(tidyverse)
library(xgboost)
library(caret)

# Read Data
setwd("D:/70. R/xgboost/CCPP/")

power_plant = as.data.frame(read_excel("Folds5x2_pp.xlsx"))

set.seed(100)  # For reproducibility

# Create index for testing and training data
inTrain <- createDataPartition(y = power_plant$PE, p = 0.8, list = FALSE)
# subset power_plant data to training
training <- power_plant[inTrain,]
# subset the rest to test
testing <- power_plant[-inTrain,]


# Convert the training and testing sets into DMatrixes: DMatrix is the recommended class in xgboost.
X_train = xgb.DMatrix(as.matrix(training %>% select(-PE)))
y_train = training$PE
X_test = xgb.DMatrix(as.matrix(testing %>% select(-PE)))
y_test = testing$PE


#Specify cross-validation method and number of folds. Also enable parallel computation
xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE
)


#This is the grid space to search for the best hyperparameters
#I am specifing the same parameters with the same values as I did for Python above. The hyperparameters to optimize are found in the website.
xgbGrid <- expand.grid(nrounds = c(100,200),  # this is n_estimators in the python code above
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)

# Finally, train your model
set.seed(0) 
xgb_model = train(
  X_train, y_train,  
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  method = "xgbTree"
)


# Model evaluation

predicted = predict(xgb_model, X_test)
residuals = y_test - predicted
RMSE = sqrt(mean(residuals^2))
cat('The root mean square error of the test data is ', round(RMSE,3),'\n')

# The root mean square error of the test data is 2.856 

y_test_mean = mean(y_test)
# Calculate total sum of squares
tss =  sum((y_test - y_test_mean)^2 )
# Calculate residual sum of squares
rss =  sum(residuals^2)
# Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')

#Plotting actual vs predicted
options(repr.plot.width=8, repr.plot.height=4)
my_data = as.data.frame(cbind(predicted = predicted,
                              observed = y_test))
# Plot predictions vs test data
ggplot(my_data,aes(predicted, observed)) + geom_point(color = "darkred", alpha = 0.5) + 
  geom_smooth(method=lm)+ ggtitle('Linear Regression ') + ggtitle("Extreme Gradient Boosting: Prediction vs Test Data") +
  xlab("Predecited Power Output ") + ylab("Observed Power Output") + 
  theme(plot.title = element_text(color="darkgreen",size=16,hjust = 0.5),
        axis.text.y = element_text(size=12), axis.text.x = element_text(size=12,hjust=.5),
        axis.title.x = element_text(size=14), axis.title.y = element_text(size=14))

