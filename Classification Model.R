
# Load the library --------------------------------------------------------

library(tidyverse)
library(corrplot)
library(caret)
library(glmnet)
library(randomForest)
library(pROC)
library(pdp)


# Import the dataset ------------------------------------------------------

repurchase_t <- read_csv('data/repurchase_training.csv')


# Look at the data structure ----------------------------------------------

glimpse(repurchase_t)


# Convert the date type ---------------------------------------------------
repurchase_t$age_band <- as.factor(repurchase_t$age_band)
repurchase_t$gender<- as.factor(repurchase_t$gender)
repurchase_t$car_model<- as.factor(repurchase_t$car_model)
repurchase_t$car_segment<- as.factor(repurchase_t$car_segment)

# Check the data type again -----------------------------------------------

glimpse(repurchase_t)

# Look at the first few rows ----------------------------------------------

head(repurchase_t)


# Check the missing values for each variable ------------------------------

cbind(
  lapply(
    lapply(repurchase_t, is.na),sum)
)


# Look at the statistical description -------------------------------------

summary(repurchase_t)


# Exploratory Data Analysis -----------------------------------------------

# Relationship between Target and categorical variables(bar charts)

# Target value by customer age
repurchase_t %>% 
  ggplot()+
  geom_bar(mapping=aes(x=age_band, fill = as.factor(Target)))+
  labs(caption ="Figure 1. Target value by customer age band",
       y="Number of records",
       x="",
       fill="Target")+
  scale_fill_manual(values=c("#FFA07A","#1E90FF"))+
  theme_bw()+
  theme(plot.caption = element_text(size=11,hjust=0.5,face="bold"),
        legend.position = c(0.2,0.9),
        legend.direction = "horizontal")


# Target value by customer gender
ggplot(repurchase_t)+
  geom_bar(mapping=aes(x=gender, fill = as.factor(Target)))+
  labs(caption ="Figure 2. Target value by customer gender",
       y="Number of records",
       x="",
       fill="Target")+
  scale_fill_manual(values=c("#FFA07A","#1E90FF"))+
  theme_bw()+
  theme(plot.caption = element_text(size=11,hjust=0.5,face="bold"),
        legend.position = c(0.25,0.85),
        legend.direction = "horizontal")


# Target value by car model
repurchase_t %>% 
  count(car_model,Target) %>% 
  ggplot()+
  geom_col(mapping=aes(x=reorder(car_model,-n), y=n,fill = as.factor(Target)))+
  labs(caption ="Figure 3. Target value by car model",
       y="Number of records",
       x="",
       fill="Target")+
  scale_fill_manual(values=c("#FFA07A","#1E90FF"))+
  theme_bw()+
  theme(plot.caption = element_text(size=11,hjust=0.5,face="bold"),
        legend.position = c(0.9,0.75),
        axis.text.x=element_text(angle=90))

# Target value by car segment

repurchase_t %>% 
  count(car_segment,Target) %>% 
ggplot()+
  geom_col(mapping=aes(x=reorder(car_segment,-n),y=n, fill = as.factor(Target)))+
  labs(caption ="Figure 4. Target value by car segment",
       y="Number of records",
       x="",
       fill="Target")+
  scale_fill_manual(values=c("#FFA07A","#1E90FF"))+
  theme_bw()+
  theme(plot.caption = element_text(size=11,hjust=0.5,face="bold"),
        legend.position = c(0.9,0.8))


# Relationship between Target and numeric variables (correlation matrix)

col_matrix <- repurchase_t[,-c(1,3,4,5,6)]
M <- cor(col_matrix)
corrplot(M,type="upper",method="color",addCoef.col = "tomato",tl.col = "tomato")


# Feature Selection -------------------------------------------------------

# Lasso Regression

repurchase_t$Target <- as.factor(repurchase_t$Target)

repurchase_t <- repurchase_t[-1]

set.seed(123)

training.samples <- createDataPartition(repurchase_t$Target,p=0.8,list=FALSE)

train.data <- repurchase_t[training.samples, ]
test.data <- repurchase_t[-training.samples, ]

x = model.matrix(~ ., train.data[, -1])
y = train.data$Target

cv.lasso <- cv.glmnet(x,y,family = 'binomial', alpha = 1)
plot(cv.lasso)

cv.lasso$lambda.min

coef(cv.lasso, s = cv.lasso$lambda.min)

prediction_lasso = predict(cv.lasso$glmnet.fit, newx = model.matrix(~ ., test.data[, -1]), 
                           type = "class",
                           s = cv.lasso$lambda.min)

#confusion matrix
cfm <- table(predicted = prediction_lasso, true=test.data$Target)
cfm

#accuracy
accuracy <- (cfm[1,1]+cfm[2,2])/(cfm[1,1]+cfm[1,2]+cfm[2,1]+cfm[2,2])
accuracy

#precision
precision <- cfm[1,1]/(cfm[1,1]+cfm[1,2])
precision

#recall
recall <- cfm[1,1]/(cfm[1,1]+cfm[2,1])
recall

#F1
f1 <- 2*(precision*recall/(precision+recall))
f1

#AUC
prediction_lasso <- as.numeric(prediction_lasso)
par(pty = "s")
auc(test.data$Target,prediction_lasso,plot=TRUE,legacy.axes=TRUE,
    xlab="False Positive Rate",
    ylab="True Positive Rate",
    col="#FF6347",
    lwd=4,
    print.auc=TRUE,
    percent=TRUE)

#Variable Importance (Coefficient)

lasso_imp <- coef(cv.lasso, s = cv.lasso$lambda.min)
var_name <- names(lasso_imp[, 1][lasso_imp[, 1] != 0])[28:38]
var_imp <- lasso_imp[, 1][lasso_imp[, 1] != 0][28:38]
var_imp <- as.data.frame(var_imp)
var_name <- as.data.frame(var_name)
lasso_vimp <- cbind(var_name,var_imp)

lasso_vimp %>% 
  mutate(pos=var_imp>=0) %>% 
ggplot()+
  geom_col(mapping=aes(x=var_name, y=var_imp,fill=pos))+
  labs(y="Variable Importance",
       x="")+
  scale_fill_manual(values=c("#FF6347","#1E90FF"))+
  theme_bw()+
  theme(axis.text.x=element_text(angle=90),
        legend.position = "none")


# Random forest -----------------------------------------------------------

# Set up Training and Test sets
set.seed(123)
train_test_split <- createDataPartition(repurchase_t$Target,p=0.8,list=FALSE)
training_data <- repurchase_t[train_test_split,] #this is used to build the training model
test_data <- repurchase_t[-train_test_split,]

#10-fold cross validataion
cvSplits <- createFolds(training_data$Target,k=10,)
#notice the number of elements in each fold
str(cvSplits)

#initialise accuracy vector
accuracies <- rep(NA,length(cvSplits))
i <- 0
#loop over all folds
for (testset_indices in cvSplits){
  i <- i+1
  trainset <- repurchase_t[-testset_indices, ]
  testset <- repurchase_t[testset_indices, ]
  rf <- randomForest(Target ~.,data = trainset, 
                       importance=TRUE, xtest=testset[,-1],ntree=100)
  
  # Accuracy on test data
  accuracies[i] <- mean(rf$test$predicted==testset$Target)
  
}

#a more unbiased error estimate (note the sd)
mean(accuracies)
sd(accuracies)

#Build final model on entire training data set (all folds)
rf <-randomForest(Target ~.,data = training_data, keep.forest=TRUE,
                        importance=TRUE, xtest=test_data[,-1],ntree=100)

plot(rf)

#out of sample error
mean(rf$test$predicted==test_data$Target)

# Model summary
# Not super useful for model analysis
summary(rf) 

# Objects returned from the model 
names(rf)

# Confusion matrix
cfm_rf <- table(rf$test$predicted,test_data$Target)
cfm_rf

#precision
precision_rf <- cfm_rf[1,1]/(cfm_rf[1,1]+cfm_rf[1,2])
precision_rf

#recall
recall_rf <- cfm_rf[1,1]/(cfm_rf[1,1]+cfm_rf[2,1])
recall_rf

#F1
f1_rf <- 2*(precision_rf*recall_rf/(precision_rf+recall_rf))
f1_rf

#AUC

rf$test$predicted <- as.numeric(rf$test$predicted)
par(pty = "s")
auc(test_data$Target,rf$test$predicted,plot=TRUE,legacy.axes=TRUE,
    xlab="False Positive Rate",
    ylab="True Positive Rate",
    col="Blue",
    lwd=4,
    print.auc=TRUE,
    percent=TRUE)

# Quantitative measure of variable importance
importance(rf)

# Sorted plot of importance
varImpPlot(rf)



#ROC-AUC (Logistic vs Random Forest) credit to Josh Starmer: https://www.youtube.com/watch?v=qcvAqAH60Yw

par(pty = "s")
auc(test.data$Target,prediction_lasso,plot=TRUE,legacy.axes=TRUE,
    xlab="False Positive Rate (1-Specificity) ",
    ylab="True Positive Rate (Sensitivity)",
    col="#FF6347",
    lwd=4,
    print.auc=TRUE,
    percent=TRUE)

plot.roc(test_data$Target,rf$test$predicted,percent=TRUE,col="#1E90FF",lwd=4,
         print.auc=TRUE,add=TRUE,print.auc.y=40)

legend("bottomright",legend = c("Logistic Regression","Random Forest"),
       col=c("#FF6347","#1E90FF"),lwd=4)


# Partial dependency plots



grid.arrange(
  partial(rf, pred.var = "mth_since_last_serv", plot = TRUE, rug = TRUE, 
          type="classification", prob=TRUE, parallel=F, which.class = "1", train=training_data),
  partial(rf, pred.var = "annualised_mileage", plot = TRUE, rug = TRUE, 
          type="classification", prob=TRUE, parallel=F, which.class = "1", train=training_data),
  partial(rf, pred.var = "gender", plot = TRUE, rug = TRUE, 
          type="classification", prob=TRUE, parallel=F, which.class = "1", train=training_data),
  partial(rf, pred.var = "age_of_vehicle_years", plot = TRUE, rug = TRUE, 
          type="classification", prob=TRUE, parallel=F, which.class = "1", train=training_data),
  partial(rf, pred.var = "num_serv_dealer_purchased", plot = TRUE, rug = TRUE, 
          type="classification", prob=TRUE,parallel=F, which.class="1",train=training_data),
  ncol = 3
)



#Build final model on entire training data set

repurchase_t$age_band <- as.character(repurchase_t$age_band)
repurchase_t$gender<- as.character(repurchase_t$gender)
repurchase_t$car_model<- as.character(repurchase_t$car_model)
repurchase_t$car_segment<- as.character(repurchase_t$car_segment)

rf_all <-randomForest(Target ~.,data = repurchase_t, keep.forest=TRUE,
                  importance=TRUE,ntree=100)



repurchase_v <- read_csv('data/repurchase_validation.csv')


pred_target <- rf_all %>% predict(repurchase_v[-1])

pred_prob <- rf_all %>% predict(repurchase_v[-1],type="prob")


pred_validation <- 
  repurchase_v %>% 
  mutate(target_probability=pred_prob[,2],target_class=pred_target)%>% 
  select(target_probability,target_class) %>% 
  mutate(ID=row_number()) %>% 
  select(ID,target_probability,target_class)

pred_validation

write.csv(pred_validation,file="repurchase_validation_14169837.csv",row.names = FALSE)
  


