## Load Libraries
library(dplyr)
library(ggplot2)
library(caret)
library(mlbench)
library(corrplot)
library(randomForest)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(data.tree)
library(caTools)
library(MLmetrics)
library(e1071)

## Load CSV file 
df<-read.csv("C:/Users/Shyam/Downloads/titanic.csv")
head(df)

## Check for null values
sum(is.na(df))

## Select and factor features
df<-select(df,Survived, Pclass, Sex, Age)

df$Survived<-as.factor(df$Survived)
df$Sex<-as.factor(df$Sex)
df$Pclass<-as.factor(df$Pclass)

## Plotting charts
## Passengers Survived Chart 
ggplot(df, aes(x = Survived, fill = Survived, color = Survived))+
  stat_count(position = "identity", alpha = 0.5) +
  labs(title = "Passenger Survived", x = "Survived", y = "Count")

## Passengers Pclass Chart 
ggplot(df, aes(x = Pclass, fill = Pclass, color = Pclass))+
  stat_count(position = "identity", alpha = 0.5) +
  labs(title = "Passenger Class", x = "Class", y = "Count")

## Gender
ggplot(df, aes(x = Sex, fill = Sex, color = Sex))+
  stat_count(position = "identity", alpha = 0.5) +
  labs(title = "Passenger Gender", x = "Gender", y = "Count")

## Finding Means
## Average age of passengers that survived and did not survive
mean(df$Age[df$Survived == "1"])
mean(df$Age[df$Survived == "0"])

## Average age for each gender
mean(df$Age[df$Sex == "male"])
mean(df$Age[df$Sex == "female"])

## Average age of passengers for each passenger class
mean(df$Age[df$Pclass == "1"])
mean(df$Age[df$Pclass == "2"])
mean(df$Age[df$Pclass == "3"])

## Set seed, train and test using 20/80 sample
set.seed(123)
indexes<-sample(1:nrow(df), size = 0.2 * nrow(df))
test<-df[indexes,]
train<-df[-indexes,]

## Create machine learning models
# Model 1: Logistic Regression
LR<-glm(Survived ~ ., data = train, family = "binomial")
y_predlr<-predict(LR, test, type = "response")
y_predlr<-round(y_predlr)
y_predlr<-as.factor(y_predlr)
confusionMatrix(data=y_predlr, reference = test$Survived)

# Model 2: Decision Tree
tree<-rpart(Survived~.,data = train)
y_predtr<-predict(tree, test, type="class")
confusionMatrix(data = y_predtr, reference = test$Survived)
prp(tree)

# Model 3: Random Forest
rf<-randomForest(Survived~., data = train)
y_predrf<-predict(rf, test, type = "class")
confusionMatrix(data = y_predrf, reference = test$Survived)

# Model 4: Support Vector Machine
svm.model<-svm(Survived ~ ., data = train, kernel ="linear")
y_pred<-predict(svm.model, test, type = "class")
confusionMatrix(data = y_pred, reference = test$Survived)