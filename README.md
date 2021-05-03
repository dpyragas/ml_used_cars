# [Python] ML Regression model to predict used cars prices
### Libraries used: 
* pandas
* numpy
* pyplot
* seaborn
* sklearn

## Aim
The aim of this project is to solve the problem of predicting the price of a used car using supervised machine learning techniques, using Scikit-Learn libraries with Pandas and Seaborn libraries to help prepare and visualize the data. As it is clear from the beginning, it is a regression problem, where we will be using linear regression algorithm and will compare its effectiveness with Lasso, Ridge and ElasticNet regression algorithms to see which produces best results using r2 and Mean Squared Error scoring metrics.

## Objectives
1. Research, examine and outline the concept of Machine Learning.
2. Explore and evaluate the dataset and formulate a problem within.
3. Performing Exploratory Data Analysis and data cleaning, preparation to build a ML model.
4. Build and evaluate different ML models, tune hyperparameters and compare the performance using different scoring metrics.

## Dataset
In this project we are using [100 000 UK Used Car Data Set](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes). It is a scraped data of used car listings which is separated into files corresponding to each car manufacturer, which of them we will be merging into one dataset. Data consists of 99 186 variables, with 9 columns in total and later added 10th column – make.

## Problem
Used car sales in United Kingdom take a huge part in buying cars. In 2019 there were 7.9 million transactions (SMMT, 2019) showing that buying new cars is not always the best option for UK buyers as there were 2 311 140 new cars registered in 2019 (Vehicle Licensing Statistics, 2019). Therefore, it is in interest of vendors to predict used car value with a certain degree of accuracy. The problem is not easy to solve as the car’s value depends on many factors such as make, the year it was made, the mileage it was driven, the size of engine, the type of transmission and so on. Not all of the features have same importance, some are more important than others and therefore is essential to identify the most important ones, on which to perform the analysis.

## Model
### Summary of the aproach
In this project we will be using regression algorithm as mentioned before. It is a type of supervised learning model to predict continuous outcomes. In regression analysis we are given a number of predictor variables and a continuous response variable and we try to find a relationship between those variables that allows us to predict an outcome.
We will be using linear regression, it explores the linear relationship between observations and targets and the relationship are represented in a linear equation or weighted sum function.

## Conclusion
By comparing all the models, we can see that the Linear Regression model performed the best closely followed by others. We could have been missing other important factors, which could make the used car price higher or lower, such as a model of the car, as we know some models despite the mileage or other factors are rarer and could be more expensive etc. Or we might be missing information about the car quality itself, might be damaged, might be missing parts. To improve the model we could more outliers, use different kind of data transformations or we could even use different set of variables.
![Final](https://i.ibb.co/2ym1sLL/image.png "Model performance comparison")
