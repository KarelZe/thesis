
title: An Introduction to Statistical Learning
authors: Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani
year: 2012


Tags: #gradient_boosting

## Boosting

- Boosting approaches learn slowly. Instead of fitting trees hard to the data, the  model is fit to the residuals of the model. A new decission tree is added into the fitted function to update the residuals. (James et. al; 2013)
- The learning rate slows the learning process down, allowing more and ifferent shaped trees to attack the residuals.
- Unlike bagging, the construction of trees is dependent on trees already grown.