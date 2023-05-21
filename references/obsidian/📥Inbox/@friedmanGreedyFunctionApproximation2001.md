
title: Greedy function approximation: A gradient boosting machine.
authors: Jerome H. Friedman
year: 2001
tags :  #supervised-learning #gbm #decision #gbm 
status : #📥  
related: 
- [[@friedmanGreedyFunctionApproximation2001]]
- [[@hastietrevorElementsStatisticalLearning2009]]

# Notes

## Notes from StatsQuest video
https://www.youtube.com/watch?v=StWY5QWMXCw

Make on prediction for all samples
Start with calculating the log-likelihood between the prediction and the true probability. As log-liklihood is used as loss-function -1 is multiplied with. Also no sum, as we look only at one sample at a time.

## Regularisation

Introducing shrinkage into gradient boosting (36) in this manner provides two regularisation parameters, the learning rate $\nu$ and the number of components $M$. Each one can control the degree-of-fit and thus affect the best value for other one. Decreasing the value of $\nu$ increases the best value for $M$. Ideally one should estimate optimal values for both by minimising a model selection criterion jointly with respect to the values of the two parameters. There are also computational considerations; increasing the size of $M$ produces a proportionate increase in computation.