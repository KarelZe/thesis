
title: Greedy function approximation: A gradient boosting machine.
authors: Jerome H. Friedman
year: 2001


## General Gradient Boosting

## Gradient Boosting for regression trees


## Regularization

Introducing shrinkage into gradient boosting (36) in this manner provides two regularization parameters, the learning rate $\nu$ and the number of components $M$. Each one can control the degree-of-fit and thus affect the best value for other one. Decreasing the value of $\nu$ increases the best value for $M$. Ideally one should estimate optimal values for both by minimizing a model selection criterion jointly with respect to the values of the two parameters. There are also computational considerations; increasing the size of $M$ produces a proportionate increase in computation.