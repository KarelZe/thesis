
title: Bagging predictors
authors: Leo Breiman
year: 1996
tags :  #bagging #dt #decision-trees
status : #📥  
related: 
- [[@friedmanGreedyFunctionApproximation2001]]
- [[@hastietrevorElementsStatisticalLearning2009]]
- [[@breimanRandomForests2001]]
- [[@hintonImprovingNeuralNetworks2012]] (Bagging is similar to dro-pout found in [[@leePseudolabelSimpleEfficient]])

Tags: #bagging #dt #decision-trees

Bagging is an ensemble method, where each tree is grown in from a random selection (without replacement) of examples in the training set. ([[@breimanRandomForests2001]], p. 1)

Bagging stands for Bootstrap aggregating.

## Main idea

Use a learning set $\mathcal{L}$ to get better estimates for a predictor compared to a single learning set predictor $\varphi(x, \mathcal{L})$

If $y$ is numerical, an obvious procedure is to replace $\varphi(x, \mathcal{L})$ by the average of $\varphi\left(x, \mathcal{L}_{k}\right)$ over $k$, i.e. by $\hat{A}_{A}(x)=E_{\mathcal{L}} \psi(x, \mathcal{L})$, where $E_{\mathcal{L}}$ denotes the expectation over $\mathcal{L}$, and the subscript $A$ in $\varphi_{A}$ denotes aggregation.

If no replicates of $\mathcal{L}$ are available, bootstrapping (random sampling with replacement) to create new bootstraped samples $\left\{\mathcal{L}^{(B)}\right\}$ from $\mathcal{L}$.

A predictor could then be trained on the bootstraped sample. It has the form of $\left\{\varphi\left(x, \mathcal{L}^{(B)}\right)\right\}$. If $y$ is numerical, take $\varphi_B$ as:
$$\varphi_{B}(x)=\operatorname{av}_{B} \varphi\left(x, \mathcal{L}^{(B)}\right),$$

with $\operatorname{av}$ being the average over all estimates. (p. 123)

## Bootstrap samples

The $\left\{\mathcal{L}^{(B)}\right\}$ form replicate data sets, each consisting of $N$ cases, drawn at random, but with replacement, from $\mathcal{L}$. Each $\left(y_{n}, x_{n}\right)$ may appear repeated times or not at all in any particular $\mathcal{L}^{(B)}$.

The $\left\{\mathcal{L}^{(B)}\right\}$ are replicate data sets drawn from the bootstrap distribution approximating the distribution underlying $\mathcal{L}$. (p. 124)


## Bagging for regression

1. randomly divide into test set $\mathcal{T}$ and a learning set $\mathcal{L}$.
2. A regression tree is contstructed in the learning set $\mathcal{L}$ using 10-fold CV. Squared error is calculated.
3. A boostrap sample $\mathcal{L}_B$ is selected from $\mathcal{L}$ and a tree is grown using $\mathcal{L}_B$. $\mathcal{L}$ is used for selecting the pruned subtree. procedure is repeated several times.
4. For $\left(y_{n}, \boldsymbol{x}_{n}\right) \in \mathcal{T}$, the bagged predictor is $\hat{y}_{n}=a v_{k} \phi_{k}\left(\boldsymbol{x}_{n}\right)$, and the squared error $e_{B}(\mathcal{L}, \mathcal{T})$ is $a v_{n}\left(y_{n}-\hat{y}_{n}\right)^{2}$ (average?)
5. The random division of the data into $\mathcal{L}$ and $\mathcal{T}$ is repeated multiple times and the errors averaged to give $\bar{e}_{S}, \bar{e}_{B}$. (p. 128)

**Summary**: Random splits into train and test set. A bootstrap sample is used to build trees, which gets then pruned. The bagged predictor is the average of the predictors learnt on the bootstrap samples.

## Discussion

- relatively easy to implement, as it is just a matter of adding a loop in front, select the bootstrap sample and send it to the procedure and finally do the aggregation
- Improved Accuracy
- Loss of the simple and interpretable structure (p. 137)