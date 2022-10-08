---
title: Gradient Boost Part 1 (of 4): Regression Main Ideas - YouTube
authors: StatQuest
year: 2019
---

- Ada-Boost only uses stumps, Gradient Boosting approaches
- Gradient Boosting starts with a single leaf instead of a tree. Leaf represents inital guess for all weights. First guess is average value
- Gradient Boosting builds a tree based on the errors
- Gradien Boosting builds trees with fixed size, but not only stumps. Gradient Boost also scales the trees based on the errors.

## Example
- First the average is calculated
- Based on the errors of the leaf a tree is built. The error is called pseudo residual. It is basically the difference between the mean and the true value. Different features are used for splitting criterion of the tree.
- The residuals in the leaf are replaced with the leaf's average.
- Learning rate is used to scale the contibution from a new tree e. g. 71.2 + learning rate * leaf's mean. Using the learning rate reduces the variance. The learning rate is the same for all trees.
- One does than calculate new residuals from the improved prediction with the learning rate. All residuals are smaller than before after adding a new tree. For each tree the branches can be different.
