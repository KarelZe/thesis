
Naturally, one would like to obtain insights into how the models arrived at the prediction and identify features relevant for the prediction. Both aspects can be subsumed under the term *interpretability*. Following, ([[@liptonMythosModelInterpretability2017]]4) interpretability can be reached through model transparency or post-hoc interpretability methods. Transparent models provide interpretability through a transparent mechanism in the model, whereas post-hoc interpretability refers to approaches that extract information from the already learned model ([[@liptonMythosModelInterpretability2017]] 4--5). 

Classical trade classification algorithms, as a rule-based approach, are transparent with an easily understandable decision process, and thus provide interpretability ([[@barredoarrietaExplainableArtificialIntelligence2020]]91). Interpretability, however decreases for deep stacked combinations involving a large feature count, such as the gls-GSU method, interactions between base rules become more complex, and the effect of single feature on the final prediction more challenging to interpret. 

The machine-learning classifiers, studied in this work, can be deemed a black box model ([[@barredoarrietaExplainableArtificialIntelligence2020]]90). Due to the sheer size of the network or ensemble, interpretability through transparency is impacted. Albeit, the attention mechanism of Transformers provides some interpretability through transparency (see discussion on attention maps),  interpretability across all classifiers can only be reached through a *model-agnostic, post-hoc interpretability techniques*. Thereby, our goal is to identify features that are important for the *correct prediction*. This is fundamentally different from methods like standard gls-SHAP, that attribute *any* prediction to the input features ([[@chenTrueModelTrue2020]]??).

Many model-agnostic methods are based on the randomly permuting features values. In this work, we specifically consider the variants *permutation feature importance* ([[@breimanRandomForests2001]]23--24) and partial-dependence plots ([[@friedmanGreedyFunctionApproximation2001]]26--28). Both serve a complementary purpose. Permutation feature importance derives the feature importance from the change in predictive accuracy before and after permuting a feature randomly, whereas partial dependence plots visualize the average change in prediction, if feature values are altered. These are widely adopted and computationally efficient.

### Permutation feature importance

Permutation feature importance derives the importance from the mean decrease in accuracy before and after permuting a feature randomly. Expectedly, permuting features breaks the association with the target. Thus, the permutation of important features leads to a sharp decrease in accuracy, whereas unimportant features leave the accuracy unaffected. 

The importance measure was originally proposed ([[@breimanRandomForests2001]]23--24) for random forests, and has later been extended by ([[@fisherAllModelsAre]]) into a model-agnostic feature importance measure. 

Given our feature matrix $\mathbf{X}$, we can define a second permuted version, $\mathbf{X}^{\pi,j}$, where the $j$-th feature is randomly permuted by $\pi$. Using $L(y_i, h(\mathbf{x}_i))$ for predicting $y_i$ from $h(\mathbf{x}_{i})$, the importance of $j$-th feature is given by:
$$
\operatorname{VI}^{\pi}_{j} = \sum_{i=1}^{N} L(y_{i}, h(\mathbf{x}_{i}^{\pi,j})) - L(y_{i}, h(\mathbf{x}_{i})),
$$
which is the increase in loss, i.e., accuracy before and after permutation ([[@hookerUnrestrictedPermutationForces2021]]82). While ([[@breimanRandomForests2001]]23--24) uses a single permutation, ([[@fisherAllModelsAre]]??) consider multiple, random permutations. By definition, random feature importance only yields global feature importances, as the change in accuracy is averaged from all $N$ samples. (averaged????)

### Extending permutation feature importance ✅
Random feature permutation has the desirable properties of being easy to interpret, computationally efficient and model-agnostic. Like other feature importance measures, including SHAP or LIME, it assumes independence between features ([[@aasExplainingIndividualPredictions2021]]2). 

As defined in cref-eq-random-feature-permutation, every feature is permuted independently from other features which artificially breaks correlations between features and creates unrealistic feature combinations. Consider, for example, the apparent correlation between the ask, bid price and trade price. Permuting only the ask, could result in strongly negative or extremely large spreads, whereas bid and trade price remain unchanged. In effect, the presence of correlated features, leads to an overestimate of the importance of correlated features ([[@stroblConditionalVariableImportance2008]]3). 

Vice versa, can the presence of a correlated features decrease the importance of the associated feature, as the feature importance now distributed across the features, thereby underestimating the true importance of the features. This effects all features, where information is encoded redundantly, such as the bid-ask ratio. (footnote-for an extended discussion of substitution effects on feature importance in the financial domain see ([[@lopezdepradoAdvancesFinancialMachine2018]]114--118).

To alleviate the bias from correlated / depenedent, we group dependent features and estimate the feature importance on the group-level. Arranging all features in a tree-like hierarchy gives us the freedom to derive feature importances at different levels, enabling cross-comparisons between classical rules and machine learning based classifiers, as grouping of raw and derived features makes the implementation of classical rules transparent. (footnote: Consider the implementation of the tick rule. Here, the implementation could use the feature price lag (ex) or calculate the price change from the trade price and price lag (ex). If not grouped, feature importances would be attributed to either the derived feature or raw features causing difficulties in comparison with machine learning classifiers, which have access to all three features simultaneously. Grouping all three features resolves this issue at the cost of interpretability.). Other than the classical permutation importance from cref-eq-random-feature-permutation, all features sharing the same parent node are permuted together. We define the following dependency structure:

```mermaid

graph TB 
A((1))-->B((2))
A-->C((3))
A-->D((4))
B-->E((5)) 
B-->F((6))
B-->G((7))
C-->H((8))
D-->I((9))
D-->J((10))
```
Groupings are created to be mutually exclusive and based on the dependency structure of classical trade classification algorithms. The computational demand is comparable to classical feature permutation, as grouping results in fewer permutations, but the analysis may be repeated on several sub-levels. 

To this end, we want to emphasize, that our approach is different from ([[@ronenMachineLearningTrade2022]]52) as we do not estimate the improvement from adding new features, but keep the feature sets fixed-sized and permute them.

### Partial dependence plots
Related to the concept of random feature permutation are partial dependence plots by ([[@friedmanGreedyFunctionApproximation2001]] 26--28). These visualize the dependency between a single (or multiple) feature and the predicted target as the feature value is adjusted. 

Following ([[@hookerUnrestrictedPermutationForces2021]]81), we newly define a feature matrix $\mathbf{X}^{x,j}$ from the feature matrix $\mathbf{X}$, where the value of the $j$-th feature is replaced by the value $x$. The partial dependence function for the $j$-th feature is now given by:
$$
\operatorname{PD}_{j}(x) = \frac{1}{N} \sum_{i=1}^{N} f(\mathbf{x}^{x,j}_{i}).
$$
$PD_{j}(x)$ now gi . Repeating 


Like random feature permutation, partial dependence plots are a global feature importance measure, unable to capture dependencies between features. Naturally, visualization is constrained to two dimensions or features at once. Despite these limitation, partial dependence plots to help us verify the assumed relationships in classical rules, such as the the linear relationship in the tick rule, with the learned relationships in our classifier.

> 📑The marginal Partial Dependence Plot (PDP) (Friedman et al., 2001) describes the average effect of the j-th feature on the prediction. P DPj (x) = E[ ˆf(x, X−j )], (3) If the expectation is conditional on Xj , E[ ˆf(x, X−j )|Xj = x], we speak of the conditional PDP. The marginal PDP evaluated at feature value x is estimated using Monte Carlo integration: P DP \j (x) = 1 n Xn i=1 ˆf(x, x (i) −j ) (4)  (https://arxiv.org/pdf/2006.04628.pdf) 

> 📑 Partial Dependence Plots (PDPs) Friedman (2001) suggested examining the effect of feature j by plotting the average prediction as the feature is changed. Specifically, letting Xx,j be the matrix of feature values where the jth entry of every row has been replaced with value x, we define the partial dependence function PDj(x) = 1 N N i=1 f (xx,j i ) as the average prediction made with the jth feature replaced with the value x. Since these are univariate functions (multivariate versions can be defined naturally), they can be readily displayed and interpreted. ([[@hookerUnrestrictedPermutationForces2021]])

> 📑Partial dependence works by marginalizing the machine learning model output over the distribution of the features in set C, so that the function shows the relationship between the features in set S we are interested in and the predicted outcome. By marginalizing over the other features, we get a function that depends only on features in S, interactions with other features included. (Molnar)

> 📑The partial function ^fS�^� is estimated by calculating averages in the training data, also known as Monte Carlo method:
For classification where the machine learning model outputs probabilities, the partial dependence plot displays the probability for a certain class given different values for feature(s) in S. An easy way to deal with multiple classes is to draw one line or plot per class. (Molnar)

> 📑 The partial dependence plot (short PDP or PD plot) shows the marginal effect one or two features have on the predicted outcome of a machine learning model (J. H. Friedman 2001[30](https://christophm.github.io/interpretable-ml-book/pdp.html#fn30)). A partial dependence plot can show whether the relationship between the target and a feature is linear, monotonic or more complex. For example, when applied to a linear regression model, partial dependence plots always show a linear relationship.

> 📑 “s, especially when f (x) is dominated by low-order interactions (10.40). Consider the subvector XS of ℓ < p of the input predictor variables XT = (X1, X2, . . . , Xp), indexed by S ⊂ {1, 2, . . . , p}. Let C be the complement set, with S ∪ C = {1, 2, . . . , p}. A general function f (X) will in principle depend on all of the input variables: f (X) = f (XS , XC). One way to define the average or partial dependence of f (X) on XS is fS (XS ) = EXC f (XS , XC). (10.47) This is a marginal average of f , and can serve as a useful description of the effect of the chosen subset on f (X) when, for example, the variables in XS do not have strong interactions with those in XC. Partial dependence functions can be used to interpret the results of any “black box” learning method. They can be estimated by ̄ fS (XS ) = 1 N N ∑ i=1 f (XS , xiC), (10.48) where {x1C, x2C, . . . , xNC} are the values of XC occurring in the training data. This requires a pass over the data for each set of joint values of XS for which ̄ fS (XS ) is to be evaluated. This can be computationally intensive, 1lattice in R” ([[@hastietrevorElementsStatisticalLearning2009]] p. 388)

“The partial dependence function (Friedman, 1991) of a model ˆ f describes the expected effect of a feature after marginalizing out the effects of all other features. Partial dependence of a feature set XS, S ⊆ {1, . . . , p} (usually |S| = 1) is defined as: P DS = EXC [ ˆ f (x, XC )], (1) where XC are the remaining features so that S ∪C = {1, . . . , p} and S ∩C = ∅. The PD is estimated using Monte Carlo integration: ̂ P DS(x) = 1 n2 n2 ∑ i=1 ˆ f (x, x(i) C ) (2) For simplicity, we write P D instead of P DS, and ̂ P D instead of ̂ P DS when we refer to an arbitrary PD. The PD plot consists of a line connecting the points {(x(g), ̂ P DS(x(g))}gG=1, with G grid points that are usually equidistant or quantiles of PXS . See Figure 6 for an example of a PD plot.” ([[@molnarRelatingPartialDependence2021]], p. 5)

Following a common track in literature, we report feature importance estimates on the test data. 