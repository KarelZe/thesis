### Interpretability vs explainability

Although it is common in the literature to see the terms "interpretability" and "explainability" used interchangeably,[17](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib17), [18](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib18), [19](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib19) they are distinct concepts, and their conflation can cause significant confusion.[20](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib20)

Although there is slight variability in the precise definition of the term "interpretable" in the literature,[21](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib21), [22](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib22), [23](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib23), [24](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib24) throughout this review we use the term to refer to models in which humans can directly understand how a model operates and the causes of its decisions.[25](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib25) Logistic-regression models are interpretable because a human can refer to the weights and odds ratios to understand how the model operates and can refer to coefficients to understand the cause of individual predictions. It should be noted that interpretability is at least somewhat subjective, as it can require expert knowledge of statistics or a domain (such as cardiology) to interpret a model’s decisions effectively.[8](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib8)




- Feature Importance of Gradient Boosted Trees
	- Possibilities to calculate feature importances in GBMs [here.](https://blog.tensorflow.org/2019/03/how-to-train-boosted-trees-models-in-tensorflow.html)

- Unified Approach for Feature Importance
	- Make feature importances comparable across models.
	- For simple methods see permutation importance, ice and coutnerfactuals (see [8.5 Permutation Feature Importance | Interpretable Machine Learning (christophm.github.io)](https://christophm.github.io/interpretable-ml-book/feature-importance.html))
	- Open question: How are correlations handled in SHAP?
	- Think about using kernel-shap. Could work. See e. g., [Feature importance in deep learning - Deep Learning - Deep Learning Course Forums (fast.ai)](https://forums.fast.ai/t/feature-importance-in-deep-learning/42026/91?page=4) and [Census income classification with Keras — SHAP latest documentation](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/neural_networks/Census%20income%20classification%20with%20Keras.html)
	- If SHAP is to complex, one could just zero-out features like in [[@guEmpiricalAssetPricing2020]], but be aware of drawbacks. Yet similar method is to randomly permutate features "within a column" and see how to prediction changes" (see [[@banachewiczKaggleBookData2022]]) also comes at the advantage that no retraining is needed, but artificially breaks correlations etc. (see my previous seminar paper).
	- [[@ronenMachineLearningTrade2022]] study the feature importance only for random forests.
	- For explanation on SHAP and difference between pre- and post-model explainability see [[@baptistaRelationPrognosticsPredictor2022]]


Compare different feature attributions:
![[feature_attributions_from_attention.png]]
(Found in [[@borisovDeepNeuralNetworks2022]])


[[@molnarRelatingPartialDependence2021]]

“Much of market microstructure analysis is built on the concept that traders learn from market data. Some of this learning is prosaic, such as inferring buys and sells from trade execution. Other learning is more complex, such as inferring underlying new information from trade executions. In this paper, we investigate the general issue of how to discern underlying information from trading data. We examine the accuracy and efficacy of three methods for classifying trades: the tick rule, the aggregated tick rule, and the bulk volume classification methodology. Our results indicate that the tick rule is a reasonably good classifier of the aggressor side of trading, both for individual trades and in aggregate. Bulk volume is shown to also be reasonably accurate for classifying buy and sell trades, but, unlike the tick-based approaches, it can also provide insight into other proxies for underlying information.” (Easley et al., 2016, p. 284)

---



- What is feature importance? How does it allign with interpretability? Explainability?

- common pitfalls: https://arxiv.org/pdf/2007.04131.pdf

- feature importance evaluation is a non-trivial problem due to missing ground truth. See [[@borisovDeepNeuralNetworks2022]] paper for citation
- https://mindfulmodeler.substack.com/p/shap-is-not-all-you-need

- Zachary C Lipton. 2016. The mythos of model interpretability. arXiv preprint arXiv:1606.03490 (difference between transparency and interpretability)

## SHAP vs. RFI
https://mindfulmodeler.substack.com/p/shap-is-not-all-you-need
Given the simulation setup where none of the features has a relation to the target, one could say that PFI results are correct and SHAP is wrong. But this answer is too simplistic. The choice of interpretation method really depends on what you use the importance values for. What is the question that you want to answer?

Because Shapley values are “correct” in the sense that they do what they are supposed to do: Attribute the prediction to the features. And in this case, changing the “important” features truly changes the model prediction. So if your goal tends towards understanding how the model “behaves”, SHAP might be the right choice.

But if you want to find out how relevant a feature was for the CORRECT prediction, SHAP is not a good option. Here PFI is the better choice since it links importance to model performance.

In a way, it boils down to the question of [audit versus insight](https://mindfulmodeler.substack.com/p/audit-or-insight-know-your-interpretation): SHAP importance is more about auditing how the model behaves. As in the simulated example, it’s useful to see how model predictions are affected by features X4, X6, and so on. For that SHAP importance is meaningful. But if your goal was to study the underlying data, then it’s completely misleading. Here PFI gives you a better idea of what’s really going on. Also, both importance plots work on different scales: SHAP may be interpreted on the scale of the prediction because SHAP importance is the average absolute change in prediction that was attributed to a feature. PFI is the average increase in loss when the feature information is destroyed (aka feature is permuted). Therefore PFI importance is on the scale of the loss.

For SHAP, it’s not so easy to answer how the Shapley values are supposed to be interpreted.

Shapley values are also expensive to compute, especially if your model is not tree-based.

So there are many reasons not to use SHAP, but an “inferior” (as the reviewer said) interpretation method.


-> Guess in my case random feature importance is correct.



- for feature importance in finance see also [[@lopezdepradoAdvancesFinancialMachine2018]] p. 118 etc.
“I find it useful to distinguish between feature importance methods based on whether they are impacted by substitution effects. In this context, a substitution effect takes place when the estimated importance of one feature is reduced by the presence of other related features. Substitution effects are the ML analogue of what the statistics and econometrics literature calls “multi-collinearity.” One way to address linear substitution effects is to apply PCA on the raw features, and then perform the feature importance analysis on the orthogonal features. See Belsley et al. [1980], Goldberger [1991, pp. 245–253], and Hill et al. [2001] for further details.” ([[@lopezdepradoAdvancesFinancialMachine2018]] p. 114)

“Substitution effects can lead us to discard important features that happen to be redundant. This is not generally a problem in the context of prediction, but it could lead us to wrong conclusions when we are trying to understand, improve, or simplify a model. For this reason, the following single feature importance method can be a good complement to MDI and MDA. 8.4.1 Single Feature Importance Single feature importance (SFI) is a cross-section predictive-importance (out-ofsample) method. It computes the OOS performance score of each feature in isolation. A few considerations: 1. This method can be applied to any classifier, not only tree-based classifiers. 2. SFI is not limited to accuracy as the sole performance score. 3. Unlike MDI and MDA, no substitution effects take place, since only one feature is taken into consideration at a time. 4. Like MDA, it can conclude that all features are unimportant, because performance is evaluated via OOS CV” The main limitation of SFI is that a classifier with two features can perform better than the bagging of two single-feature classifiers. For example, (1) feature B may be useful only in combination with feature A; or (2) feature B may be useful in explaining the splits from feature A, even if feature B alone is inaccurate. In other words, joint effects and hierarchical importance are lost in SFI. One alternative would be to compute the OOS performance score from subsets of features, but that calculation will become intractable as more features are considered. Snippet 8.4 demonstrates one possible implementation of the SFI method. A discussion of the function cvScore can be found in Chapter 7.” ([[@lopezdepradoAdvancesFinancialMachine2018]], p. 118)




- Discussion with Florian Kalinke: Feature importance Be aware of averages from averages. Symmetric trees. Why something is no problem. feature importance with tree. group features. 

## Random Feature permutation

Again, our definition of empirical MR is very similar to the permutation-based variable importance approach of Breimar (2001), where Breimar uses a single random permutation and we consider all possible pairs. To compare these two approaches more precisely, let $\left\{\pi_1, \ldots, \pi_{n !}\right\}$ be a set of $n$-length vectors, each containing a different permutation of the set $\{1, \ldots, n\}$. The approach of Breimar. (2001) is analogous to computing the loss $\sum_{i=1}^n L\left\{f,\left(\mathbf{y}_{[i]}, \mathbf{X}_{1\left[\pi_{[[i]},\right]}, \mathbf{X}_{2[i,]]}\right)\right\}$ for a randomly chosen permutation vector $\pi_l \in\left\{\pi_1, \ldots, \pi_{n !}\right\}$. Similarly, our calculation in Eq 3.i. is proportional to the sum of losses over all possible ( $n$ !) permutations, excluding the $n$ unique combinations of the rows of $\mathbf{X}_1$ and the rows of $\left[\begin{array}{ll}\mathbf{X}_2 & \mathbf{y}\end{array}\right]$ that appear in the original sample (see Appendix A.i.). Excluding these observations is necessary to preserve the (finite-sample) unbiasedness of $\hat{e}_{\text {switch }}(f)$


## High-level description
https://epub.ub.uni-muenchen.de/2821/1/deck.pdf
The permutation accuracy importance, that is described in more detail in Section 3, follows the rationale that a random permutation of the values of the predictor variable is supposed to mimic the absence of the variable from the model. The difference in the prediction accuracy before and after permuting the predictor variable, i.e. with and without the help of this predictor variable, is used as an importance measure.

## Random Feature permutation

8.5.1 Theory
The concept is really straightforward: We measure the importance of a feature by calculating the increase in the model's prediction error after permuting the feature. A feature is "important" if shuffling its values increases the model error, because in this case the model relied on the feature for the prediction. A feature is "unimportant" if shuffling its values leaves the model error unchanged, because in this case the model ignored the feature for the prediction. The permutation feature importance measurement was introduced by Breiman $(2001)^{43}$ for random forests. Based on this idea, Fisher, Rudin, and Dominici $(2018)^{44}$ [[@fisherAllModelsAre]] proposed a model-agnostic version of the feature importance and called it model reliance. They also introduced more advanced ideas about feature importance, for example a (model-specific) version that takes into account that many prediction models may predict the data well. Their paper is worth reading.

The permutation feature importance algorithm based on Fisher, Rudin, and Dominici (2018): Input: Trained model $\hat{f}$, feature matrix $X$, target vector $y$, error measure $L(y, \hat{f})$.
1. Estimate the original model error $e_{\text {orig }}=L(y, \hat{f}(X))$ (e.g. mean squared error)
2. For each feature $j \in\{1, \ldots, p\}$ do:
- Generate feature matrix $X_{\text {perm }}$ by permuting feature $j$ in the data $X$. This breaks the association between feature $\mathrm{j}$ and true outcome $\mathrm{y}$.
- Estimate error $e_{\text {perm }}=L\left(Y, \hat{f}\left(X_{\text {perm }}\right)\right)$ based on the predictions of the permuted data.
- Calculate permutation feature importance as quotient $F I_j=e_{\text {perm }} / e_{\text {orig }}$ or difference $F I_j=e_{\text {perm }}-e_{\text {orig }}$
3. Sort features by descending $\mathrm{FI}$.
Fisher, Rudin, and Dominici (2018) suggest in their paper to split the dataset in half and swap the values of feature $j$ of the two halves instead of permuting feature $j$. This is exactly the same as permuting feature j, if you think about it. If you want a more accurate estimate, you can estimate the error of permuting feature $j$ by pairing each instance with the value of feature $j$ of each other instance (except with itself). This gives you a dataset of size $n(n-1)$ to estimate the permutation error, and it takes a large amount of computation time. I can only recommend using the $n(n-1)$-method if you are serious about getting extremely accurate estimates.

## Theory
https://epub.ub.uni-muenchen.de/2821/1/deck.pdf
3. The permutation importance
The rationale of the original random forest permutation importance is the following: By randomly permuting the predictor variable $X_j$, its original association with the response $Y$ is broken. When the permuted variable $X_j$, together with the remaining non-permuted predictor variables, is used to predict the response for the out-of-bag observations, the prediction accuracy (i.e. the number of observations classified correctly) decreases substantially if the original variable $X_j$ was associated with the response. Thus, Breiman (2001a) suggests the difference in prediction accuracy before and after permuting $X_j$, averaged over all trees, as a measure for variable importance, that we formalize as follows: Let $\overline{\mathfrak{B}}^{(t)}$ be the out-of-bag (oob) sample for a tree $t$, with $t \in\{1, \ldots, n t r e e\}$. Then the variable importance of variable $X_j$ in tree $t$ is
$$
V I^{(t)}\left(\mathbf{X}_j\right)=\frac{\sum_{i \in \overline{\mathfrak{B}}^{(t)}} I\left(y_i=\hat{y}_i^{(t)}\right)}{\left|\overline{\mathfrak{B}}^{(t)}\right|}-\frac{\sum_{i \in \overline{\mathfrak{B}}^{(t)}} I\left(y_i=\hat{y}_{i, \pi_j}^{(t)}\right)}{\left|\overline{\mathfrak{B}}^{(t)}\right|}
$$
6
Conditional Variable Importance for Random Forests
where $\hat{y}_i^{(t)}=f^{(t)}\left(\mathbf{x}_i\right)$ is the predicted class for observation $i$ before and $\hat{y}_{i, \pi_j}^{(t)}=f^{(t)}\left(\mathbf{x}_{i, \pi_j}\right)$ is the predicted class for observation $i$ after permuting its value of variable $X_j$, i.e. with $\mathbf{x}_{i, \pi_j}=$ $\left(x_{i, 1}, \ldots, x_{i, j-1}, x_{\pi_j(i), j}, x_{i, j+1}, \ldots, x_{i, p}\right)$. (Note that $V I^{(t)}\left(\mathbf{X}_j\right)=0$ by definition, if variable $X_j$ is not in tree t.) The raw variable importance score for each variable is then computed as the mean importance over all trees: $V I\left(\mathbf{X}_j\right)=\frac{\sum_{t=1}^{\text {the }} V I^{(t)}\left(\mathbf{X}_j\right)}{n \text { tree }}$
In standard implementations of random forests an additional scaled version of the permutation importance (often called $z$-score), that is achieved by dividing the raw importance by its standard error, is provided. However, since the results of Strobl and Zeileis (2008) indicate that the raw importance $V I\left(\mathbf{X}_j\right)$ has better statistical properties, we will only consider the unscaled version here.


### 8.5.4 Advantages[](https://christophm.github.io/interpretable-ml-book/feature-importance.html#advantages-9)
https://christophm.github.io/interpretable-ml-book/feature-importance.html

**Nice interpretation**: Feature importance is the increase in model error when the feature’s information is destroyed.

Feature importance provides a **highly compressed, global insight** into the model’s behavior.

A positive aspect of using the error ratio instead of the error difference is that the feature importance measurements are **comparable across different problems**.

The importance measure automatically **takes into account all interactions** with other features. By permuting the feature you also destroy the interaction effects with other features. This means that the permutation feature importance takes into account both the main feature effect and the interaction effects on model performance. This is also a disadvantage because the importance of the interaction between two features is included in the importance measurements of both features. This means that the feature importances do not add up to the total drop in performance, but the sum is larger. Only if there is no interaction between the features, as in a linear model, the importances add up approximately.

Permutation feature importance **does not require retraining the model**. Some other methods suggest deleting a feature, retraining the model and then comparing the model error. Since the retraining of a machine learning model can take a long time, “only” permuting a feature can save a lot of time. Importance methods that retrain the model with a subset of features appear intuitive at first glance, but the model with the reduced data is meaningless for the feature importance. We are interested in the feature importance of a fixed model. Retraining with a reduced dataset creates a different model than the one we are interested in. Suppose you train a sparse linear model (with Lasso) with a fixed number of features with a non-zero weight. The dataset has 100 features, you set the number of non-zero weights to 5. You analyze the importance of one of the features that have a non-zero weight. You remove the feature and retrain the model. The model performance remains the same because another equally good feature gets a non-zero weight and your conclusion would be that the feature was not important. Another example: The model is a decision tree and we analyze the importance of the feature that was chosen as the first split. You remove the feature and retrain the model. Since another feature is chosen as the first split, the whole tree can be very different, which means that we compare the error rates of (potentially) completely different trees to decide how important that feature is for one of the trees.

### 8.5.5 Disadvantages[](https://christophm.github.io/interpretable-ml-book/feature-importance.html#disadvantages-9)

Permutation feature importance is **linked to the error of the model**. This is not inherently bad, but in some cases not what you need. In some cases, you might prefer to know how much the model’s output varies for a feature without considering what it means for performance. For example, you want to find out how robust your model’s output is when someone manipulates the features. In this case, you would not be interested in how much the model performance decreases when a feature is permuted, but how much of the model’s output variance is explained by each feature. Model variance (explained by the features) and feature importance correlate strongly when the model generalizes well (i.e. it does not overfit).

You **need access to the true outcome**. If someone only provides you with the model and unlabeled data – but not the true outcome – you cannot compute the permutation feature importance.

The permutation feature importance depends on shuffling the feature, which adds randomness to the measurement. When the permutation is repeated, the **results might vary greatly**. Repeating the permutation and averaging the importance measures over repetitions stabilizes the measure, but increases the time of computation.


## Different versions to do random feature permutation
https://arxiv.org/pdf/2109.01433.pdf
(have not yet fully understood)

## Correlations
https://www.borealisai.com/research-blogs/feature-importance-and-explainability/

Here, there are a variety of model-specific gradient-based methods (e.g., DeepLift [12] and GradCAM [11]) as well as various model-agnostic methods (e.g., LIME [10], SHAP [9], and permutation importance [6]). For example, LIME tries to learn a local surrogate linear model to provide local feature importance. SHAP is also mainly used for local explanation, but it can be modified to calculate global feature importance. Permutation importance is the most well-known method to calculate global feature importance for black-box models. It works by shuffling the values of a feature and measuring the changes in the model score, where the model score is defined based on the evaluation metric (e.g., R2 score for regression or accuracy for classification). Permutation importance, as well as LIME and SHAP, assume feature independence for calculating feature importance [1]. This is a fundamental problem with these methods, which is commonly overlooked and can provide misleading explanations when correlated features are present. For example, in the permutation importance algorithm, each feature is independently permuted, and the score change is calculated based on the individual feature changes. However, in practice, when a feature value changes, the correlated features are also changing.  
  
To alleviate this problem, we will introduce a **simple extension of the permutation importance algorithm for the scenarios where correlated features exist**. The new extension works by grouping the correlated features and then calculating the group-level imputation feature importance. The group-level imputation importance is similar to the original imputation importance, except that all the features in the group are permuted together.  
  
You can check out our release of this method on [GitHub](https://github.com/BorealisAI/group-feature-importance)!

## Correlations
https://explained.ai/rf-importance/index.html#4

Permute in groups.


## Correlations between features
[1]  Kjersti Aas, Martin Jullum, and Anders Løland. Explaining individual predictions when features are dependent: More accurate approximations to shapley values. Artificial Intelligence, 298:103502, 2021. 
- [[@aasExplainingIndividualPredictions2021]]
-


In its original definition, the MDA measure is applied to ensemble of trees using bootstrapping which consists in training each tree of the ensemble on a 11 different subset of the sample set. The importance of a feature xj is measured by its mean decrease of accuracy based on the out-of-bag (OOB) error when this feature is removed. The removal of a feature is simulated by permuting its value[22] and its impact is measured on all OOB samples. (https://matheo.uliege.be/bitstream/2268.2/13302/6/WeydersPFMasterThesis.pdf)

RF can determine the importance of a feature to STLF by calculating the PI value of each feature. When calculating the importance value of feature $F^j$ based on the $i$ th tree, OOBError $r_i$ is first calculated based on Equation (3). Then, the values of feature $F^j$ in the $\mathrm{OOB}$ dataset are randomly rearranged and those of the other features are unchanged, thereby forming a new $O O B$ dataset $O O B_i^{\prime}$. With the new $O O B_i{ }^{\prime}$ set, $O O B$ Error $_i{ }^{\prime}$ can also be calculated using Equation (3). The PI value of feature $F^j$ based on the $i$ th tree can be obtained by subtracting $O O B$ Error $_i$ from $O O B E r r r_i^{\prime}$.
$$
P I_i\left(F^j\right)=O O B E r r o r_i^{\prime}-O O B E r r o r_i
$$
The calculation process is repeated for each tree. The final PI value of feature $F i$ can be obtained by averaging the PI values of each tree:
$$
P I\left(F^j\right)=\frac{1}{c} \sum_{i=1}^c P I_i\left(F^j\right),
$$


(We know that the original permutation importance overestimates the importance of correlated predictor variables. Part of this artefact may be due to the preference of correlated predictor variables in early splits as illustrated in Section 2.2. However, we also have to take into account the permutation scheme that is employed in the computation of the permutation importance.) (Conditional variable importance for random forests Carolin Strobl*1, Anne-Laure Boulesteix2, Thomas Kneib1, Thomas Augustin1 and Achim Zeileis3)

(problems at the moment [33]), it shall remain untouched here.
2.3 The permutation importance
The rationale of the original random forest permutation importance is the following: By randomly permuting the predictor variable $X_j$, its original association with the response $Y$ is broken. When the permuted variable $X_{j^{\prime}} \quad$ where $\hat{\gamma}_i^{(t)}=f^{(t)}\left(\mathrm{x}_i\right)$ is the predicted class for observation together with the remaining non-permuted predictor var- $\quad i$ before and $\hat{y}_{i, \pi_j}^{(t)}=f^{(t)}\left(\mathrm{x}_{i, \pi_j}\right)$ is the predicted class for iables, is used to predict the response for the out-of-bag observations, the prediction accuracy (i.e. the number of observations classified correctly) decreases substantially if with $\mathrm{x}_{i, \pi_j}=\left(x_{i, 1}, \ldots, x_{i, j-1,1} x_{\pi_j(i), j}, x_{i, j+1}, \ldots, x_{i, p}\right)$. (Note that the original variable $X_j$ was associated with the response. $V I(t)\left(\mathbf{X}_j\right)=0$ by definition, if variable $X_j$ is not in tree $t$.) The Thus, Breiman [1] suggests the difference in prediction raw variable importance score for each variable is then accuracy before and after permuting $X_j$, averaged over all computed as the mean importance over all trees: trees, as a measure for variable importance, that we formalize as follows: Let $\overline{\mathcal{B}}^{(t)}$ be the out-of-bag (oob) sam$V I\left(\mathrm{x}_j\right)=\frac{\sum_{t=1}^{n \text { tree } V I}(t)\left(\mathrm{x}_j\right)}{\text { ntree }}$ ple for a tree $t$, with $t \in\{1, \ldots$, ntree $\}$. Then the variable
In standard implementations of random forests an addiimportance of variable $X_j$ in tree $t$ is tional scaled version of the permutation importance (often called $z$-score), that is achieved by dividing the raw importance by its standard error, is provided. However,) (Conditional variable importance for random forests Carolin Strobl*1, Anne-Laure Boulesteix2, Thomas Kneib1, Thomas Augustin1 and Achim Zeileis3)


**Correlations:**
Even though Archer and Kimes [20] show in their extensive simulation study that the Gini importance can identify influential predictor variables out of sets of correlated covariates in many settings, the preliminary results of the simulation study of Nicodemus and Shugart [21] indicate that the ability of the permutation importance to detect influential predictor variables in sets of correlated covariates is less reliable than that of alternative machine learning methods and highly depends on the number of previously selected splitting variables mtry. These studies, as well as our simulation results, indicate that random forests show a preference for correlated predictor variables, that is also carried forward to any significance test or variable selection scheme constructed from the importance measures. (Conditional variable importance for random forests Carolin Strobl*1, Anne-Laure Boulesteix2, Thomas Kneib1, Thomas Augustin1 and Achim Zeileis)

(In other words, for the permutation feature importance of a correlated feature, we consider how much the model performance decreases when we exchange the feature with values we would never observe in reality. Check if the features are strongly correlated and be careful about the interpretation of the feature importance if they are. However, pairwise correlations might not be sufficient to reveal the problem.

Another tricky thing: **Adding a correlated feature can decrease the importance of the associated feature** by splitting the importance between both features. Let me give you an example of what I mean by “splitting” feature importance https://christophm.github.io/interpretable-ml-book/feature-importance.html) 

Siyu Zhou
siz25@pitt.edu Variable Importance These methods are designed to provide a score for each feature based on how much difference its values in the training data and examining the corre-
sponding drop in predictive accuracy when these new data are used in a model built with the original training data. that correspond to tracing out the prediction given to any one example as the $j$ th feature is changed. PDPs are then Given a training set consisting of a matrix of feature values exactly the average of the corresponding ICE plots, but the $X$ with rows $\boldsymbol{x}_i$ giving each observation and corresponding latter allows an investigation in how the effect of feature response vector $y$, let $X^{\pi, j}$ be a matrix achieved by ran- $j$ may change for different combinations of the remaining domly permuting the $j$ th column of $X$. Using $L\left(y_i, f\left(x_i\right)\right.$ ) inputs. When $N$ is very large, a random selection of ICE as the loss for predicting $y_i$ from $f\left(x_i\right)$, the importance of plots can be presented as examples. Goldstein et al. (2015) the $j$ th feature can be defined as: also described how these ICE plots can potentially be used to detect the kind of extrapolation we discuss in detail in $\mathrm{VI}_j^\pi=\sum_{i=1}^N L\left(y_i, f\left(x_i^{\pi, j}\right)\right)-L\left(y_i, f\left(\boldsymbol{x}_i\right)\right)$ this paper.
(Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance)

**Partial dependence Plots:**

Partial Dependence Plots (PDPs) Friedman (2001) suggested examining the effect of feature j by plotting the average prediction as the feature is changed. Specifically, letting Xx,j be the matrix of feature values where the jth entry of every row has been replaced with value x, we define the partial dependence function PDj(x) = 1 N N i=1 f (xx,j i ) as the average prediction made with the jth feature replaced with the value x. Since these are univariate functions (multivariate versions can be defined naturally), they can be readily displayed and interpreted. (Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance https://link.springer.com/article/10.1007/s11222-021-10057-z)

## 8.1 Partial Dependence Plot (PDP)[](https://christophm.github.io/interpretable-ml-book/pdp.html#pdp)

The partial dependence plot (short PDP or PD plot) shows the marginal effect one or two features have on the predicted outcome of a machine learning model (J. H. Friedman 2001[30](https://christophm.github.io/interpretable-ml-book/pdp.html#fn30)). A partial dependence plot can show whether the relationship between the target and a feature is linear, monotonic or more complex. For example, when applied to a linear regression model, partial dependence plots always show a linear relationship.

The partial dependence function for regression is defined as:

^fS(xS)=EXC[^f(xS,XC)]=∫^f(xS,XC)dP(XC)�^�(��)=���[�^(��,��)]=∫�^(��,��)��(��)

The xS�� are the features for which the partial dependence function should be plotted and XC�� are the other features used in the machine learning model ^f�^, which are here treated as random variables. Usually, there are only one or two features in the set S. The feature(s) in S are those for which we want to know the effect on the prediction. The feature vectors xS�� and xC�� combined make up the total feature space x. Partial dependence works by marginalizing the machine learning model output over the distribution of the features in set C, so that the function shows the relationship between the features in set S we are interested in and the predicted outcome. By marginalizing over the other features, we get a function that depends only on features in S, interactions with other features included.

The partial function ^fS�^� is estimated by calculating averages in the training data, also known as Monte Carlo method:

^fS(xS)=1nn∑i=1^f(xS,x(i)C)�^�(��)=1�∑�=1��^(��,��(�))The partial function tells us for given value(s) of features S what the average marginal effect on the prediction is. In this formula, x(i)C��(�) are actual feature values from the dataset for the features in which we are not interested, and n is the number of instances in the dataset. An assumption of the PDP is that the features in C are not correlated with the features in S. If this assumption is violated, the averages calculated for the partial dependence plot will include data points that are very unlikely or even impossible (see disadvantages).

For classification where the machine learning model outputs probabilities, the partial dependence plot displays the probability for a certain class given different values for feature(s) in S. An easy way to deal with multiple classes is to draw one line or plot per class.

The partial dependence plot is a global method: The method considers all instances and gives a statement about the global relationship of a feature with the predicted outcome.

**Categorical features**

So far, we have only considered numerical features. For categorical features, the partial dependence is very easy to calculate. For each of the categories, we get a PDP estimate by forcing all data instances to have the same category. For example, if we look at the bike rental dataset and are interested in the partial dependence plot for the season, we get four numbers, one for each season. To compute the value for “summer”, we replace the season of all data instances with “summer” and average the predictions.

### 8.1.1 PDP-based Feature Importance[](https://christophm.github.io/interpretable-ml-book/pdp.html#pdp-based-feature-importance)

Greenwell et al. (2018) [31](https://christophm.github.io/interpretable-ml-book/pdp.html#fn31) proposed a simple partial dependence-based feature importance measure. The basic motivation is that a flat PDP indicates that the feature is not important, and the more the PDP varies, the more important the feature is. For numerical features, importance is defined as the deviation of each unique feature value from the average curve:

I(xS)= ⎷1K−1K∑k=1(^fS(x(k)S)−1KK∑k=1^fS(x(k)S))2�(��)=1�−1∑�=1�(�^�(��(�))−1�∑�=1��^�(��(�)))2

Note that here the x(k)S��(�) are the K unique values of feature the XS��. For categorical features we have:

I(xS)=(maxk(^fS(x(k)S))−mink(^fS(x(k)S)))/4�(��)=(����(�^�(��(�)))−����(�^�(��(�))))/4

This is the range of the PDP values for the unique categories divided by four. This strange way of calculating the deviation is called the range rule. It helps to get a rough estimate for the deviation when you only know the range. And the denominator four comes from the standard normal distribution: In the normal distribution, 95% of the data are minus two and plus two standard deviations around the mean. So the range divided by four gives a rough estimate that probably underestimates the actual variance.

This PDP-based feature importance should be interpreted with care. It captures only the main effect of the feature and ignores possible feature interactions. A feature could be very important based on other methods such as [permutation feature importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance), but the PDP could be flat as the feature affects the prediction mainly through interactions with other features. Another drawback of this measure is that it is defined over the unique values. A unique feature value with just one instance is given the same weight in the importance computation as a value with many instances (https://christophm.github.io/interpretable-ml-book/pdp.html)

**True to the model / to the data**
In this paper, we analyzed two approaches to explain models using the Shapley value solution concept for cooperative games. In order to compare these approaches we focus on explaining linear models and present a novel methodology for explaining linear models with correlated features. We analyze two different settings where either the interventional Shapley values or the observational Shapley values are preferable. In the first setting, we consider a model trained on loans data that might be used to determine which applicants obtain loans. Because applicants in this setting are ultimately interested in why the model makes a prediction, we call this case ”true to the model” and show that interventional Shapley values serve to modify the model’s prediction more effectively. In the second setting we consider a model trained on biological data that aims to understand an underlying causal relationship. Because this setting is focused on scientific discovery, we call this case ”true to the data” and show that for a sparse model (Lasso regularized) observational Shapley values discover more of the true features. We also find that modeling decisions can achieve some of the same effects, by demonstrating that the interventional Shapley values recover more of the true features when applied to a model that itself spreads credit among correlated features than when applied to a sparse model. ([[@chenTrueModelTrue2020]])


(Many model-agnostic machine learning (ML) interpretation methods (see Molnar (2019); Guidotti et al. (2018) for an overview) are based on making predictions on perturbed input features, such as permutations of features. The partial dependence plot (PDP) (Friedman et al., 1991) visualizes how changing a feature affects the prediction on average. The permutation feature importance (PFI) (Breiman, 2001; Fisher et al., 2019) quantifies the importance of a feature as the reduction in model performance after permuting a feature. PDP and PFI change feature values without conditioning on the remaining features https://arxiv.org/pdf/2006.04628.pdf)



**Notation:**
(https://arxiv.org/pdf/2006.04628.pdf)
We consider ML prediction functions ˆf : Rp 7→ R, where ˆf(x) is a model prediction and x ∈ Rp is a p-dimensional feature vector. We use xj ∈ Rn to refer to an observed feature (vector) and Xj to refer to the j-th feature as a random variable. With x−j we refer to the complementary feature space x{1,...,p}\{j} ∈ Rn×(p−1) and with X−j to the corresponding random variables. We refer to the value of the j-th feature from the i-th instance as x (i) j and to the tuples D = { x (i) , y(i)  } n i=1 as data. The Permutation Feature Importance (PFI) is defined as the increase in loss when feature Xj is permuted: P F Ij = E[L(Y, ˆf(X˜ j , X−j ))] − E[L(Y, ˆf(Xj , X−j ))] (1) If the random variable X˜ j has the same marginal distribution as Xj (e.g., permutation), the estimate yields the marginal PFI. If X˜ j follows the conditional distribution X˜ j ∼ Xj |X−j , we speak of the conditional PFI. The PFI is estimated with the following formula: P F I [j = 1 n Xn i=1 1 M X M m=1 L˜m(i) − L (i) ) ! (2 Importance and Effects with Dependent Features 5 where L (i) = L(y (i) , ˆf(x (i) )) is the loss for the i-th observation and L˜(i) = L(y (i) , ˆf(˜x (i) j , x (i) −j )) is the loss where x (i) j was replaced by the m-th sample x˜ m(i) j . The latter refers to the i-th feature value obtained by a sample of xj . The sample can be repeated M-times for a more stable estimation of L˜(i) . Numerous variations of this formulation exist. Breiman (2001) proposed the PFI for random forests, which is computed from the out-of-bag samples of individual trees. Subsequently, Fisher et al. (2019) introduced a model-agnostic PFI version. The marginal Partial Dependence Plot (PDP) (Friedman et al., 1991) describes the average effect of the j-th feature on the prediction. P DPj (x) = E[ ˆf(x, X−j )], (3) If the expectation is conditional on Xj , E[ ˆf(x, X−j )|Xj = x], we speak of the conditional PDP. The marginal PDP evaluated at feature value x is estimated using Monte Carlo integration: P DP \j (x) = 1 n Xn i=1 ˆf(x, x (i) −j ) (4) 3 Related Work

**Permutation in conditional groups:**
https://arxiv.org/pdf/2006.04628.pdf