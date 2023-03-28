
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


## Correlations between features
[1]  Kjersti Aas, Martin Jullum, and Anders Løland. Explaining individual predictions when features are dependent: More accurate approximations to shapley values. Artificial Intelligence, 298:103502, 2021. 
- [[@aasExplainingIndividualPredictions2021]]
- 