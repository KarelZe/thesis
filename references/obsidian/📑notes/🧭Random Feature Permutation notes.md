

**Explainability and interpretablity**
> 📑Although it is common in the literature to see the terms "interpretability" and "explainability" used interchangeably,[17](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib17), [18](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib18), [19](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib19) they are distinct concepts, and their conflation can cause significant confusion.[20](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib20)

> 📑Although there is slight variability in the precise definition of the term "interpretable" in the literature,[21](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib21), [22](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib22), [23](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib23), [24](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib24) throughout this review we use the term to refer to models in which humans can directly understand how a model operates and the causes of its decisions.[25](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib25) Logistic-regression models are interpretable because a human can refer to the weights and odds ratios to understand how the model operates and can refer to coefficients to understand the cause of individual predictions. It should be noted that interpretability is at least somewhat subjective, as it can require expert knowledge of statistics or a domain (such as cardiology) to interpret a model’s decisions effectively.[8](https://www.sciencedirect.com/science/article/pii/S0828282X21007030#bib8)

**Interpretability and transparency**
> 📑 “The field of in-model interpretability [23] focuses on intrinsically interpretable models. These “transparent” [84]models naturally, and by design, provide some degree of interpretability. Lipton [84] classifies transparency in three dimensions, namely simulatability, decomposability, and algorithmic transparency. Simulatability relates to the ability to understand the entire model. Lipton [84]notesthat simulatability is not a direct consequence of the use of a particular model. For example, and even though models such as linear regression, rule-based systems, and decision trees are typically easier to interpret [8], in some cases, a compact neural network may be more transparent than the former alternatives. Note that even simple methods such as linear regression can become very challenging as the number of predictors increases. In expert systems based on if-then rules, it may not be possible to grasp all the rules and their interactions. Seemingly, decision trees can become too deep or too broad for graphical visualization and comprehension. Lipton [84]’s second notion of transparency, decomposability, defines to which degree the user can understand the model components − input data, parameters, and calculation rules. The third notion of Lipton [84]isalgorithmic transparency, which relates to the ability to understand the inferential process. It is important to consider these three notions when designing “transparent” machine learning models. We hereafter review some of the approaches proposed to achieve in-model transparency.” ([[@baptistaRelationPrognosticsPredictor2022]] p. 4) -> classical trade classification rules are interpretable / transparent

Classical trade classification algorithms, as a rule-based approach, are transparent and thus provide some degree of interpretability. (What is transparency? -> [[@liptonMythosModelInterpretability2017]]). However, for deep stacked stacked combinations involving a large feature count, such as the one of ([[@grauerOptionTradeClassification2022]] ), interactions between base rules become more complex, and the effect of single feature on the final prediction more challenging to interpret. 

> 📑“Here, we adopt the definition of Biran and Cotton [19], as the level that an observer can understand the cause of a decision.” (Baptista et al., 2022, p. 2)

**Need for  interpetability**
> 📑In addition to pre-model and in-model interpretability, there is <mark style="background: #FFB8EBA6;">post-model interpretability</mark> [23]. Post-model techniques analyze the model after its creation (post-hoc); they are devised as independent methods that can interpret the final decisions. There approaches can be model-specific or model-agnostic [8]. Post-hoc model-specific interpretability consists of methods specifically designed for a given machine learning algorithm. In contrast, post-hoc model-agnostic interpretability is agnostic to the analyzed machine learning model.

> 📑“Rule based learners are great models in terms of interpretability across fields. Their natural and seamless relation to human behaviour makes them very suitable to understand and explain other models. If a certain threshold of coverage is acquired, a rule wrapper can be thought to contain enough information about a model to explain its behavior to a non-expert user, without forfeiting the possibility of using the generated rules as an standalone prediction model.” ([[@barredoarrietaExplainableArtificialIntelligence2020]] 2020, p. 91)

> 📑“Following simplification procedures, feature relevance techniques are also used in the field of tree ensembles. Breiman [286] was the first to analyze the variable importance within Random Forests. His method is based on measuring MDA (Mean Decrease Accuracy) or MIE (Mean Increase Error) of the forest when a certain variable is randomly permuted in the out-of-bag samples. Following this contribution [241] shows, in an real setting, how the usage of variable importance reflects the underlying relationships of a complex system modeled by a Random Forest” ([[@barredoarrietaExplainableArtificialIntelligence2020]]., 2020, p. 94)

> 📑“When ML models do not meet any of the criteria imposed to declare them transparent, a separate method must be devised and applied to the model to explain its decisions. This is the purpose of post-hoc explainability techniques (also referred to as post-modeling explainability), which aim at communicating understandable information about how an already developed model produces its predictions for any given input.” ([[@barredoarrietaExplainableArtificialIntelligence2020]] p. 92)

**estimating feature importances is non-trivial**
>“Evaluation of the quality of feature attribution is known to be a non-trivial problem [206].” ([[@borisovDeepNeuralNetworks2022]], p. 13)
> Y. Rong, T. Leemann, V. Borisov, G. Kasneci, and E. Kasneci, “A consistent and efficient evaluation strategy for feature attribution methods,” in International Conference on Machine Learning. PMLR, 2022.

**audit vs insight / true to the model - true to the data**
> 📑 Given the simulation setup where none of the features has a relation to the target, one could say that PFI results are correct and SHAP is wrong. But this answer is too simplistic. The choice of interpretation method really depends on what you use the importance values for. What is the question that you want to answer? Because Shapley values are “correct” in the sense that they do what they are supposed to do: Attribute the prediction to the features. And in this case, changing the “important” features truly changes the model prediction. So if your goal tends towards understanding how the model “behaves”, SHAP might be the right choice. But if you want to find out how relevant a feature was for the CORRECT prediction, SHAP is not a good option. Here PFI is the better choice since it links importance to model performance. For SHAP, it’s not so easy to answer how the Shapley values are supposed to be interpreted. Shapley values are also expensive to compute, especially if your model is not tree-based. So there are many reasons not to use SHAP, but an “inferior” (as the reviewer said) interpretation method. -> Guess in my case random feature importance is correct. (https://mindfulmodeler.substack.com/p/shap-is-not-all-you-need)

> 📑In a way, it boils down to the question of [audit versus insight](https://mindfulmodeler.substack.com/p/audit-or-insight-know-your-interpretation): SHAP importance is more about auditing how the model behaves. As in the simulated example, it’s useful to see how model predictions are affected by features X4, X6, and so on. For that SHAP importance is meaningful. But if your goal was to study the underlying data, then it’s completely misleading. Here PFI gives you a better idea of what’s really going on. Also, both importance plots work on different scales: SHAP may be interpreted on the scale of the prediction because SHAP importance is the average absolute change in prediction that was attributed to a feature. PFI is the average increase in loss when the feature information is destroyed (aka feature is permuted). Therefore PFI importance is on the scale of the loss.

> 📑In this paper, we analyzed two approaches to explain models using the Shapley value solution concept for cooperative games. In order to compare these approaches we focus on explaining linear models and present a novel methodology for explaining linear models with correlated features. We analyze two different settings where either the interventional Shapley values or the observational Shapley values are preferable. In the first setting, we consider a model trained on loans data that might be used to determine which applicants obtain loans. Because applicants in this setting are ultimately interested in why the model makes a prediction, we call this case ”true to the model” and show that interventional Shapley values serve to modify the model’s prediction more effectively. In the second setting we consider a model trained on biological data that aims to understand an underlying causal relationship. Because this setting is focused on scientific discovery, we call this case ”true to the data” and show that for a sparse model (Lasso regularized) observational Shapley values discover more of the true features. We also find that modeling decisions can achieve some of the same effects, by demonstrating that the interventional Shapley values recover more of the true features when applied to a model that itself spreads credit among correlated features than when applied to a sparse model. ([[@chenTrueModelTrue2020]])

The machine-learning classifiers, studied in this work can be deemed a black box model. Due to the sheer size of the network or ensemble, both classifiers are no longer transparent-> reformulate, necessitating post-hoc techniques for interpretability. We require a model-agnostic approach, which suffice for a cross-comparison between all classifiers. Our goal is to identify features that are important for the *correct prediction* in the model. This is fundamentally different from methods, such as LIME or SHAP (wrong?), which attribute *any* prediction to the input features. 

> 📑“The model size (number of parameters) can provide a first intuition of the interpretability of the models. Therefore, we provide a size comparison of deep learning models in Fig. 5. Admittedly, explanations can be provided in very different forms, which may each have their own use-cases. Hence, we can only compare explanations that have a common form. In this work, we chose feature attributions as the explanation format because they are the prevalent form of post-hoc explainability for the models considered in this work.” ([[@borisovDeepNeuralNetworks2022]], p. 13)

> 📑 Many model-agnostic machine learning (ML) interpretation methods (see Molnar (2019); Guidotti et al. (2018) for an overview) are based on making predictions on perturbed input features, such as permutations of features. The partial dependence plot (PDP) (Friedman et al., 1991) visualizes how changing a feature affects the prediction on average. The permutation feature importance (PFI) (Breiman, 2001; Fisher et al., 2019) quantifies the importance of a feature as the reduction in model performance after permuting a feature. PDP and PFI change feature values without conditioning on the remaining features (https://arxiv.org/pdf/2006.04628.pdf)

> Permutation methods are some of the oldest, most popular, and computationally convenient means of understanding complex learning algorithms. In this paper, we will focus primarily on three commonly used techniques: [[@hookerUnrestrictedPermutationForces2021]]

“Post-hoc interpretability presents a distinct approach to extracting information from learned models. While posthoc interpretations often do not elucidate precisely how a model works, they may nonetheless confer useful information for practitioners and end users of machine learning.” ([[@liptonMythosModelInterpretability2017]] 4).

It connotes some sense of understanding the mechanism by which the model works. ([[@liptonMythosModelInterpretability2017]] 5).

A class of approaches, suitable for studying the data (better model-agnostic techniques) are based on random feature permutation. We study permutation feature importance / partial feature importance in detail.

## Permutation Based Methods
> 📑The permutation accuracy importance, that is described in more detail in Section 3, follows the rationale that a random permutation of the values of the predictor variable is supposed to mimic the absence of the variable from the model. The difference in the prediction accuracy before and after permuting the predictor variable, i.e. with and without the help of this predictor variable, is used as an importance measure. (https://epub.ub.uni-muenchen.de/2821/1/deck.pdf)

### Random Feature permutation

> 📑The concept is really straightforward: We measure the importance of a feature by calculating the increase in the model's prediction error after permuting the feature. A feature is "important" if shuffling its values increases the model error, because in this case the model relied on the feature for the prediction. A feature is "unimportant" if shuffling its values leaves the model error unchanged, because in this case the model ignored the feature for the prediction. The permutation feature importance measurement was introduced by Breiman $(2001)^{43}$ for random forests. Based on this idea, Fisher, Rudin, and Dominici $(2018)^{44}$ [[@fisherAllModelsAre]] proposed a model-agnostic version of the feature importance and called it model reliance. They also introduced more advanced ideas about feature importance, for example a (model-specific) version that takes into account that many prediction models may predict the data well. Their paper is worth reading. (Molnar)

> 📑RF can determine the importance of a feature to STLF by calculating the PI value of each feature. When calculating the importance value of feature $F^j$ based on the $i$ th tree, OOBError $r_i$ is first calculated based on Equation (3). Then, the values of feature $F^j$ in the $\mathrm{OOB}$ dataset are randomly rearranged and those of the other features are unchanged, thereby forming a new $O O B$ dataset $O O B_i^{\prime}$. With the new $O O B_i{ }^{\prime}$ set, $O O B$ Error $_i{ }^{\prime}$ can also be calculated using Equation (3). The PI value of feature $F^j$ based on the $i$ th tree can be obtained by subtracting $O O B$ Error $_i$ from $O O B E r r r_i^{\prime}$.
$$
P I_i\left(F^j\right)=O O B E r r o r_i^{\prime}-O O B E r r o r_i
$$
The calculation process is repeated for each tree. The final PI value of feature $F i$ can be obtained by averaging the PI values of each tree:
$$
P I\left(F^j\right)=\frac{1}{c} \sum_{i=1}^c P I_i\left(F^j\right),
$$


> 📑Again, our definition of empirical MR is very similar to the permutation-based variable importance approach of Breimar (2001), where Breimar uses a single random permutation and we consider all possible pairs. To compare these two approaches more precisely, let $\left\{\pi_1, \ldots, \pi_{n !}\right\}$ be a set of $n$-length vectors, each containing a different permutation of the set $\{1, \ldots, n\}$. The approach of Breimar. (2001) is analogous to computing the loss $\sum_{i=1}^n L\left\{f,\left(\mathbf{y}_{[i]}, \mathbf{X}_{1\left[\pi_{[[i]},\right]}, \mathbf{X}_{2[i,]]}\right)\right\}$ for a randomly chosen permutation vector $\pi_l \in\left\{\pi_1, \ldots, \pi_{n !}\right\}$. Similarly, our calculation in Eq 3.i. is proportional to the sum of losses over all possible ( $n$ !) permutations, excluding the $n$ unique combinations of the rows of $\mathbf{X}_1$ and the rows of $\left[\begin{array}{ll}\mathbf{X}_2 & \mathbf{y}\end{array}\right]$ that appear in the original sample (see Appendix A.i.). Excluding these observations is necessary to preserve the (finite-sample) unbiasedness of $\hat{e}_{\text {switch }}(f)$




> 📑The rationale of the original random forest permutation importance is the following: By randomly permuting the predictor variable $X_j$, its original association with the response $Y$ is broken. When the permuted variable $X_j$, together with the remaining non-permuted predictor variables, is used to predict the response for the out-of-bag observations, the prediction accuracy (i.e. the number of observations classified correctly) decreases substantially if the original variable $X_j$ was associated with the response. Thus, Breiman (2001a) suggests the difference in prediction accuracy before and after permuting $X_j$, averaged over all trees, as a measure for variable importance, that we formalize as follows: Let $\overline{\mathfrak{B}}^{(t)}$ be the out-of-bag (oob) sample for a tree $t$, with $t \in\{1, \ldots, n t r e e\}$. Then the variable importance of variable $X_j$ in tree $t$ is (https://epub.ub.uni-muenchen.de/2821/1/deck.pdf)
$$
V I^{(t)}\left(\mathbf{X}_j\right)=\frac{\sum_{i \in \overline{\mathfrak{B}}^{(t)}} I\left(y_i=\hat{y}_i^{(t)}\right)}{\left|\overline{\mathfrak{B}}^{(t)}\right|}-\frac{\sum_{i \in \overline{\mathfrak{B}}^{(t)}} I\left(y_i=\hat{y}_{i, \pi_j}^{(t)}\right)}{\left|\overline{\mathfrak{B}}^{(t)}\right|}
$$
> 📑The rationale of the original random forest permutation importance is the following: By randomly permuting the predictor variable $X_j$, its original association with the response $Y$ is broken. When the permuted variable $X_{j^{\prime}} \quad$ where $\hat{\gamma}_i^{(t)}=f^{(t)}\left(\mathrm{x}_i\right)$ is the predicted class for observation together with the remaining non-permuted predictor var- $\quad i$ before and $\hat{y}_{i, \pi_j}^{(t)}=f^{(t)}\left(\mathrm{x}_{i, \pi_j}\right)$ is the predicted class for iables, is used to predict the response for the out-of-bag observations, the prediction accuracy (i.e. the number of observations classified correctly) decreases substantially if with $\mathrm{x}_{i, \pi_j}=\left(x_{i, 1}, \ldots, x_{i, j-1,1} x_{\pi_j(i), j}, x_{i, j+1}, \ldots, x_{i, p}\right)$. (Note that the original variable $X_j$ was associated with the response. $V I(t)\left(\mathbf{X}_j\right)=0$ by definition, if variable $X_j$ is not in tree $t$.) The Thus, Breiman [1] suggests the difference in prediction raw variable importance score for each variable is then accuracy before and after permuting $X_j$, averaged over all computed as the mean importance over all trees: trees, as a measure for variable importance, that we formalize as follows: Let $\overline{\mathcal{B}}^{(t)}$ be the out-of-bag (oob) sam$V I\left(\mathrm{x}_j\right)=\frac{\sum_{t=1}^{n \text { tree } V I}(t)\left(\mathrm{x}_j\right)}{\text { ntree }}$ ple for a tree $t$, with $t \in\{1, \ldots$, ntree $\}$. Then the variable In standard implementations of random forests an addiimportance of variable $X_j$ in tree $t$ is tional scaled version of the permutation importance (often called $z$-score), that is achieved by dividing the raw importance by its standard error, is provided. However,) (Conditional variable importance for random forests Carolin Strobl*1, Anne-Laure Boulesteix2, Thomas Kneib1, Thomas Augustin1 and Achim Zeileis3)

> 📑We consider ML prediction functions ˆf : Rp 7→ R, where ˆf(x) is a model prediction and x ∈ Rp is a p-dimensional feature vector. We use xj ∈ Rn to refer to an observed feature (vector) and Xj to refer to the j-th feature as a random variable. With x−j we refer to the complementary feature space x{1,...,p}\{j} ∈ Rn×(p−1) and with X−j to the corresponding random variables. We refer to the value of the j-th feature from the i-th instance as x (i) j and to the tuples D = { x (i) , y(i)  } n i=1 as data. The Permutation Feature Importance (PFI) is defined as the increase in loss when feature Xj is permuted: P F Ij = E[L(Y, ˆf(X˜ j , X−j ))] − E[L(Y, ˆf(Xj , X−j ))] (1) If the random variable X˜ j has the same marginal distribution as Xj (e.g., permutation), the estimate yields the marginal PFI. If X˜ j follows the conditional distribution X˜ j ∼ Xj |X−j , we speak of the conditional PFI. The PFI is estimated with the following formula: P F I j = 1 n Xn i=1 1 M X M m=1 L˜m(i) − L (i) ) ! (2 Importance and Effects with Dependent Features 5 where L (i) = L(y (i) , ˆf(x (i) )) is the loss for the i-th observation and L˜(i) = L(y (i) , ˆf(˜x (i) j , x (i) −j )) is the loss where x (i) j was replaced by the m-th sample x˜ m(i) j . The latter refers to the i-th feature value obtained by a sample of xj . The sample can be repeated M-times for a more stable estimation of L˜(i) . Numerous variations of this formulation exist. Breiman (2001) proposed the PFI for random forests, which is computed from the out-of-bag samples of individual trees. Subsequently, Fisher et al. (2019) introduced a model-agnostic PFI version. ((https://arxiv.org/pdf/2006.04628.pdf)) 

### Discussion
> 📑(In other words, for the permutation feature importance of a correlated feature, we consider how much the model performance decreases when we exchange the feature with values we would never observe in reality. Check if the features are strongly correlated and be careful about the interpretation of the feature importance if they are. However, pairwise correlations might not be sufficient to reveal the problem. https://christophm.github.io/interpretable-ml-book/feature-importance.html)

> 📑(We know that the original permutation importance overestimates the importance of correlated predictor variables. Part of this artefact may be due to the preference of correlated predictor variables in early splits as illustrated in Section 2.2. However, we also have to take into account the permutation scheme that is employed in the computation of the permutation importance.) (Conditional variable importance for random forests Carolin Strobl*1, Anne-Laure Boulesteix2, Thomas Kneib1, Thomas Augustin1 and Achim Zeileis3) 

> 📑Here, there are a variety of model-specific gradient-based methods (e.g., DeepLift [12] and GradCAM [11]) as well as various model-agnostic methods (e.g., LIME [10], SHAP [9], and permutation importance [6]). For example, LIME tries to learn a local surrogate linear model to provide local feature importance. SHAP is also mainly used for local explanation, but it can be modified to calculate global feature importance. Permutation importance is the most well-known method to calculate global feature importance for black-box models. It works by shuffling the values of a feature and measuring the changes in the model score, where the model score is defined based on the evaluation metric (e.g., R2 score for regression or accuracy for classification). Permutation importance, as well as LIME and SHAP, assume feature independence for calculating feature importance [1]. This is a fundamental problem with these methods, which is commonly overlooked and can provide misleading explanations when correlated features are present. For example, in the permutation importance algorithm, each feature is independently permuted, and the score change is calculated based on the individual feature changes. However, in practice, when a feature value changes, the correlated features are also changing.  To alleviate this problem, we will introduce a **simple extension of the permutation importance algorithm for the scenarios where correlated features exist**. The new extension works by grouping the correlated features and then calculating the group-level imputation feature importance. The group-level imputation importance is similar to the original imputation importance, except that all the features in the group are permuted together.  (https://www.borealisai.com/research-blogs/feature-importance-and-explainability/ )

> 📑Another tricky thing: **Adding a correlated feature can decrease the importance of the associated feature** by splitting the importance between both features.  https://christophm.github.io/interpretable-ml-book/feature-importance.html) 

> 📑“I find it useful to distinguish between feature importance methods based on whether they are impacted by substitution effects. In this context, a substitution effect takes place when the estimated importance of one feature is reduced by the presence of other related features. Substitution effects are the ML analogue of what the statistics and econometrics literature calls “multi-collinearity.” One way to address linear substitution effects is to apply PCA on the raw features, and then perform the feature importance analysis on the orthogonal features. See Belsley et al. [1980], Goldberger [1991, pp. 245–253], and Hill et al. [2001] for further details.” ([[@lopezdepradoAdvancesFinancialMachine2018]] p. 114)

> 📑“Substitution effects can lead us to discard important features that happen to be redundant. This is not generally a problem in the context of prediction, but it could lead us to wrong conclusions when we are trying to understand, improve, or simplify a model. For this reason, the following single feature importance method can be a good complement to MDI and MDA. 8.4.1 Single Feature Importance Single feature importance (SFI) is a cross-section predictive-importance (out-ofsample) method. It computes the OOS performance score of each feature in isolation. A few considerations: 1. This method can be applied to any classifier, not only tree-based classifiers. 2. SFI is not limited to accuracy as the sole performance score. 3. Unlike MDI and MDA, no substitution effects take place, since only one feature is taken into consideration at a time. 4. Like MDA, it can conclude that all features are unimportant, because performance is evaluated via OOS CV” The main limitation of SFI is that a classifier with two features can perform better than the bagging of two single-feature classifiers. For example, (1) feature B may be useful only in combination with feature A; or (2) feature B may be useful in explaining the splits from feature A, even if feature B alone is inaccurate. In other words, joint effects and hierarchical importance are lost in SFI. One alternative would be to compute the OOS performance score from subsets of features, but that calculation will become intractable as more features are considered. Snippet 8.4 demonstrates one possible implementation of the SFI method. A discussion of the function cvScore can be found in Chapter 7.” ([[@lopezdepradoAdvancesFinancialMachine2018]], p. 118)


We group dependent features and estimate the feature importance on a group-level. Similar idea https://www.borealisai.com/research-blogs/feature-importance-and-explainability/ or  https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html.


## Partial dependence plots
Closely related to the idea of random feature permutation, are partial dependence plots (...)

> 📑The marginal Partial Dependence Plot (PDP) (Friedman et al., 1991) describes the average effect of the j-th feature on the prediction. P DPj (x) = E[ ˆf(x, X−j )], (3) If the expectation is conditional on Xj , E[ ˆf(x, X−j )|Xj = x], we speak of the conditional PDP. The marginal PDP evaluated at feature value x is estimated using Monte Carlo integration: P DP \j (x) = 1 n Xn i=1 ˆf(x, x (i) −j ) (4)  (https://arxiv.org/pdf/2006.04628.pdf)

> 📑 Partial Dependence Plots (PDPs) Friedman (2001) suggested examining the effect of feature j by plotting the average prediction as the feature is changed. Specifically, letting Xx,j be the matrix of feature values where the jth entry of every row has been replaced with value x, we define the partial dependence function PDj(x) = 1 N N i=1 f (xx,j i ) as the average prediction made with the jth feature replaced with the value x. Since these are univariate functions (multivariate versions can be defined naturally), they can be readily displayed and interpreted. (Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance https://link.springer.com/article/10.1007/s11222-021-10057-z)

> 📑This PDP-based feature importance should be interpreted with care. It captures only the main effect of the feature and ignores possible feature interactions. A feature could be very important based on other methods such as [permutation feature importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance), but the PDP could be flat as the feature affects the prediction mainly through interactions with other features. Another drawback of this measure is that it is defined over the unique values. A unique feature value with just one instance is given the same weight in the importance computation as a value with many instances (https://christophm.github.io/interpretable-ml-book/pdp.html) -> As we use both random feature permutation and partial dependence plotss problem is less severe


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

Greenwell et al. (2018) [31](https://christophm.github.io/interpretable-ml-book/pdp.html#fn31) proposed a simple partial dependence-based feature importance measure. The basic motivation is that a flat PDP indicates that the feature is not important, and the more the PDP varies, the more important the feature is. For numerical features, importance is defined as the deviation of each unique feature value from the average curve:

I(xS)= ⎷1K−1K∑k=1(^fS(x(k)S)−1KK∑k=1^fS(x(k)S))2�(��)=1�−1∑�=1�(�^�(��(�))−1�∑�=1��^�(��(�)))2

Note that here the x(k)S��(�) are the K unique values of feature the XS��. For categorical features we have:

I(xS)=(maxk(^fS(x(k)S))−mink(^fS(x(k)S)))/4�(��)=(����(�^�(��(�)))−����(�^�(��(�))))/4

This is the range of the PDP values for the unique categories divided by four. This strange way of calculating the deviation is called the range rule. It helps to get a rough estimate for the deviation when you only know the range. And the denominator four comes from the standard normal distribution: In the normal distribution, 95% of the data are minus two and plus two standard deviations around the mean. So the range divided by four gives a rough estimate that probably underestimates the actual variance. (https://christophm.github.io/interpretable-ml-book/pdp.html)


This PDP-based feature importance should be interpreted with care. It captures only the main effect of the feature and ignores possible feature interactions. A feature could be very important based on other methods such as [permutation feature importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance), but the PDP could be flat as the feature affects the prediction mainly through interactions with other features. Another drawback of this measure is that it is defined over the unique values. A unique feature value with just one instance is given the same weight in the importance computation as a value with many instances (https://christophm.github.io/interpretable-ml-book/pdp.html)

## Word salad 🥗
As the 

One such 

This leaves us 


Consequently, we derive feature importances using random feature permutation, which is model-agnostic and computationally efficient.

Random feature permutation only yields global 


Our goal is to understand the contribution of a feature to the correct prediction, rather than attributing the prediction to specific features. 

Our setting is unique. 

As we defined derived features, such as the proximity to the quotes, features can not assumed to be independent. 

Substitution effects

However, feature independence is 

The features used by the model may also be different. The quote

Also machine learning classifiers have simultaneous access to 
Also feature importances, may be diluted over several features, known as as features may encode the same information redundantly

Also, features. Classical 

As such, we adapt random feature importance to our setting 

Random feature permutation was originally proposed in 


Random feature permutation is model-agnostic and can be used with different error estimates. For consistency the change in accuracy is used in our work.

The change can be estimated, as the absolute or relative difference.

Random feature permutation as proposed by b

Permuting features also

The complete algorithm is given in:

Two major drawbacks of random feature permutation, are 

One major drawback of random feature permutation is, that it doesn't help with local interpretability. Correlations are artificially broken

unrealistic permutations