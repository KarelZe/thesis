
**High-level classification**
“We refer to this class of approaches as removal-based explanations and identify 261 existing methods that rely on the feature removal principle, including several of the most widely used methods (SHAP, LIME, Meaningful Perturbations, permutation tests).” ([[@covertExplainingRemovingUnified]], p. 2)

![[removal-based-explanations.png]]

**When features are not independent**
“The main disadvantage of the Shapley value is that the computational complexity grows exponentially and becomes intractable for more than, say, ten features. This has led to approximations like the Shapley Sampling Values [9,10] and Kernel SHAP [11]. The latter requires less computational power to obtain a similar approximation accuracy. Hence, in this paper, the focus is on the Kernel SHAP method. While having many desirable properties, this method assumes feature independence. In observational studies and machine learning problems, it is very rare that the features are statistically independent, meaning that the Shapley value methods suffer from inclusion of predictions based on unrealistic data instances when features are correlated. This is the case even if a simple linear model is used.” ([[@aasExplainingIndividualPredictions2021]], p. 2)

“The two previous categories of methods provide imperfect notions of feature importance because they do not account for feature interactions. For example, two perfectly correlated features with significant predictive power would both be deemed unimportant by a feature ablation study, and two complementary features would have their importance underestimated by univariate models.” ([[@covertUnderstandingGlobalFeature2020]], p. 4)

A violation of this assumption can lead to substitution effects, see ([[@lopezdepradoAdvancesFinancialMachine2018]]) for a detailed discussion in the context of finance. 

**Coalitional Game**
![[coalitational-games.png]]
([[@chenAlgorithmsEstimateShapley2022]])

**Exact Shapley value and cooperative game theory**
“Consider a cooperative game with M players aiming at maximizing a payoff, and let S ⊆ M ={1, ...,M} be a subset consisting of |S| players. Assume that we have a contribution function v(S) that maps subsets of players to the real numbers, called the worth or contribution of coalition S. It describes the total expected sum of payoffs the members of S can obtain by cooperation. The Shapley value [12]isone way to distribute the total gains to the players, assuming that they all collaborate. It is a “fair” distribution in the sense that it is the only distribution with certain desirable properties listed below. According to the Shapley value, the amount that player j gets is φj(v) = φj = ∑ S⊆M\{ j} |S|!(M −|S|−1)! M! (v(S ∪{j}) − v(S)), j = 1,...,M, (1) that is, a weighted mean over contribution function differences for all subsets S of players not containing player j. Note that the empty set S =∅ is also part of this sum. The formula can be interpreted as follows: Imagine the coalition being formed for one player at a time, with each player demanding their contribution v(S ∪{j}) − v(S) as a fair compensation. Then, for each player, compute the average of this contribution over all permutations of all possible coalitions, yielding a weighted mean over the unique coalitions. To illustrate the application of (1), let us consider a game with three players such that M ={1, 2, 3}. Then, there are eight possible subsets; ∅, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, and {1, 2, 3}. Using (1), the Shapley values for the three players are given by φ1 = 1 3 (v({1, 2, 3}) − v({2, 3})) + 1 6 (v({1, 2}) − v({2})) + 1 6 (v({1, 3}) − v({3})) + 1 3 (v({1}) − v(∅)), φ2 = 1 3 (v({1, 2, 3}) − v({1, 3})) + 1 6 (v({1, 2}) − v({1})) + 1 6 (v({2, 3}) − v({3})) + 1 3 (v({2}) − v(∅)), φ3 = 1 3 (v({1, 2, 3}) − v({1, 2})) + 1 6 (v({1, 3}) − v({1})) + 1 6 (v({2, 3}) − v({2})) + 1 3 (v({3}) − v(∅)). Let us also define the non-distributed gain φ0 = v(∅), that is, the fixed payoff which is not associated to the actions of any of the players, although this is often zero for coalition games. By summarizing the right hand sides above, we easily see that they add up to the total worth of the game: φ0 + φ1 + φ2 + φ3 = v({1, 2, 3}). The Shapley value has the following desirable properties:” (Aas et al., 2021, p. 3)

**Shapley values for prediction explanation**
“2.2. Shapley values for prediction explanation Consider a classical machine learning scenario where a training set { yi , xi }i=1,...,ntrain of size ntrain has been used to train a predictive model f (x) attempting to resemble a response value y as closely as possible. Assume now that we want to explain the prediction from the model f (x∗), for a specific feature vector x = x∗. Štrumbel and Kononenko [9,10]and Lundberg and Lee [11]suggestto do this using Shapley values. By moving from game theory to decomposing an individual prediction into feature contributions, the single prediction takes the place of the payout, and the features take the place of the players. We have that the prediction f (x∗) is decomposed as follows f (x∗) = φ0 + M ∑ j=1 φj ∗, where φ0 = E[ f (x)] and φ j ∗ is the φ j for the prediction x = x∗. That is, the Shapley values explain the difference between the prediction y∗ = f (x∗) and the global average prediction. A model of this form is an additive feature attribution method, and it is the only additive feature attribution method that adhers to the properties listed in Section 2.1 [11]. In Appendix A we discuss why these properties are useful in the prediction explanation setting. To be able to compute the Shapley values in the prediction explanation setting, we need to define the contribution function v(S) for a certain subset S. This function should resemble the value of f (x∗) when we only know the value of the subset S of these features. To quantify this, we follow [11] and use the expected output of the predictive model, conditional on the feature values xS = x∗S of this subset: v(S) = E[ f (x)|xS = x∗S ]. (2) Other measures, such as the conditional median, may also be appropriate. However, the conditional expectation summarizes the whole probability distribution and it is the most common estimator in prediction applications. When viewed as a prediction for f (x∗), it is also the minimizer of the commonly used squared error loss function. We show in Appendix B that if the predictive model is a linear regression model f (x) = β0 + ∑M j=1 β j x j , where all features x j, j = 1, ...,M are independent, then, under (2), the Shapley values take the simple form: φ0 = β0 + M ∑ j=1 β j E[x j], and φ j = β j (x∗j − E[x j]), j = 1,...,M. (3) Note that for ease of notation, we have here and in the rest of the paper dropped the superscript * for the φ j values. Every prediction f (x∗) to be explained will result in different sets of φ j values.” (Aas et al., 2021, p. 4)


**Defining absence**
“Alternatively, absent features can be replaced according to a set of baselines with different distributional assumptions. In particular, the uniform approach uses the range of the baselines’ absent features to define independent uniform distributions to draw absent features from. The product of marginals approach draws each absent feature independently according to the values seen in the baselines. The marginal approach draws groups of absent feature values that appeared in the baselines. Finally, the conditional approach only considers samples that exactly match on the present features. Note that this figure depicts empirically estimating each expectation; however, in practice, the conditional approach is estimated by fitting models (Section 5.1.3” (Chen et al., 2022, p. 5)

**Global and Local Feature Importances**
“Given a model f and features x1, . . . , xd, feature attributions explain predictions by assigning scalar values that represent each feature’s importance. For an intuitive description of feature attributions, we first consider linear models. Linear models of the form f (x) = β0 + β1x1 + · · · + βdxd are often considered interpretable because each feature is linearly related to the prediction via a single parameter. In this case, a common global feature attribution that describes the model’s overall dependence on feature i is the corresponding coefficient βi. For linear models, each coefficient βi describes the influence that variations in feature xi have on the model output.” ([[@chenTrueModelTrue2020]], p. 3)

**Feature Importance Methods in a Nutshell**
“Feature importance methods quantify the contribution of a feature to the model performance (e.g. via a loss function) or to the variance of the prediction function. Importance methods include the PFI, ICI, PI, or SAGE. See Fig. 1 for a visual summary.” (“xxAI - Beyond Explainable AI: International Workshop, Held in Conjunction with ICML 2020, July 18, 2020, Vienna, Austria, Revised and Extended Papers”, 2022, p. 41)

For example, there are several ways to quantify or rank the features according to their relevance. The relevance measured by PFI can be very different from the relevance measured by the SHAP importance. If a practitioner aims to gain insight into the relevance of a feature regarding the model’s generalization error, a loss-based method (on unseen test data) such as PFI should be used. If we aim to expose which features the model relies on for its prediction or classification irrespective of whether they aid the model’s generalization performance – PFI on test data is misleading. In such scenarios, one should quantify the relevance of a feature regarding the model’s prediction (and not the model’s generalization error) using methods like the SHAP importance [76]. (Explaining data vs. Model) ([[@chenTrueModelTrue2020]])

**Global Importance Measure:**
“Global importance measures Let Y denote the outcome and let XD ={X1, ., Xd} collectively denote d variables, where D = {1, ., d} is the set of all variable indices. A model of Y built using the d variables is denoted by f(XD), with expected loss E{L(f(XD), Y)}. Fisher and team12 proposed a permutation-based measure of variable contribution, referred to as model reliance (MR). The MR of variable Xj (j ̨D) is the increase in expected loss when the contribution of this variable is removed by random permutation: mrjðfÞ = E n L f XDyfjg; X0 j ;Y o EfLðfðXDÞ; YÞg ; where XDyfjg denotes the set XD after excluding Xj, and X0 j follows the marginal distribution of Xj. mrj(f) = 1 suggests model f does not rely on Xj, and larger mrj(f) indicates increased reliance. Although straightforward and easy to implement, the permutation approach does not account for interactions among variables, as it removes one variable at a time.” ([[@ningShapleyVariableImportance2022]], p. 2)

**Dependency / Interaction**
“When features are dependent, perturbation-based IML methods such as PFI, PDP, LIME, and Shapley values extrapolate in areas where the model was trained with little or no training data, which can cause misleading interpretations [55]. This is especially true if the ML model relies on feature interactions [45] – which is often the case.” (“xxAI - Beyond Explainable AI: International Workshop, Held in Conjunction with ICML 2020, July 18, 2020, Vienna, Austria, Revised and Extended Papers”, 2022, p. 47)

**Build up argument of dependency / interaction**

“Pitfall: Features with a Pearson correlation coefficient (PCC) close to zero can still be dependent and cause misleading model interpretations (see Fig. 5). While independence between two features implies that the PCC is zero, the converse is generally false. The PCC, which is often used to analyze dependence, only tracks linear correlations and has other shortcomings such as sensitivity to outliers [113]. Any type of dependence between features can have a strong impact on the interpretation of the results of IML methods (see Sect. 5.1). Thus, knowledge about the (possibly non-linear) dependencies between features is crucial for an informed use of IML methods. Solution: Low-dimensional data can be visualized to detect dependence (e.g. scatter plots) [80]. For high-dimensional data, several other measures of dependence in addition to PCC can be used. If dependence is monotonic, Spearman’s rank correlation coefficient [72] can be a simple, robust alternative to PCC. For categorical or mixed features, separate dependence measures have been proposed, such as Kendall’s rank correlation coefficient for ordinal features, or the phi coefficient and Goodman & Kruskal’s lambda for nominal features [59].” (“xxAI - Beyond Explainable AI: International Workshop, Held in Conjunction with ICML 2020, July 18, 2020, Vienna, Austria, Revised and Extended Papers”, 2022, p. 48)

**Problems / Properties of SAGE**
“Similarly, conditional SAGE and conditional SHAP value functions sample the remaining features conditional on the feature of interest and therefore violate sensitivity [25,56,61,109].” (“xxAI - Beyond Explainable AI: International Workshop, Held in Conjunction with ICML 2020, July 18, 2020, Vienna, Austria, Revised and Extended Papers”, 2022, p. 50)

**Difference between conditional / marginal**

**True to the model / true to the data**

“Conditional variants of interpretation techniques avoid extrapolation but require a different interpretation. Interpretation methods that perturb features independently of others will extrapolate under dependent features but provide insight into the model’s mechanism [56,61]. Therefore, these methods are said to be true to the model but not true to the data [21]. For feature effect methods such as the PDP, the plot can be interpreted as the isolated, average effect the feature has on the prediction. For the PFI, the importance can be interpreted as the drop in performance when the feature’s information is “destroyed” (by perturbing it). Marginal SHAP value functions [78] quantify a feature’s contribution to a specific prediction, and marginal SAGE value functions [25] quantify a feature’s contribution to the overall prediction performance. All the aforementioned methods extrapolate under dependent features (see also Sect. 5.1), but satisfy sensitivity, i.e. are zero if a feature is not used by the model [25,56,61,110].” (“xxAI - Beyond Explainable AI: International Workshop, Held in Conjunction with ICML 2020, July 18, 2020, Vienna, Austria, Revised and Extended Papers”, 2022, p. 49)

**Shapley values and SAGE**

“explanations account for this by viewing variables as players in a cooperative game15,16 and measures the impact of variable Xj on model f based on its marginal contribution when some variables, XS3XD, are already present. The Shapley values are defined as: 4jðwÞ = 1 d X S4fDyfjgg d1 jSj 1 ½wðSWfjgÞ wðSÞ: (Equation 1) w(S) quantifies the contribution of subset XS to the model, which is defined differently for different types of Shapley-based variable importance measures and will be explicitly defined below for SHAP and SAGE” ([[@ningShapleyVariableImportance2022]], p. 3)

jSj denotes the number of variables in this subset, and d  1 jSj  is the number of ways to choose jSj variables from XDyfjg. 4j(w) = 0 indicates no contribution, and larger values indicate increased contribution.16

**Shapley Values, SAGE, and conditional random permutation**

“(Marginalize with conditional) SHAP (Lundberg and Lee, 2017), LossSHAP (Lundberg et al., 2020) and SAGE (Covert et al., 2020) present a strategy for removing features by marginalizing them out using their conditional distribution p(X ̄ S | XS = xS): F (xS) = E[f (X) | XS = xS ]. (4) This approach is computationally challenging in practice, but recent work tries to achieve close approximations (Aas et al., 2019, 2021; Frye et al., 2020). Shapley Effects (Owen, 2014) implicitly uses this convention to analyze function sensitivity, while conditional permutation tests (Strobl et al., 2008) and Prediction Difference Analysis (PredDiff, Zintgraf et al., 2017) propose simple approximations, with the latter conditioning only on groups of bordering pixels.” ([[@covertExplainingRemovingUnified]] p. 12)

“(Dataset loss) Shapley Net Effects, SAGE, SPVIM, feature ablation, permutation tests and univariate predictors consider the expected loss across the entire dataset: v(S) = −EXY ( ell (F (XS), Y ) ) . (16) These methods quantify how much the model’s performance degrades when different features are removed. This set function can also be viewed as the predictive power derived from sets of features (Covert et al., 2020), and recent work has proposed a SHAP value aggregation that is a special case of this approach (Frye et al., 2020).” ([[@covertExplainingRemovingUnified]]  p. 15)

“• (Shapley value) Shapley Net Effects, IME, Shapley Effects, QII, SHAP (including KernelSHAP, TreeSHAP and LossSHAP), SPVIM and SAGE all calculate feature attributions using the Shapley value, which we denote as ai = φi(u). Described in more detail in Section 7, Shapley values are the only attributions that satisfy several desirable properties.” ([[@covertExplainingRemovingUnified]]  p. 18)

**SHAP**
“SHAP [85]isa model-agnostic approach from XAI that draws its foundations from game theory [128]. The goal of SHAP is to explain a prediction f (x) of an instance x by computing the relative contribution of each feature value to the specific outcome. The explanation function g(.) receives as input a coalition vector z′ ⊂{0, 1}N where N is the number of features in the original instance vector x. The coalition vector represents the presence or absence of each feature in a binary format: an entry of 1 means that the corresponding feature contributes to the explanation, while an entry of 0 means that the feature is considered to have no contribution. We have that the explanation function g(z′) can be decomposed as follows: g(z′) = φ0 + N ∑ i=1 φi zi′,φi ∈ R (5) where: N = number of input features in x, the instance vector g = explanation model z′ = coalition vector such that z′ ⊂{0, 1}N φi = decomposition factor Several methods match the definition in Equation (5), namely LIME [108], DeepLIFT [123]andLayer-wise Relevance Propagation (LRP) [12]. These are all additive feature attribution methods that, as SHAP, attribute an effect (or importance) φi to each predictor (feature), and the sum of these effects, g(z′), approximates the output f (x) of the original model. As an example, consider Fig. 1. The picture displays the relationship between an input vector and the corresponding prediction. Here, the feature values, xi lead to the prediction f (x). SHAP, and the other referred models, work by assigning a decomposition factor φi , to each feature value, which aims to reflect the importance of the feature to that particular prediction. Assuming the four axioms of efficiency, symmetry, dummy and additivity, the previous decomposition has been shown [85]tohave a unique solution known as Shapley value, proposed by Lloyd Shapley [121]incooperative game theory: φi( f , x) = 1 N! ∑ S⊆P \{xi } [ |S|!(N −|S|−1)! ][ f (S ∪{xi}) − f (S) ] (6) where: x = instance vector N = number of input features in x” ([[@baptistaRelationPrognosticsPredictor2022]], p. 8)

![[shap-visualisation.png]]

“SHAP [85]isa model-agnostic approach from XAI that draws its foundations from game theory [128]. The goal of SHAP is to explain a prediction f (x) of an instance x by computing the relative contribution of each feature value to the specific outcome. The explanation function g(.) receives as input a coalition vector z′ ⊂{0, 1}N where N is the number of features in the original instance vector x. The coalition vector represents the presence or absence of each feature in a binary format: an entry of 1 means that the corresponding feature contributes to the explanation, while an entry of 0 means that the feature is considered to have no contribution. We have that the explanation function g(z′) can be decomposed as follows: g(z′) = φ0 + N ∑ i=1 φi zi′,φi ∈ R (5) where: N = number of input features in x, the instance vector g = explanation model z′ = coalition vector such that z′ ⊂{0, 1}N φi = decomposition factor Several methods match the definition in Equation (5), namely LIME [108], DeepLIFT [123]andLayer-wise Relevance Propagation (LRP) [12]. These are all additive feature attribution methods that, as SHAP, attribute an effect (or importance) φi to each predictor (feature), and the sum of these effects, g(z′), approximates the output f (x) of the original model. As an example, consider Fig. 1. The picture displays the relationship between an input vector and the corresponding prediction. Here, the feature values, xi lead to the prediction f (x). SHAP, and the other referred models, work by assigning a decomposition factor φi , to each feature value, which aims to reflect the importance of the feature to that particular prediction. Assuming the four axioms of efficiency, symmetry, dummy and additivity, the previous decomposition has been shown [85]tohave a unique solution known as Shapley value, proposed by Lloyd Shapley [121]incooperative game theory: φi( f , x) = 1 N! ∑ S⊆P \{xi } [ |S|!(N −|S|−1)! ][ f (S ∪{xi}) − f (S) ] (6) where: x = instance vector N = number of input features in x” ([[@baptistaRelationPrognosticsPredictor2022]], p. 8)


“When w(S) is the expected reduction in loss over the mean prediction by including XS, i.e., wðSÞ = vf ðSÞ = EfLðE½fðXDÞ; YÞg EfLðfðXDjXS = xSÞ; YÞg, 4j(vf) is the SAGE value for a formal global interpretation.16”  ([[@ningShapleyVariableImportance2022]], p. 3)


“In this work we seek to understand how much models rely on each feature overall, which is often referred to as the problem of global feature importance.”  ([[@ningShapleyVariableImportance2022]], p. 3)


“We then present a new tool for calculating feature importance, SAGE,1 a model-agnostic approach to summarizing a model’s dependence on each feature while accounting for complex interactions (Section 3).” (Covert et al., 2020, p. 1)

Look up what they exactly mean with feature interactions


“The two previous categories of methods provide imperfect notions of feature importance because they do not account for feature interactions. For example, two perfectly correlated features with significant predictive power would both be deemed unimportant by a feature ablation study, and two complementary features would have their importance underestimated by univariate models. The third category of methods addresses these issues by considering all feature subsets S ⊆ D. Methods in the third category account for complex feature interactions by modeling v across its entire domain P(D). These methods therefore supersede the two other categories, which either exclude or include individual features. Our method, SAGE, belongs to this category, and we show that SAGE assigns scores by modeling vf optimally via a weighted least squares objective (Section 3.2).” (Covert et al., 2020, p. 4)


Naturally, one would like to obtain insights into how the models arrived at the  
prediction and identify features relevant for the prediction. Both aspects can be  
subsumed under the term interpretability. Following, Lipton (2017, p. 4) inter-  
pretability can be reached through model transparency or post-hoc interpretability  
methods. Transparent models provide interpretability through a transparent mech-  
anism in the model, whereas post-hoc methods extract information from the already  
learnt model (Lipton, 2017, pp. 4–5).  
Classical trade classification algorithms, as a rule-based classifier, are transparent  
with an easily understandable decision process and thus provide interpretability  
(Barredo Arrieta et al., 2020, p. 91). Interpretability, however, decreases for deep,  
stacked combinations involving a large feature count, when interactions between base  
rules become more complex and the effect of a single feature on the final prediction  
more challenging to interpret.  
15The ISE test set consists of 48.60 % of buy trades and 46.13 % of the CBOE test set are buy  
trades.
The machine-learning classifiers, studied in this work, can be deemed a black box  
model (Barredo Arrieta et al., 2020, p. 90). Due to the sheer size of the network  
or ensemble, interpretability through transparency is impacted. Albeit, the atten-  
tion mechanism of Transformers provides some interpretability through the atten-  
tion mechanism, interpretability across all classifiers can only be reached through  
a model-agnostic, post-hoc interpretability techniques. Thereby, our goal is to iden-  
tify features that are important for the correct prediction. This is fundamentally  
different from methods like standard SHapley Additive exPlanations (SHAP), that  
attribute any prediction to the input features (H. Chen et al., 2020, ??).  
Many model-agnostic methods are based on (...)  
Shapley Additive Global Importance  
(...)  
Attention Maps  
In addition to permutation-based methods, Transformer-based models offer some  
interpretability through their attention mechanism. In recent research a major con-  
troversy embarked around the question, of whether attention offers explanations to  
model predictions (cp. Bastings & Filippova, 2020, p. 150; Jain & Wallace, 2019,  
pp. 5–7; Wiegreffe & Pinter, 2019, p. 9). The debate sparked around opposing  
definitions of explainability and the consistency of attention scores with other, es-  
tablished feature-importance measures. Our focus is less on post-hoc explainability  
of the model, but rather on transparency. Consistent with Wiegreffe and Pinter  
(2019, p. 8) we view attention scores as a vehicle to model transparency.  
Recall from our discussion on attention (cp. Section 4.4.4) that the attention matrix  
stores how much attention a token pays to each of the keys. Thus, feature attri-  
butions can be derived from attention by visualising features to which the model  
attends to in an attention map. While attention maps are specific to Transform-  
ers or other attention-based architectures, rendering them useless for cross-model  
comparisons, they give additional insights from different attention layers and atten-  
tion heads of the model on a per-trade and global basis. An example is shown in  
Figure 15.