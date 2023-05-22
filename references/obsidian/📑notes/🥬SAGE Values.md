
**High-level classification**
“We refer to this class of approaches as removal-based explanations and identify 261 existing methods that rely on the feature removal principle, including several of the most widely used methods (SHAP, LIME, Meaningful Perturbations, permutation tests).” ([[@covertExplainingRemovingUnified]], p. 2)

![[removal-based-explanations.png]]

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