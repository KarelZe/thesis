Naturally, we aim to gain insights into the prediction process and identify relevant features, which fall under the umbrella of *interpretability*. Following, ([[@liptonMythosModelInterpretability2017]]4) interpretability can be reached through model transparency or post-hoc interpretability methods. Transparent models provide interpretability through a transparent mechanism in the model, whereas post-hoc interpretability refers to approaches that extract information from the already learnt model ([[@liptonMythosModelInterpretability2017]] 4--5). 

Classical trade classification algorithms, as a rule-based approach, are transparent with an easily understandable decision process, and thus provide interpretability ([[@barredoarrietaExplainableArtificialIntelligence2020]]91). Interpretability, however decreases for deep stacked combinations involving a large feature count, such as the gls-GSU method, interactions between base rules become more complex, and the effect of single feature on the final prediction more challenging to interpret.-footnote Consider the deep stacked combination from cref ... both the ordering plays a role for ...

The machine learning classifiers, studied in this work, can be deemed a black box model ([[@barredoarrietaExplainableArtificialIntelligence2020]]90). Due to the sheer size of the network or ensemble, interpretability through transparency is impacted. Albeit, the attention mechanism of Transformers provides some interpretability through transparency, interpretability across all classifiers can only be reached through *model-agnostic, post-hoc interpretability techniques*.

Thereby, our goal is to estimate how much a feature contributes to the performance of the classifier *overall*, which urges for *global* feature attribution measures. The appropriate approach is guided by properties of the data. Features are dependent due to the data generating process with strongly correlated quotes and trade prices at the exchange and nation-wide level. The redundant feature encoding of ratio features exacerbates this effect. Feature independence, however, is the central assumption of most popular feature importance measures, including gls-SHAP, gls-LIME, or gls-rfpm ([[@aasExplainingIndividualPredictions2021]]2). A violation of this constraint, can lead for two perfectly correlated, predictive features to the effect that both are deemed unimportant as the feature importance is distributed between features thereby underestimating the true importance of the features ([[@covertUnderstandingGlobalFeature2020]], p. 4). (for more detailled investigation see [[@hookerUnrestrictedPermutationForces2021]], not so important here I guess)

For this reason we estimate feature importances using gls-SAGE, which can account for complex interactions between features and yields global importances. 

**Shapley Additive Global Importance**
gls-SAGE is an additive feature importance measure with its foundations in cooperative game theory. As put forth by ([[@lundbergUnifiedApproachInterpreting2017]]3) feature contributions can be estimated through Shapley values (Source Lloyd). Instead of allocating credit in a cooperative game to players, as originally proposed, the problem transfers to assign credit across features based on a value function. Intuitionally, for gls-SAGE, credit is assigned based the contribution to the model's performance, which is different from classical gls-SHAP formulation, where the importance is determined by the contribution to the prediction.

Shapley values are defined as:





Strong mathematical foundation.


“2.2. Shapley values for prediction explanation Consider a classical machine learning scenario where a training set { yi , xi }i=1,...,ntrain of size ntrain has been used to train a predictive model f (x) attempting to resemble a response value y as closely as possible. Assume now that we want to explain the prediction from the model f (x∗), for a specific feature vector x = x∗. Štrumbel and Kononenko [9,10]and Lundberg and Lee [11]suggest to do this using Shapley values. By moving from game theory to decomposing an individual prediction into feature contributions, the single prediction takes the place of the payout, and the features take the place of the players. We have that the prediction f (x∗) is decomposed as follows f (x∗) = φ0 + M ∑ j=1 φj ∗, where φ0 = E[ f (x)] and φ j ∗ is the φ j for the prediction x = x∗. That is, the Shapley values explain the difference between the prediction y∗ = f (x∗) and the global average prediction. A model of this form is an additive feature attribution method, and it is the only additive feature attribution method that adhers to the properties listed in Section 2.1 [11]. In Appendix A we discuss why these properties are useful in the prediction explanation setting. To be able to compute the Shapley values in the prediction explanation setting, we need to define the contribution function v(S) for a certain subset S. This function should resemble the value of f (x∗) when we only know the value of the subset S of these features. To quantify this, we follow [11] and use the expected output of the predictive model, conditional on the feature values xS = x∗S of this subset: v(S) = E[ f (x)|xS = x∗S ]. (2) Other measures, such as the conditional median, may also be appropriate. However, the conditional expectation summarizes the whole probability distribution and it is the most common estimator in prediction applications. When viewed as a prediction for f (x∗), it is also the minimizer of the commonly used squared error loss function. We show in Appendix B that if the predictive model is a linear regression model f (x) = β0 + ∑M j=1 β j x j , where all features x j, j = 1, ...,M are independent, then, under (2), the Shapley values take the simple form: φ0 = β0 + M ∑ j=1 β j E[x j], and φ j = β j (x∗j − E[x j]), j = 1,...,M. (3) Note that for ease of notation, we have here and in the rest of the paper dropped the superscript * for the φ j values. Every prediction f (x∗) to be explained will result in different sets of φ j values.” (Aas et al., 2021, p. 4)


Additive refers 

<mark style="background: #ABF7F7A6;">These (SHAP) are all additive feature attribution methods that, as SHAP, attribute an effect (or importance) φi to each predictor (feature), and the sum of these effects, g(z′), approximates the output f (x) of the original model. </mark>  ([[@baptistaRelationPrognosticsPredictor2022]], p. 8)

<mark style="background: #ADCCFFA6;">“Definition 1 Additive feature attribution methods have an explanation model that is a linear function of binary variables: g(z′) = φ0 + M ∑ i=1 φiz′ i, (1) where z′ ∈ {0, 1}M , M is the number of simplified input features, and φi ∈ R. Methods with explanation models matching Definition 1 attribute an effect φi to each feature, and summing the effects of all feature attributions approximates the output f (x) of the original model. Many current methods match Definition 1, several of which are discussed below” (Lundberg and Lee, 2017, p. 2)
</mark>

Definition 1 Additive feature attribution methods have an explanation model that is a linear function of binary variables:
$$
g\left(z^{\prime}\right)=\phi_0+\sum_{i=1}^M \phi_i z_i^{\prime},
$$
where $z^{\prime} \in\{0,1\}^M, M$ is the number of simplified input features, and $\phi_i \in \mathbb{R}$.
Methods with explanation models matching Definition [] attribute an effect $\phi_i$ to each feature, and summing the effects of all feature attributions approximates the output $f(x)$ of the original model. Many current methods match Definition several of which are discussed below.

*Shapley values*


**Shapley values**
Recall that the function $v_f$ describes the amount of predictive power that a model $f$ derives from subsets of features $S \subseteq D$. We define feature importance via $v_f$ to quantify how critical each feature $X_i$ is for $f$ to make accurate predictions. It is natural to view $v_f$ as a cooperative game, representing the profit (predictive power) when each player (feature) participates (is available to the model). Research in game theory has extensively analyzed credit allocation for cooperative games, so we apply a game theoretic solution known as the Shapley value [35].

Shapley values are the unique credit allocation scheme that satisfies a set of fairness axioms. For any cooperative game $w: \mathcal{P}(D) \mapsto \mathbb{R}$ (such as $v$ or $v_f$ ) we may want the scores $\phi_i(w)$ assigned to each player to satisfy the following desirable properties:
1. (Efficiency) They sum to the total improvement over the empty set, $\sum_{i=1}^d \phi_i(w)=w(D)-w(\varnothing)$.
2. (Symmetry) If two players always make equal contributions, or $w(S \cup\{i\})=w(S \cup\{j\})$ for all $S$, then $\phi_i(w)=\phi_j(w)$
3. (Dummy) If a player makes zero contribution, or $w(S)=w(S \cup\{i\})$ for all $S$, then $\phi_i(w)=0$.
4. (Monotonicity) If for two games $w$ and $w^{\prime}$ a player always make greater contributions to $w$ than $w^{\prime}$, or $w(S \cup\{i\})-w(S) \geq w^{\prime}(S \cup\{i\})-w^{\prime}(S)$ for all $S$, then $\phi_i(w) \geq \phi_i\left(w^{\prime}\right)$.
5. (Linearity) The game $w(S)=\sum_{k=1}^n c_k w_k(S)$, which is a linear combination of multiple games $\left(w_1, \ldots, w_n\right)$, has scores given by $\phi_i(w)=\sum_{k=1}^n c_k \phi_i\left(w_k\right)$

The Shapley values $\phi_i(w)$ are the unique credit allocation scheme that satisfies properties 1-5 [35], and they are given by the expression:
$$
\phi_i(w)=\frac{1}{d} \sum_{S \subseteq D \backslash\{i\}}\left(\begin{array}{c}
d-1 \\
|S|
\end{array}\right)^{-1}(w(S \cup\{i\})-w(S))
$$



**How you calculate Shapley values:**
Consider a cooperative game with $M$ players aiming at maximizing a payoff, and let $\mathcal{S} \subseteq \mathcal{M}=\{1, \ldots, M\}$ be a subset consisting of $|\mathcal{S}|$ players. Assume that we have a contribution function $v(\mathcal{S})$ that maps subsets of players to the real numbers, called the worth or contribution of coalition $\mathcal{S}$. It describes the total expected sum of payoffs the members of $\mathcal{S}$ can obtain by cooperation. The Shapley value [12] is one way to distribute the total gains to the players, assuming that they all collaborate. It is a "fair" distribution in the sense that it is the only distribution with certain desirable properties listed below. According to the Shapley value, the amount that player $j$ gets is
$$
\phi_j(v)=\phi_j=\sum_{\mathcal{S} \subseteq \mathcal{M} \backslash\{j\}} \frac{|\mathcal{S}| !(M-|\mathcal{S}|-1) !}{M !}(v(\mathcal{S} \cup\{j\})-v(\mathcal{S})), \quad j=1, \ldots, M,
$$
that is, a weighted mean over contribution function differences for all subsets $\mathcal{S}$ of players not containing player $j$. Note that the empty set $\mathcal{S}=\emptyset$ is also part of this sum. The formula can be interpreted as follows: Imagine the coalition being formed for one player at a time, with each player demanding their contribution $v(\mathcal{S} \cup\{j\})-v(\mathcal{S})$ as a fair compensation. Then, for each player, compute the average of this contribution over all permutations of all possible coalitions, yielding a weighted mean over the unique coalitions.

To illustrate the application of (1), let us consider a game with three players such that $\mathcal{M}=\{1,2,3\}$. Then, there are eight possible subsets; $\emptyset,\{1\},\{2\},\{3\},\{1,2\},\{1,3\},\{2,3\}$, and $\{1,2,3\}$. Using (1), the Shapley values for the three players are given by
$$
\begin{aligned}
\phi_1 & =\frac{1}{3}(v(\{1,2,3\})-v(\{2,3\}))+\frac{1}{6}(v(\{1,2\})-v(\{2\}))+\frac{1}{6}(v(\{1,3\})-v(\{3\}))+\frac{1}{3}(v(\{1\})-v(\emptyset)), \\
\phi_2 & =\frac{1}{3}(v(\{1,2,3\})-v(\{1,3\}))+\frac{1}{6}(v(\{1,2\})-v(\{1\}))+\frac{1}{6}(v(\{2,3\})-v(\{3\}))+\frac{1}{3}(v(\{2\})-v(\emptyset)), \\
\phi_3 & =\frac{1}{3}(v(\{1,2,3\})-v(\{1,2\}))+\frac{1}{6}(v(\{1,3\})-v(\{1\}))+\frac{1}{6}(v(\{2,3\})-v(\{2\}))+\frac{1}{3}(v(\{3\})-v(\emptyset)) .
\end{aligned}
$$
Let us also define the non-distributed gain $\phi_0=v(\emptyset)$, that is, the fixed payoff which is not associated to the actions of any of the players, although this is often zero for coalition games.

By summarizing the right hand sides above, we easily see that they add up to the total worth of the game: $\phi_0+\phi_1+$ $\phi_2+\phi_3=v(\{1,2,3\})$
The Shapley value has the following desirable properties
**How it transfers**


“Although Shapley values are an attractive solution for allocating credit among players in coalitional games, our goal is to allocate credit among features x1, . . . , xd in a machine learning model f (x) ∈ R. Machine learning models are not coalitional games by default, so to use Shapley values we must first define a coalitional game v(S) based on the model f (x) (Figure 3a). The coalitional game can be chosen to represent various model behaviors, including the model’s loss for a single sample or for the entire dataset [26], but our focus is the most common choice: explaining the prediction f (xe) for a single sample xe. When explaining a machine learning model, it is natural to view each feature xi as a player in the” (Chen et al., 2022, p. 4)

“coalitional game. However, we then must define what is meant by the presence or absence of each feature. Given our focus on a single explicand xe, the presence of feature i will mean that the model is evaluated with the observed value xe i (Figure 3b). As for the absent features, we next consider how to remove them to properly assess the influence of the present features.” (Chen et al., 2022, p. 5)



The Shapley values $\phi_i(w)$ are the unique credit allocation scheme that satisfies properties 1-5 [35], and they are given by the expression:
$$
\phi_i(w)=\frac{1}{d} \sum_{S \subseteq D \backslash\{i\}}\left(\begin{array}{c}
d-1 \\
|S|
\end{array}\right)^{-1}(w(S \cup\{i\})-w(S)) .
$$
The expression above shows that each Shapley value $\phi_i(w)$ is a weighted average of the incremental changes from adding $i$ to subsets $S \subseteq D \backslash\{i\}$. For SAGE, we propose assigning feature importance using the Shapley values of our model-based predictive power, or $\phi_i\left(v_f\right)$ for $i=1,2, \ldots, d$, which we refer to as $S A G E$ values.

Shapley-based explanations account for this by viewing variables as players in a cooperative game ${ }^{15,16}$ and measures the impact of variable $X_j$ on model $f$ based on its marginal contribution when some variables, $X_S \subset X_D$, are already present. The Shapley values are defined as:
$$
\varphi_j(w)=\frac{1}{d} \sum_{S \subseteq\{D \{j\}\}}\left(\frac{d-1}{}\right)^{-1}[w \mid(S \cup\{j\})-w(S)] \text {. (Equation 1) }
$$
$w(S)$ quantifies the contribution of subset $X_S$ to the model, which is defined differently for different types of Shapley-based variable importance measures and will be explicitly defined below for SHAP and SAGE. $|S|$ denotes the number of variables in this subset, and $\left(\begin{array}{l}d-1 \\ |S|\end{array}\right)$ is the number of ways to choose $|S|$ variables from $X_{D \backslash\{j\}} \cdot \varphi_j(w)=0$ indicates no contribution, and larger values indicate increased contribution. ${ }^{16}$

When $w(S)$ is the expectation of a single prediction, i.e., $w(S)=v_{f, x}(S)=E\left[f\left(X_D \mid X_S=x_S\right)\right], \varphi_j\left(v_{f, x}\right)$ gives the SHAP value for local explanation. ${ }^{15}$ Absolute SHAP values reflect the magnitude of variable impact, and the signs indicate the direction; therefore, the mean absolute SHAP value may be used as a heuristic global importance measure. ${ }^{15,16}$

When $w(S)$ is the expected reduction in loss over the mean prediction by including $X_S$, i.e., $w(S)=v_f(S)=E\left\{L\left(E\left[f\left(X_D\right)\right]\right.\right.$, $Y)\}-E\left\{L\left(f\left(X_D \mid X_S=X_S\right), Y\right)\right\}, \varphi_j\left(v_f\right)$ is the SAGE value for a formal global interpretation. ${ }^{16}$ Our proposed ShapleyVIC follows the VIC approach to extend the global and model-agnostic SAGE across models.

“explanations account for this by viewing variables as players in a cooperative game15,16 and measures the impact of variable Xj on model f based on its marginal contribution when some variables, XS3XD, are already present. The Shapley values are defined as: 4jðwÞ = 1 d X S4fDyfjgg d1 jSj 1 ½wðSWfjgÞ wðSÞ: (Equation 1) w(S) quantifies the contribution of subset XS to the model, which is defined differently for different types of Shapley-based variable importance measures and will be explicitly defined below for SHAP and SAGE” ([[@ningShapleyVariableImportance2022]], p. 3)

For consistency to cref-[[🧭Evaluation metric]] we use the zero-one loss as a loss function and estimate importances on the test set. -footnote(As an artificat of this thesis we contributed to the implementation of the zero-one loss.) 
