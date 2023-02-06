- local vs. global attention
- Visualize attention
- make models comparable. Find a notion of feature importance that can be shared across models.
 - compare feature importances between approaches like in paper
 - How do they selected features relate to what is being used in classical formulas? (see [[#^ce4ff0]]) Could a hybrid formula be derived from the selection by the algorithm?
 - What is the economic intuition?
- Use [Captum · Model Interpretability for PyTorch](https://captum.ai/) to learn what the model picks up as a relevant feature.

![[informative-uniformative-features.png]]
[[@grinsztajnWhyTreebasedModels2022]]
Interesting comments: https://openreview.net/forum?id=Fp7__phQszn
- Most finance papers e. g., [[@finucaneDirectTestMethods2000]] (+ other examples as reported in expose) use logistic regression to find features that affect the classification most. Poor choice due to linearity assumption? How would one handle categorical variables? If I opt to implement logistic regression, also report $\chi^2$.
- Think about approximating SHAP values on a sample or using some improved implementation like https://github.com/linkedin/FastTreeSHAP
- Do ablation studies. That is, the removal of one feature shouldn't cause the model to collapse. (idea found in [[@huyenDesigningMachineLearning]])
- discuss and address correlations of features

Compare different feature attributions:
![[feature_attributions_from_attention.png]]
(Found in [[@borisovDeepNeuralNetworks2022]])
