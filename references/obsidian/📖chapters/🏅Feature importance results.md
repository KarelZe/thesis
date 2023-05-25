**SAGE**
- emphasize that these are global feature attributions
- explain definition of feature groups -> Limitation of implementation. Classical classifier sees only a fraction of all features, but features are inherently redundant.
- configuration
	- explain how sampling is done? Why is sampling even necessary.
	- why zero-one loss and why not cross-entropy loss? -> penalize trade classification rules for over-confident predictions
- visualize in subplots how feature importances align (3x3 (benchmark, gbm, fttransformer), distinguished by feature set)
- What features are important?
	- Are there particularly dominant feature groups?
	- How does it align with literature?
	- Are features that are important in smaller feature sets also important in larger feature sets?
	- How does adding more features influence the impact of the other features?
	- Which ones are unimportant? Do models omit features that perform poorly in the empirical setting?
	- Are features important that have not been considered yet? Why's that?
- interpretation
	- Why are size-related features so important? Can we confirm the limit order theory? 

![[informative-uniformative-features.png]]
([[@grinsztajnWhyTreebasedModels2022]])

**Attention**
- emphasize that these are local feature attributions
- Visualize attention for some trades
- interpret pattern. How does it align with the feature importances from SAGE?

**Rank correlation between approaches**
Compare different feature attributions:
![[feature_attributions_from_attention.png]]
(Found in [[@borisovDeepNeuralNetworks2022]])
