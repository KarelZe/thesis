- Feature Importance of Gradient Boosted Trees
	- Possibilities to calculate feature importances in GBMs [here.](https://blog.tensorflow.org/2019/03/how-to-train-boosted-trees-models-in-tensorflow.html)
- Feature Importance of TabNet
	- allows to obtain both local / and global importance
- Feature Importance of TabTransformer
- Unified Approach for Feature Importance
	- Make feature importances comparable across models.
	- For simple methods see permutation importance, ice and coutnerfactuals (see [8.5 Permutation Feature Importance | Interpretable Machine Learning (christophm.github.io)](https://christophm.github.io/interpretable-ml-book/feature-importance.html))
	- Open question: How are correlations handled in SHAP?
	- Think about using kernel-shap. Could work. See e. g., [Feature importance in deep learning - Deep Learning - Deep Learning Course Forums (fast.ai)](https://forums.fast.ai/t/feature-importance-in-deep-learning/42026/91?page=4) and [Census income classification with Keras — SHAP latest documentation](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/neural_networks/Census%20income%20classification%20with%20Keras.html)
	- If SHAP is to complex, one could just zero-out features like in [[@guEmpiricalAssetPricing2020]], but be aware of drawbacks. Yet similar method is to randomly permutate features "within a column" and see how to prediction changes" (see [[@banachewiczKaggleBookData2022]]) also comes at the advantage that no retraining is needed, but artificially breaks correlations etc. (see my previous seminar paper).
	- [[@ronenMachineLearningTrade2022]] study the feature importance only for random forests.

- For explanation on SHAP and difference between pre- and post-model explainability see [[@baptistaRelationPrognosticsPredictor2022]]

![[feature_attributions.png]]
(compare kernelshap with feature attributions from attention [[@borisovDeepNeuralNetworks2022]])
