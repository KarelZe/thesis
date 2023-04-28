All of our machine learning models feature a set of tunable hyperparameters. The results of previous studies, exemplary the one of ([[@grinsztajnWhyTreebasedModels2022]]5), emphasize the need for tuning routines, as the test performance of the FT-Transformer and gradient-boosted trees largely fluctuates with the hyperparameter configuration.  For a fair comparison, we employ an exhaustive hyperparameter search, to find suitable hyperparameter configuration for each of our models. 

General overview for neural nets in [[@melisStateArtEvaluation2017]]. Also, [[@kadraWelltunedSimpleNets2021]]

**Bayesian search**
We perform a novel Bayesian search to suggest and tune the hyperparameters automatically. In Bayesian search, a prior belief for all possible objective functions is formulated from the parameter intervals, which is then gradually refined by updating the Bayesian posterior with data from previous trials thereby approximating the likely objective function ([[@shahriariTakingHumanOut2016]]2). Compared to brute-force approaches, such as grid search, unpromising search regions are omitted, resulting in fewer trials.

While different algorithmic implementations exist for Bayesian optimization, we choose the Optuna library by ([[@akibaOptunaNextgenerationHyperparameter2019]]1--10), which implements the *tree parzen estimator* and is capable of handling both continuous and categorical hyperparameters. We maximize for the accuracy on the validation set and run 50 trials per combination of model and feature set. 

![[search-space.png]]

Depth refers to the depth of trees within the ensemble. 





![[Pasted image 20230428111917.png]]

**Gradient Boosting**
Figure-Xa) visualizes the hyperparameter search space of the gls-gbm on the gls-ise dataset with classical features. We can derive several observations from it. First, hyperparameter tuning has a significant impact on the prediction, as the validation accuracy varies between (...) and (...) for different trials. Second, the best hyperparameter combination, marked in (), achieves a validation accuracy of sunitx-percent. As it lies off-the-borders surrounded by other promising trials, indicated by the contours, from which we can conclude, that the found solution is stable and reasonable for further analysis.

In Figure-Xb) we repeat the analysis for gls-gbm trained on classical-size features. The loss surface is smooth with with large connected regions. As the best solution lies within a splayed region of dense sampling, it is a good choice for further analysis. Consistent with the loss-surface of Figure-Xa), the trees are grown to the maximum depth with a high learning rate, indicating the need for complex ensemble members highly corrective to previous predictions. Part of this could be due to the low signal-to-noise ratio in financial data.

The loss surface of the gls-gbm trained on the feature set including option features is least fragmented. While the validation accuracy of the best combinations improves significantly to sunitx-percent, worst trials even under-perform these of smaller feature sets. Based on this finding we conjecture, that more data does not *per-se* improve the model and that models require a thoughtful tuning procedure. By this means, our conclusion contradict the one of ([[@ronenMachineLearningTrade2022]]14), who find no advantage from tuning their tree-based ensemble.

**Gradient Boosting + Self-Training**
The results for the gls-gbm in combination with self-training are similar and visualized in cref-a) c). To conserve space, we summarize the important findings.
(...)
- where does depth come from?
- Why just two iterations

**Classical rules**
Akin to selecting the machine learning classifiers, we determine our classical baselines on the gls-ise validation set. This prevents overfitting the test set and maintains consistency between both paradigms. For the same reason, baselines are kept constant in the transfer setting on the gls-cboe sample. Entirely for reference, we also report accuracies of the tick rule, quote rule, and gls-lr algorithm, due to their widespread adoption in literature.

While optimizing the combination of trade classification rules through Bayesian search is theoretically feasible, we found no out-performance over hybrid rules reported in literature \footnote{We performed a Bayesian search with 50 trials for trade classification rules, stacking up to five rules. Experiment available under: \url{https://wandb.ai/fbv/thesis}}.  Thus, \cref{tab:ise-classical-hyperparam-classical-size} reports the accuracies of common trade classification rules on the \gls{ISE} validation set.

**Notes:**
[[💡Hyperparameter tuning notes]]