- Visualize results https://github.com/LeoGrin/tabular-benchmark
![[comparsion-of-results.png]]

- we set a time budget and hyperparameter constraints
- repeat with different random initializations e. g., use first https://de.wikipedia.org/wiki/Mersenne-Zahl as seeds
- there is a trade-off between robustness of results and the computational effort / search space

- Explain the importance why hyperparam tuning deserves its own chapter. - > even simple architectures can obtain SOTA-results with proper hyperparameter settings. -> See in-depth analysis in [[@melisStateArtEvaluation2017]] (found in [[@kadraWelltunedSimpleNets2021]])
- [[@melisStateArtEvaluation2017]] investigate hyperparam tuning by plotting validation losses against the hyperparams. 
- ![[validation-loss-vs-hyperparam 1.png]]
- [[@melisStateArtEvaluation2017]] also they try out different seeds. Follow their recommendations.
- See e. g., [[@olbrysEvaluatingTradeSide2018]][[@owenHyperparameterTuningPython2022]] for ideas / most adequate application.
- What optimizer is chosen? Why? Could try out Adam or Adan?
- Start with something simple like GridSearch. Implement in Optuna, so that one can easily switch between grid search, randomized search, Bayesian search etc. [09_Hyperparameter-Tuning-via-Optuna.ipynb - Colaboratory (google.com)](https://colab.research.google.com/github/PacktPublishing/Hyperparameter-Tuning-with-Python/blob/main/09_Hyperparameter-Tuning-via-Optuna.ipynb#scrollTo=580226e9-cc08-4dc7-846b-914876343071) 
- For optuna integration into weights and biases see [this article.](https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893)
- Perform comparsion between different samplers to study how sampler effects parameter search. e. g. see best estimate after $n$ trials.
- Also possible to optimize for multiple objectives e. g., accuracy and ... [optuna.visualization.plot_pareto_front — Optuna 3.0.2 documentation](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_pareto_front.html)
- See reasoning towards Bayesian search in my last paper. (see e. g., [[@shahriariTakingHumanOut2016]]) 
- For implementations on tab transformer, tabnet and tabmlp see: pytorch wide-deep package.
- for most important hyperparams in litegbm, catboost etc. (see [[@banachewiczKaggleBookData2022]])
- Visualize training and validation curves (seimilar to [3.4. Validation curves: plotting scores to evaluate models — scikit-learn 1.1.2 documentation](https://scikit-learn.org/stable/modules/learning_curve.html))
![[sample-validation-curve 1.png]]
When using optuna draw a boxplot. optimal value should lie near the median. Some values should be outside the IQR.
![[optuna-as-boxplot 1.png]]

- compare results of untuned and tuned models. Similar to [[@gorishniyRevisitingDeepLearning2021]].

[[💡Hyperparameter tuning]]

Repeat search with different random initializations:
![[random-searches-hyperparms 1.png]]
(found in [[@grinsztajnWhyTreebasedModels2022]])

Show differences from different initializations using a violin plot. (suggested in [[@melisStateArtEvaluation2017]])

- For tree-parzen estimator see: https://neptune.ai/blog/optuna-guide-how-to-monitor-hyper-parameter-optimization-runs
- Framing hyperparameter search as an optimization problem. https://www.h4pz.co/blog/2020/10/3/optuna-and-wandb
- perform ablation study (https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)) when making important changes to the architecture. This has been done in [[@gorishniyRevisitingDeepLearning2021]].
- For implementation of permutation importance see https://www.rasgoml.com/feature-engineering-tutorials/how-to-generate-feature-importance-plots-using-catboost