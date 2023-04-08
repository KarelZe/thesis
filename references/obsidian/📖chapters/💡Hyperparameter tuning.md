All of our machine learning models feature a set of tunable hyperparameters. The results of previous studies, exemplary the one of ([[@grinsztajnWhyTreebasedModels2022]] 5), emphasize the need for tuning routines, as the test performance of the FT-Transformer and gradient-boosted trees largely fluctuates with the hyperparameter configuration.  As such, we employ an exhaustive hyperparameter search, to find suitable hyperparameter configuration for all our models. 


We perform a novel Bayesian search built  on top of the tree parzen algorithm to suggest and tune the hyperparameters automatically. In Bayesian search, a prior belief for all possible objective functions is formulated from the parameter intervals, which is then gradually refined by updating the Bayesian posterior with data from previous trials thereby approximating the likely objective function ([[@shahriariTakingHumanOut2016]]). Compared to brute-force approaches, such as grid search, unpromising search regions are omitted, resulting in fewer trials.

Our search space is reported in Table-X, which we laid out based on the recommendations in ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]] 999) and ([[@gorishniyRevisitingDeepLearning2021]]999) with minor deviations. For gradient-boosting we raise the border count to $256$, which increases the number of possible split candidates per feature through a finer quantization.


Implementation. We fix and do not tune the following hyperparameters: • early-stopping-rounds = 50 • od-pval = 0.001 • iterations = 2000 In Table 17, we provide hyperparameter space used for Optuna-driven tuning (Akiba et al., 2019). We set the task_type parameter to “GPU” (the tuning was unacceptably slow on CPU). Evaluation. We set the task_type parameter to “CPU”, since for the used version of the CatBoost library it is crucial for performance in terms of target metrics.

As we were experiencing exploding gradients in preliminary tests for the FT-Transformer due to too high learning rates, we downward adjust the intervals for the learning rate. Lower learning rates result in smaller weight updates which prevents overshooting in local loss minima, but also requires more training epochs. 



```python
        # kaggle book + https://catboost.ai/en/docs/concepts/parameter-tuning
        # friedman paper
learning_rate = trial.suggest_float("learning_rate", 0.001, 0.125, log=True)
        depth = trial.suggest_int("depth", 1, 12)
        l2_leaf_reg = trial.suggest_int("l2_leaf_reg", 2, 30)
        random_strength = trial.suggest_float("random_strength", 1e-9, 10.0, log=True)
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)

        kwargs_cat = {
            "iterations": 2000,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "grow_policy": "Lossguide",
            "border_count": 254,
            "logging_level": "Silent",
            "task_type": task_type,
            "devices": devices,
            "random_seed": set_seed(),
            "eval_metric": "Accuracy",
            "early_stopping_rounds": 100,

        }
```




![[Pasted image 20230408165957.png]]

We report the 

Implementation. We fix and do not tune the following hyperparameters: • early-stopping-rounds = 50 • od-pval = 0.001 • iterations = 2000 In Table 17, we provide hyperparameter space used for Optuna-driven tuning (Akiba et al., 2019). We set the task_type parameter to “GPU” (the tuning was unacceptably slow on CPU). Evaluation. We set the task_type parameter to “CPU”, since for the used version of the CatBoost library it is crucial for performance in terms of target metrics.




We define a hyperparameter search space and run a Bayesian optimization 

We run a Bayesian search and optimize for the accuracy, which is also our decisive metric for evaluation ([[🧭Evaluation metric]]), on the validation set. 
We compute at maximum 50 trials. To compensate for varying computational costs between both machine learning models, we set an additional time budget of 12 hours.

We grow symmetric trees, which acts as a regularizer.


Figure-X visualizes the hyperparameter search space. It serves two purposes,


Tuning. For every dataset, we carefully tune each model’s hyperparameters. The best hyperparameters are the ones that perform best on the validation set, so the test set is never used for tuning. For most algorithms, we use the Optuna library (Akiba et al., 2019) to run Bayesian optimization (the Tree-Structured Parzen Estimator algorithm), which is reported to be superior to random search (Turner et al., 2021). For the rest, we iterate over predefined sets of configurations recommended by corresponding papers. We provide parameter spaces and grids in supplementary. We set the budget for Optuna-based tuning in terms of iterations and provide additional analysis on setting the budget in terms of time in supplementary


- Visualize results https://github.com/LeoGrin/tabular-benchmark
![[comparsion-of-results.png]]

- we set a time budget and hyperparameter constraints
- repeat with different random initializations e. g., use first https://de.wikipedia.org/wiki/Mersenne-Zahl as seeds
- there is a trade-off between robustness of results and the computational effort / search space

- Explain the importance why hyperparam tuning deserves its own chapter. - > even simple architectures can obtain SOTA-results with proper hyperparameter settings. -> See in-depth analysis in [[@melisStateArtEvaluation2017]] (found in [[@kadraWelltunedSimpleNets2021]])
- [[@melisStateArtEvaluation2017]] investigate hyperparam tuning by plotting validation losses against the hyperparams. 
- ![[validation-loss-vs-hyperparam.png]]
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
![[sample-validation-curve.png]]
When using optuna draw a boxplot. optimal value should lie near the median. Some values should be outside the IQR.
![[optuna-as-boxplot.png]]

- compare results of untuned and tuned models. Similar to [[@gorishniyRevisitingDeepLearning2021]].

[[💡Hyperparameter tuning]]

Repeat search with different random initializations:
![[random-searches-hyperparms.png]]
(found in [[@grinsztajnWhyTreebasedModels2022]])

Show differences from different initializations using a violin plot. (suggested in [[@melisStateArtEvaluation2017]])

- For tree-parzen estimator see: https://neptune.ai/blog/optuna-guide-how-to-monitor-hyper-parameter-optimization-runs
- Framing hyperparameter search as an optimization problem. https://www.h4pz.co/blog/2020/10/3/optuna-and-wandb
- perform ablation study (https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)) when making important changes to the architecture. This has been done in [[@gorishniyRevisitingDeepLearning2021]].
- For implementation of permutation importance see https://www.rasgoml.com/feature-engineering-tutorials/how-to-generate-feature-importance-plots-using-catboost

![[1680261360735 2.jpg]]