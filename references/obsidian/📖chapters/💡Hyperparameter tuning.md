All of our machine learning models feature a set of tunable hyperparameters. The results of previous studies, exemplary the one of ([[@grinsztajnWhyTreebasedModels2022]]5), emphasize the need for tuning routines, as the test performance of the FT-Transformer and gradient-boosted trees largely fluctuates with the hyperparameter configuration.  For a fair comparison, we employ an exhaustive hyperparameter search, to find suitable hyperparameter configuration for each of our models. 

 General overview for neural nets in [[@melisStateArtEvaluation2017]].

**Bayesian search**
We perform a novel Bayesian search to suggest and tune the hyperparameters automatically. In Bayesian search, a prior belief for all possible objective functions is formulated from the parameter intervals, which is then gradually refined by updating the Bayesian posterior with data from previous trials thereby approximating the likely objective function ([[@shahriariTakingHumanOut2016]]2). Compared to brute-force approaches, such as grid search, unpromising search regions are omitted, resulting in fewer trials.

While different algorithmic implementations exist for Bayesian optimization, we choose the Optuna library by ([[@akibaOptunaNextgenerationHyperparameter2019]]1--10), which implements the *tree parzen estimator* and is capable of handling both continuous and categorical hyperparameters. We maximize for the accuracy on the validation set and run 50 trials per combination of model and feature set. 

(Table)

Our search space is reported in Table-X, which we laid out based on the recommendations in ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]]20) and ([[@gorishniyRevisitingDeepLearning2021]]18) and ([[@rubachevRevisitingPretrainingObjectives2022]]., 2022, p. 4) with minor deviations. For gradient-boosting we raise the border count to $256$, which increases the number of split candidates per feature through a finer quantization, Expectedly, accuracy increases at the cost of computational efficiency. The size of the ensemble $M$ may not be fully exhausted. Acknowleding the observations ([[@friedmanGreedyFunctionApproximation2001]]14), that the learning rate $\lambda$ the learning rate and the size of the ensemble have a strong interdependence, we only tune the learning rate and stop adding new trees to the ensemble, once the validation accuracy decreases for consecutive (...) steps.

We grow symmetric trees, which acts as a regularizer.

The hyperparameter search for the FT-Transformer is identical to ([[@gorishniyRevisitingDeepLearning2021]]18) variant (b). From preliminary tests, we observed that the use of a learning rate schedule with a short learning rate warm-up phase both stabilizes training and improves accuracy (cp. cref-training-of-supervised). Their constant learning rate and our decayed learning rate may thus not be entirely comparable. Additionally, we employ early stopping and halt training after 15 consecutive decreases in validation accuracy, affecting the effective number of epochs. Both techniques have not been used by the orginal author's to provide a conservative baseline ([[@gorishniyRevisitingDeepLearning2021]]5), for the sake of a fair comparison in our context both techniques should be used.

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


**Gradient Boosting**
Figure-Xa) visualizes the hyperparameter search space of the gls-gbm on the gls-ise dataset with classical features. We can derive several observations from it. First, hyperparameter tuning has a significant impact on the prediction, as the validation accuracy varies between (...) and (...) for different trials. Second, the best hyperparameter combination, marked in (), achieves a validation accuracy of sunitx-percent. As it lies off-the-borders surrounded by other promising trials, indicated by the contours, from which we can conclude, that the found solution is stable and reasonable for further analysis.

In Figure-Xb) we repeat the analysis for gls-gbm trained on classical-size features. The loss surface is smooth with with large connected regions. As the best solution lies within a splayed region of dense sampling, it is a good choice for further analysis. Consistent with the loss-surface of Figure-Xa), the trees are grown to the maximum depth with a high learning rate, indicating the need for complex ensemble members highly corrective to previous predictions. Part of this could be due to the low signal-to-noise ratio in financial data.

The loss surface of the gls-gbm trained on the feature set including option features is least fragmented. While the validation accuracy of the best combinations improves significantly to sunitx-percent, worst trials even under-perform these of smaller feature sets. Based on this finding we conjecture, that more data does not *per-se* improve the model and that models require a thoughtful tuning procedure. By this means, our conclusion contradict the one of ([[@ronenMachineLearningTrade2022]]14), who find no advantage from tuning their tree-based ensemble.

**Gradient Boosting + Self-Training**
The results for the gls-gbm in combination with self-training are similar and visualized in cref-a) c). To conserve space, we summarize the important findings.
(...)



- we set a time budget and hyperparameter constraints

- there is a trade-off between robustness of results and the computational effort / search space

- Explain the importance why hyperparam tuning deserves its own chapter. - > even simple architectures can obtain SOTA-results with proper hyperparameter settings. -> See in-depth analysis in [[@melisStateArtEvaluation2017]] (found in [[@kadraWelltunedSimpleNets2021]])


- Start with something simple like GridSearch. Implement in Optuna, so that one can easily switch between grid search, randomized search, Bayesian search etc. [09_Hyperparameter-Tuning-via-Optuna.ipynb - Colaboratory (google.com)](https://colab.research.google.com/github/PacktPublishing/Hyperparameter-Tuning-with-Python/blob/main/09_Hyperparameter-Tuning-via-Optuna.ipynb#scrollTo=580226e9-cc08-4dc7-846b-914876343071) 
- For optuna integration into weights and biases see [this article.](https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893)
- Perform comparsion between different samplers to study how sampler effects parameter search. e. g. see best estimate after $n$ trials.
- Also possible to optimize for multiple objectives e. g., accuracy and ... [optuna.visualization.plot_pareto_front — Optuna 3.0.2 documentation](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_pareto_front.html)


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

**Notes:**
[[💡Hyperparameter tuning notes]]