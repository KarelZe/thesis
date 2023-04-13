All of our machine learning models feature a set of tunable hyperparameters. The results of previous studies, exemplary the one of ([[@grinsztajnWhyTreebasedModels2022]]5), emphasize the need for tuning routines, as the test performance of the FT-Transformer and gradient-boosted trees largely fluctuates with the hyperparameter configuration.  For a fair comparison, we employ an exhaustive hyperparameter search, to find suitable hyperparameter configuration for each of our models. 

**Bayesian search**
We perform a novel Bayesian search to suggest and tune the hyperparameters automatically. In Bayesian search, a prior belief for all possible objective functions is formulated from the parameter intervals, which is then gradually refined by updating the Bayesian posterior with data from previous trials thereby approximating the likely objective function ([[@shahriariTakingHumanOut2016]]2). Compared to brute-force approaches, such as grid search, unpromising search regions are omitted, resulting in fewer trials.

While different algorithmic implementations exist for Bayesian optimization, we choose the Optuna library by ([[@akibaOptunaNextgenerationHyperparameter2019]]1--10), which implements the *tree parzen estimator* and is capable of handling both continuous and categorical hyperparameters. We maximize for the accuracy on the validation set and run 50 trials per combination of model and feature set. 

(Table)

Our search space is reported in Table-X, which we laid out based on the recommendations in ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]]20) and ([[@gorishniyRevisitingDeepLearning2021]]18) and ([[@rubachevRevisitingPretrainingObjectives2022]]., 2022, p. 4) with minor deviations. For gradient-boosting we raise the border count to $256$, which increases the number of split candidates per feature through a finer quantization, Expectedly, accuracy increases at the cost of computational efficiency. The size of the ensemble $M$ may not be fully exhausted. Acknowleding the observations ([[@friedmanGreedyFunctionApproximation2001]]14), that the learning rate $\lambda$ the learning rate and the size of the ensemble have a strong interdependence, we only tune the learning rate and stop adding new trees to the ensemble, once the validation accuracy decreases for consecutive (...) steps.

The hyperparameter search for the FT-Transformer is identical to ([[@gorishniyRevisitingDeepLearning2021]]18) variant (b). From preliminary tests, we observed that the use of a learning rate schedule with a short learning rate warm-up phase both stabilizes training and improves accuracy (cp. cref-training-of-supervised). Their constant learning rate and our decayed learning rate may thus not be entirely comparable. Additionally, we employ early stopping and halt training after 15 consecutive decreases in validation accuracy, affecting the effective number of epochs. Both techniques have not been used by the orginal author's to provide a conservative baseline ([[@gorishniyRevisitingDeepLearning2021]]5), for the sake of a fair comparison in our context both techniques should be used.

**What others wrote**

> “Hyperparameters & Evaluation. Hyperparameter tuning is crucial for a fair comparison, therefore, we use Optuna [1] to optimize the model and pretraining hyperparameters for each method on each dataset. We use the validation subset of each dataset for hyperparameter tuning. The exact search spaces for the hyperparameters of each method are provided in Appendix B.” ([[@rubachevRevisitingPretrainingObjectives2022]]., 2022, p. 4)

> "“Tuning. For every dataset, we carefully tune each model’s hyperparameters. The best hyperparameters are the ones that perform best on the validation set, so the test set is never used for tuning. For most algorithms, we use the Optuna library (Akiba et al., 2019) to run Bayesian optimization (the Tree-Structured Parzen Estimator algorithm), which is reported to be superior to random search ([[@turnerBayesianOptimizationSuperior2021]]). For the rest, we iterate over predefined sets of configurations recommended by corresponding papers. We provide parameter spaces and grids in supplementary. We set the budget for Optuna-based tuning in terms of iterations and provide additional analysis on setting the budget in terms of time in supplementary.” ([[@gorishniyRevisitingDeepLearning2021]], p. 6)"

> Implementation. We fix and do not tune the following hyperparameters: • early-stopping-rounds = 50 • od-pval = 0.001 • iterations = 2000 In Table 17, we provide hyperparameter space used for Optuna-driven tuning (Akiba et al., 2019). We set the task_type parameter to “GPU” (the tuning was unacceptably slow on CPU). Evaluation. We set the task_type parameter to “CPU”, since for the used version of the CatBoost library it is crucial for performance in terms of target metrics.


> “These black-box optimization problems are often solved using Bayesian optimization (BO) methods [19]. BO methods rely on a (probabilistic) surrogate model for the objective function that provides a measure of uncertainty. This model is often a Gaussian process (GP) [55], but other models such as Bayesian neural networks are also commonly used as long as they provide a measure of uncertainty. Using this surrogate model, an acquisition function is used to determine the most promising point to evaluate next, where popular options include expected improvement (EI) [35], knowledge gradient (KG) [18], and entropy search (ES) [31]. There are also other surrogate optimization methods that rely on deterministic surrogate models such as radial basis functions [16, 66], see Forrester et al. [17] for an overview. The choice of surrogate model and acquisition function are both problem-dependent and the goal of this challenge is to compare. Using this surrogate model, an acquisition function is used to determine the most promising point to evaluate next, where popular options include expected improvement (EI) [35], knowledge gradient (KG) [18], and entropy search (ES) [31]. There are also other surrogate optimization methods that rely on deterministic surrogate models such as radial basis functions [16, 66], see Forrester et al. [17] for an overview. different approaches over a large number of different problems. This was the first challenge aiming to find the best black box optimizers specially for ML-related problems.” ([[@turnerBayesianOptimizationSuperior2021]] p. 2)

**Tree-Parzen Estimator**
> “TPE is another variant of BO that performs well in general and can be utilized for both categorical and continuous types of hyperparameters. Unlike BOGP, which has cubical time complexity, TPE runs in linear time. TPE is suggested if you have a huge hyperparameter space and have a very tight budget for evaluating the cross-validation score. The main difference between TPE and BOGP or SMAC is in the way that it models the relationship between hyperparameters and the cross-validation score. Unlike BOGP or SMAC, which approximate the value of the objective function, or the posterior probability, 𝑝𝑝(𝑦𝑦|𝑥𝑥), TPE works the other way around. It tries to get the optimal hyperparameters based on the condition of the objective function, or the likelihood probability, 𝑝𝑝(𝑥𝑥|𝑦𝑦) (see the explanation of Bayes Theorem in the Understanding BO GP section)” (Owen, 2022, p. 51)

> “Exploring Bayesian Optimization 52 In other words, unlike BOGP or SMAC, which construct a predictive distribution over the objective function, TPE tries to utilize the information of the objective function to model the hyperparameter distributions. To be more precise, when the optimization problem is in the form of a minimization problem, 𝑝𝑝(𝑥𝑥|𝑦𝑦) is defined as follows: 𝑝𝑝(𝑥𝑥|𝑦𝑦) = 𝑙𝑙(𝑥𝑥) 𝑖𝑖𝑖𝑖 𝑦𝑦 < 𝑦𝑦∗ 𝑎𝑎𝑎𝑎𝑎𝑎 𝑔𝑔(𝑥𝑥) 𝑖𝑖𝑖𝑖 𝑦𝑦 ≥ 𝑦𝑦∗ Here, 𝑙𝑙(𝑥𝑥) and 𝑔𝑔(𝑥𝑥) are utilized when the value of the objective function is lower or higher than the threshold, 𝑦𝑦∗ , respectively. There is no specific rule on how to choose the threshold, 𝑦𝑦∗ . However, in the Hyperopt and Microsoft NNI implementations, this threshold is chosen based on the TPE’s hyperparameter, 𝛾𝛾, and the number of observed points in D up to the current trial. The definition of 𝑝𝑝(𝑥𝑥|𝑦𝑦) tells us that TPE has two models that act as the learning algorithm based on the value of the objective function, which is ruled by the threshold, 𝑦𝑦∗ . When the distribution of hyperparameters is continuous, TPE will utilize Gaussian mixture models (GMMs), along with the EI acquisition function, to suggest the next set of hyperparameters to be tested. If the continuous distribution is not a Gaussian distribution, then TPE will convert it to mimic the Gaussian distribution. For example, if the specified hyperparameter distribution is the uniform distribution, then it will be converted into a truncated Gaussian distribution. The probabilities of the different possible outcomes for the multinomial distribution within the GMM, and the mean and variance values for the normal distribution within the GMM, are generated by the adaptive Parzen estimator. This estimator is responsible for constructing the two probability distributions, 𝑙𝑙(𝑥𝑥) and 𝑔𝑔(𝑥𝑥), based on the mean and variance of the normal hyperparameter distribution, as well as the hyperparameter value of all observed points in D up to the current trial. When the distribution is categorical or discrete, TPE will convert the categorical distribution into a re-weighted categorical and use weighted random sampling, along with the EI acquisition function, to suggest the expected best set of hyperparameters. The weights in the random sampling procedure are generated based on the historical counts of the hyperparameter value. The EI acquisition function definition in TPE is a bit different from the definition we learned about in the Introducing BO section. In TPE, we are using Bayes Theorem when deriving the EI formula. The simple formulation of the EI acquisition function in TPE is defined as follows: 𝐸𝐸𝐸𝐸(𝑥𝑥) ∝ 𝑙𝑙(𝑥𝑥) 𝑔𝑔(𝑥𝑥) The proportionality defined here tells us that to get a high value of EI, we need to get a high 𝑙𝑙(𝑥𝑥) 𝑔𝑔(𝑥𝑥) ratio. In other words, when the optimization problem is in the form of a minimization problem, the EI acquisition function must suggest more hyperparameters from 𝑙𝑙(𝑥𝑥) over 𝑔𝑔(𝑥𝑥) . It is the other way around when the optimization problem is in the form of a maximization problem. For example, when we use accuracy to measure the performance of our classification model, then we should sample more hyperparameters from 𝑔𝑔(𝑥𝑥) over 𝑙𝑙(𝑥𝑥)” (Owen, 2022, p. 52)

> “Understanding TPE 53 To summarize, TPE works as follows. Note that the following procedure describes how TPE works for the minimization problem. This procedure replaces Steps 7 to 11 in the Introducing BO section: 6. (The first few steps are the same as we saw earlier). 7. Divide pairs of hyperparameter values and cross-validation scores in D into two groups based on the threshold, 𝑦𝑦∗, namely below and above groups (see Figure 4.19). 8. Sample the next set of hyperparameters by utilizing the EI acquisition function: I. For each group, calculate the probabilities, means, and variances for the GMM using the adaptive Parzen estimator (if it’s a continuous type) or weights for random sampling (if it’s a categorical type). II. For each group, fit the GMM (if it’s a continuous type), or perform random sampling (if it’s a categorical type), to sample which hyperparameters will be passed to the EI acquisition function. III.For each group, calculate the probability of those samples being good samples (for the below group), or the probability of those samples being bad samples (for the above group). IV. Get the expected optimal set of hyperparameters based on the EI acquisition function. 9. Compute the cross-validation score using the objective function, f, based on the output from Step 8. 10. Add the hyperparameters and cross-validation score pair from Step 8 and Step 9 to set D. 11. Repeat Steps 7 to 10 until the stopping criteria have been met. 12. (The last few steps are the same as we saw earlier): Figure 4.20 – Illustration of groups division in TP” (Owen, 2022, p. 53)

 > Based on the stated procedure and the preceding plot, we can see that, unlike BOGP or SMAC, which constructs a predictive distribution over the objective function, TPE tries to utilize the information of the objective function to model the hyperparameter distributions. This way, we are not only focusing on the best-observed points during the trials – we are focusing on the distribution of the best-observed points instead. You may be wondering why the Tree-structured term is within the TPE method’s name. This term refers to the conditional hyperparameters that we discussed in the previous section. This means that there are hyperparameters in the space that will only be utilized when a certain condition is met. We will see what a tree-structured or conditional hyperparameter space looks like in Chapter 8, Hyperparameter Tuning via Hyperopt, and Chapter 9, Hyperparameter Tuning via Optuna. One of the drawbacks that TPE has is that it may overlook the interdependencies among hyperparameters in a certain space since the Parzen estimators work univariately. However, this is not the case for BOGP or SMAC, since the surrogate model is constructed based on the configurations in the hyperparameter space. Thus, they can take into account the interdependencies among hyperparameters. Fortunately, there is an implementation of TPE that overcomes this drawback. The Optuna package provides the multivariate T” ([[@owenHyperparameterTuningPython2022]], p. 54)

**Why hyperparam tuning is necessary**
> “Hyperparameter tuning leads to uncontrolled variance on a benchmark [Bouthillier et al., 2021], especially with a small budget of model evaluations. We design a benchmarking procedure that jointly samples the variance of hyperparameter tuning and explores increasingly high budgets of model evaluations. It relies on random searches for hyper-parameter tuning [Bergstra et al., 2013]. We use hyperparameter search spaces from the Hyperopt-Sklearn [Komer et al., 2014] when available, from the original paper when possible, and from Gorishniy et al. [2021] for MLP, Resnet and XGBoost (see A.3). We run a random search of ≈ 400 iterations per dataset, on CPU for tree-based models and GPU for neural networks (more details in A.3). To study performance as a function of the number n of random search iterations, we compute the best hyperparameter combination on the validation set on these n iterations (for each model and dataset), and evaluate it on the test set. We do this 15 times while shuffling the random search order at each time. This gives us bootstrap-like estimates of the expected test score of the best (on the validation set) model after each number of random search iterations. In addition, we always start the random searches with the default hyperparameters of each model. In A.7, we show that using Bayesian optimization instead of random search does not seem to change our results.” ([[@grinsztajnWhyTreebasedModels2022]], 2022, p. 4)




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



Implementation. We fix and do not tune the following hyperparameters: • early-stopping-rounds = 50 • od-pval = 0.001 • iterations = 2000 In Table 17, we provide hyperparameter space used for Optuna-driven tuning (Akiba et al., 2019). We set the task_type parameter to “GPU” (the tuning was unacceptably slow on CPU). Evaluation. We set the task_type parameter to “CPU”, since for the used version of the CatBoost library it is crucial for performance in terms of target metrics.



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
