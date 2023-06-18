All of our machine learning models feature a set of tunable hyperparameters. The results of previous studies, exemplary the one of ([[@grinsztajnWhyTreebasedModels2022]]5), emphasise the need for tuning routines, as the test performance of the FT-Transformer and gradient-boosted trees largely fluctuates with the hyperparameter configuration.  For a fair comparison, we employ an exhaustive hyperparameter search, to find suitable hyperparameter configuration for each of our models. 

**Bayesian search**
We perform a novel Bayesian search to suggest and tune the hyperparameters automatically. In Bayesian search, a prior belief for all possible objective functions is formulated from the parameter intervals, which is then gradually refined by updating the Bayesian posterior with data from previous trials thereby approximating the likely objective function ([[@shahriariTakingHumanOut2016]]2). Compared to brute-force approaches, such as grid search, unpromising search regions are omitted, resulting in fewer trials.

While different algorithmic implementations exist for Bayesian optimisation, we choose the *Optuna* library by ([[@akibaOptunaNextgenerationHyperparameter2019]]1--10), which implements the *tree parzen estimator* and is capable of handling both continuous and categorical hyperparameters. We maximise for the accuracy on the validation set and run 50 trials per combination of model and feature set. 

![[search-space.png]]

(Where does search space come from?)

**Gradient Boosting:**

As documented in cref-table, we tune five hyperparameters. Depth refers to the depth or number of levels of the regression trees. Other than ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]]), we increase the upper bound to twelve to allow for more complex ensemble members. The learning rate scales the contribution of individual trees to the ensemble. Random strength, bagging temperature, and $\ell_2$ leaf regularisation are all measures to counter overfitting. Specifically, random strength controls the degree of Gaussian noise added to the scores of split candidates to introduce randomness in the selected splits. In a similar vain, the algorithm introduces randomness on the sample level through Bayesian bootstrap. The hyperparameter controls the distribution used for sampling, and thereby the aggressiveness of Bagging. Finally, $\ell_2$ leaf regularisation adds a penalty term to the terminal leaf's estimates. The hyperparameter controls the degree of regularisation.

**FT-Transformer:**

Analogously, we define a hyperparameter search space for FT-Transformer based on ([[@gorishniyRevisitingDeepLearning2021]]18). We vary the layer count and the embedding dimension, which directly affect the capacity of the network. Layers refers to the number of layers in the encoder stack. The dimension of numerical and continuous embeddings $d_e$ is at 256 at maximum, which is half the dimension used in the author's work. We make this sacrifice, due to being computation-bound by the size of the dataset. Dropout ([[@srivastavaDropoutSimpleWay]]1930) in the attention modules and the gls-FFN is used to prevent overfitting the training data. We treat the weight decay term in the weight update rule of *AdamW* optimizer as a hyperparameter, with larger values for $\lambda$ enforcing a stronger shrinkage of weights and thereby reducing overfitting.

![[hyperparameter-ft-transformer.png]]

**Transformer With Pre-Training**
The hyperparameter search space for Transformers with a pre-training objective is identical to that shown in cref-table-hyperparameters. Following ([[@rubachevRevisitingPretrainingObjectives2022]]4), we share the learning rate and weight decay for both the pre-training and fine-tuning stages. Given the nature of pre-training, all other hyperparameters related to the model are identical.

**Gradient-Boosting with Self-Training** 
The search space for the semi-supervised variant is identical to the supervised gradient boosting.


**Gradient Boosting**
Figure-Xa) visualises the hyperparameter search space of the gls-gbm on the gls-ise dataset with classical features. We can derive several observations from it. First, hyperparameter tuning has a significant impact on the prediction, as the validation accuracy varies between (...) and (...) for different trials. Second, the best hyperparameter combination, marked in (), achieves a validation accuracy of sunitx-percent. As it lies off-the-borders surrounded by other promising trials, indicated by the contours, from which we can conclude, that the found solution is stable and reasonable for further analysis.

In Figure-Xb) we repeat the analysis for gls-gbm trained on classical-size features. The loss surface is smooth with with large connected regions. As the best solution lies within a splayed region of dense sampling, it is a good choice for further analysis. Consistent with the loss-surface of Figure-Xa), the trees are grown to the maximum depth with a high learning rate, indicating the need for complex ensemble members highly corrective to previous predictions. Part of this could be due to the low signal-to-noise ratio in financial data.

The loss surface of the gls-gbm trained on the feature set including option features is least fragmented. While the validation accuracy of the best combinations improves significantly to sunitx-percent, worst trials even under-perform these of smaller feature sets. Based on this finding we conjecture, that more data does not *per-se* improve the model and that models require a thoughtful tuning procedure. By this means, our conclusion contradict the one of ([[@ronenMachineLearningTrade2022]]14), who find no advantage from tuning their tree-based ensemble.

**Gradient Boosting + Self-Training**
The results for the gls-gbm in combination with self-training are similar and visualised in cref-a) c). To conserve space, we summarise the important findings.
(...)
- where does depth come from?
- Why just two iterations

**Transformer with Pre-Training**
To conserve space we only report the . Overall, the.  A visualisation of the search space for semi-supervised methods can be found  

**Classical rules**
Akin to selecting the machine learning classifiers, we determine our classical baselines on the gls-ise validation set. This prevents overfitting the test set and maintains consistency between both paradigms. For the same reason, baselines are kept constant in the transfer setting on the gls-cboe sample. Entirely for reference, we also report accuracies of the tick rule, quote rule, and gls-lr algorithm, due to their widespread adoption in literature.

<mark style="background: #CACFD9A6;">(by drawing on the stacking principle)</mark>
While optimising the combination of trade classification rules through Bayesian search is theoretically feasible, we found no out-performance over hybrid rules reported in literature \footnote{We performed a Bayesian search with 50 trials for trade classification rules, stacking up to five rules. Experiment available under: \url{https://wandb.ai/fbv/thesis}}.  Thus, \cref{tab:ise-classical-hyperparam-classical-size} reports the accuracies of common trade classification rules on the \gls{ISE} validation set.

![[table-classical-rules.png]]
(two more columns for Grauer combination)

Do like $\operatorname{Categorical}\left[\operatorname{tick}_{\text{ex}},\ldots,\operatorname{Id}\right]$ or even simpler $\left[\ldots\right]$ done so in Grinzsjastin (p. 23)  



![[training-validation-accuracy.png]]

![[training-vs-validation-accuracy.png]]

![[Pasted image 20230617144732.png]]
![[Pasted image 20230617144757.png]]

![[Pasted image 20230617200347.png]]

![[Pasted image 20230617200407.png]]

https://arxiv.org/pdf/1603.02754.pdf

https://albertum.medium.com/l1-l2-regularisation-in-xgboost-regression-7b2db08a59e0

random_strength  
This parameter helps to overcome overfitting of the model.

When selecting a new split, each possible split gets a score (for example, by how much does adding this split improve the loss function on train). After that all scores are sorted and a split with the highest score is selected.  
The scores are not random. This parameter adds a normally distributed random variable to the score of the feature. It has zero mean and variance that is larger in the start of the training and decreases during the training. random_strength is the multiplier of the variance.

Using randomness during split selection procedure helps to overcome overfitting and increase the resulting quality of the model.

bagging_temperature  
This parameter is responsible for Bayesian bootstrap.  
Bayesian bootstrap is used by default in classification and regression modes. In ranking we use Bernoulli bootstrap by default.

In bayesian bootstrap each object is assigned random weight. If bagging temperature is equal to 1 then weights are sampled from exponential distribution. If bagging temperature is equal to 0 then all weights are equal to 1. By changing this parameter from 0 to +infty you can controll intensity of the bootstrap.



Coefficient at the L2 regularisation term of the cost function. Any positive value is allowed.

We use Epsilon dataset and we measure mean tree construction time one can achieve without using feature subsampling and/or bagging by CatBoost (both Ordered and Plain modes), XGBoost (we use histogram-based version, which is faster) and LightGBM.

Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.
Use the Bayesian bootstrap to assign random weights to objects. The weights are sampled from exponential distribution if the value of this parameter is set to 1. All weights are equal to 1 if the value of this parameter is set to 0. Possible values are in the range $[0;\inf⁡)$The higher the value the more aggressive the bagging is.

The amount of randomness to use for scoring splits when the tree structure is selected. Use this parameter to avoid overfitting the model.
The value of this parameter is used when selecting splits. On every iteration each possible split gets a score (for example, the score indicates how much adding this split will improve the loss function for the training dataset). The split with the highest score is selected. The scores have no randomness. A normally distributed random variable is added to the score of the feature. It has a zero mean and a variance that decreases during the training. The value of this parameter is the multiplier of the variance.

**Notes:**
[[💡Hyperparameter tuning notes]]