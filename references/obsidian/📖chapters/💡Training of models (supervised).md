Our search space is reported in Table-X, which we laid out based on the recommendations in ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]]20) and ([[@gorishniyRevisitingDeepLearning2021]]18) and ([[@rubachevRevisitingPretrainingObjectives2022]]., 2022, p. 4) with minor deviations. For gradient-boosting we raise the border count to $256$, which increases the number of split candidates per feature through a finer quantization, Expectedly, accuracy increases at the cost of computational efficiency. The size of the ensemble $M$ may not be fully exhausted. Acknowleding the observations ([[@friedmanGreedyFunctionApproximation2001]]14), that the learning rate $\lambda$ the learning rate and the size of the ensemble have a strong interdependence, we only tune the learning rate and stop adding new trees to the ensemble, once the validation accuracy decreases for consecutive (...) steps.

We grow symmetric trees, which acts as a regularizer.

The hyperparameter search for the FT-Transformer is identical to ([[@gorishniyRevisitingDeepLearning2021]]18) variant (b). From preliminary tests, we observed that the use of a learning rate schedule with a short learning rate warm-up phase both stabilizes training and improves accuracy (cp. cref-training-of-supervised). Their constant learning rate and our decayed learning rate may thus not be entirely comparable. Additionally, we employ early stopping and halt training after 15 consecutive decreases in validation accuracy, affecting the effective number of epochs. Both techniques have not been used by the orginal author's to provide a conservative baseline ([[@gorishniyRevisitingDeepLearning2021]]5), for the sake of a fair comparison in our context both techniques should be used.

**Failed attempts:**
We experimented with label smoothing, 

Visualize the decision of 


Visualize effect 

![[Pasted image 20230428110412.png]]


Classical trade signing algorithms, such as the tick test, are also impacted by missing values. In theses cases, we defer to a random classification or a subsequent rule, if rules can not be computed. Details are provided in section [[💡Training of models (supervised)]].


## Transformer
[[🤖Training of the Transformer]]


Look into grooking: https://arxiv.org/pdf/2201.02177.pdf
![[grocking.png]]

- What optimizer is chosen? Why? Could try out Adam or Adan?

[[@somepalliSAINTImprovedNeural2021]] use logistic regression. I really like the fact they also compare a simple logistic regression to these models, because if you’re not able to perform notably better relative to the simplest model one could do, then why would we care? The fact that logistic regression is at times competitive and even beats boosting/SAINT methods occasionally gives me pause though. Perhaps some of these data are not sufficiently complex to be useful in distinguishing these methods? It is realistic though. While it’s best not to assume as such, sometimes a linear model is appropriate given the features and target at hand.


Many practical implementations of boosting like XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017), and CatBoost (Prokhorenkova et al., 2018) use constant learning rate in their default settings as in practice it outperforms dynamically decreasing ones. However, existing works on the convergence of boosting algorithms assume decreasing learning rates (Zhang & Yu, 2005; Zhou & Hooker, 2018), thus leaving an open question: if we assume constant learning rate  > 0, can convergence be guaranteed? https://arxiv.org/pdf/2001.07248.pdf

## Categoricals
- The problem of high number of categories is called a high cardinality problem of categoricals see e. g., [[@huangTabTransformerTabularData2020]]
- To inform our models which features are categorical, we pass the index the index of categorical features and the their cardinality to the models.
- Discuss cardinality of categoricals.
- strict assumption as we have out-of-vocabulary tokens e. g., unseen symbols like "TSLA".  (see done differently here https://keras.io/examples/structured_data/tabtransformer/)
- Idea: Instead of assign an unknown token it could help assign to map the token to random vector. https://stackoverflow.com/questions/45495190/initializing-out-of-vocabulary-oov-tokens
- Idea: reduce the least frequent root symbols.
- Apply an idea similar to sentence piece. Here, the number of words in vocabulary is fixed https://github.com/google/sentencepiece. See repo for paper / algorithm.
- For explosion in parameters also see [[@tunstallNaturalLanguageProcessing2022]]. Could apply their reasoning (calculate no. of parameters) for my work. 
- KISS. Dimensionality is probably not so high, that it can not be handled. It's much smaller than common corpi sizes. Mapping to 'UKNWN' character. -> Think how this can be done using the current `sklearn` implementation.
- **Solutions:** 
	- Use a linear projection: https://www.kaggle.com/code/limerobot/dsb2019-v77-tr-dt-aug0-5-3tta/notebook
	- https://en.wikipedia.org/wiki/Additive_smoothing



Studies adressing high cardinality
<mark style="background: #ADCCFFA6;">“Study how both tree-based models and neural networks cope with specific challenges such as missing data or high-cardinality categorical features, thus extending to neural networks prior empirical work [Cerda et al., 2018, Cerda and Varoquaux, 2020, Perez-Lebel et al., 2022].” ([Grinsztajn et al., 2022, p. 9](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=9&annotation=PCA3SDUE))</mark>




## Resources
- Do less Alchemy at NIPS: https://www.youtube.com/watch?v=Qi1Yry33TQE
- Practical guide for researchers by Google: https://github.com/google-research/tuning_playbook or [[@tuningplaybookgithub]]


Following the example of ([[@rubachevRevisitingPretrainingObjectives2022]]), we share the 

We aim to be transparent about the training setup





## Task to be done⛑️
1.  create an action plan for improving module results
2. restructure all notes and info, that I've gathered so far and incorporate into this plan
3. Find optimal batch size, remove batch size code from objective
4. Implement a much simpler approach e. g., logistic regression
5. Allow to keep certain hyperparameters fixed
6. add learning rate scheduler
7. decide on a optimizer
8.  complete evaluation pipeline

## Chapter structure
- Write about the general idea of training / tuning.
- Differentiate into exploration and exploitation. Do this in a structured way.
- Set up a simple baseline
- Write why we use a quasi-random search during exploration and Bayesian search during exploitation phase. (see https://github.com/google-research/tuning_playbook)
- Isolate the optimization of different hyperparameters e. g., optimizer, activation functions etc. during the exploration phase. Take into account nuance parameters. Decompose into smaller problems, that are manageable.
- Discuss why retraining of the best model could make sense.
- Keep ideas simple and gradually add complexity and make it visible in the structure of the chapter. Helps with reasoning later. Possible steps could be:
	- tbd
	- tbd

## Batch size
- Estimate the maximum batch size early on, as many parameters depend on it. (see https://github.com/google-research/tuning_playbook)
- Base implementation on https://github.com/BlackHC/toma or this blog post https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1 (nice idea with the dummy data.)
- Introduce notion of effective batch size (batch size when training is split across multiple gpus; see [[@popelTrainingTipsTransformer2018]] p. 46)
- Reason why we aim for compleete batches. See e. g., https://mccormickml.com/2020/07/29/smart-batching-tutorial/#why-we-pad
- results are the same when trained on multiple gpus, if batch size across all gpus remains the same. [[@poppeSensitivityVPINChoice2016]] confirmed this empirically.

## Advanced logging 🗞️
- advance experiment tracking https://www.learnpytorch.io/07_pytorch_experiment_tracking/
- log gradients and loss using `wandb.watch` as shown here https://www.youtube.com/watch?v=k6p-gqxJfP4 with `wandb.log({"epoch":epoch, "loss":loss}, step)` (nested in `if ((batch_ct +1) % 25) == 0:`) and `wandb.watch(model, criterion, log="all", log_freq=10)`
- In-depth weights and bias blog post: https://wandb.ai/site/articles/debugging-neural-networks-with-pytorch-and-w-b-using-gradients-and-visualizations
- Mind the double descent effect https://openai.com/blog/deep-double-descent/

## Learning rate
- Lower the learning rate when the model stagnates, but don't start too low.  Try cyclic learning rates https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html
- cycling procedure was proposed in [[@loshchilovSGDRStochasticGradient2017]] and [[@smithCyclicalLearningRates2017]]
- for intuitive explanation on learning rate warm up see https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
- One might has to adjust the lr when scaling across multiple gpus [[@poppeSensitivityVPINChoice2016]] contains a nice discussion.
- Use learning rate warmup for post-LN transformers and maybe also for other
- Some intuition on learning rate warm-up can be found in [[@liuVarianceAdaptiveLearning2021]] and https://stackoverflow.com/a/55942518/5755604

## Random Shuffle
- For random shuffling with stochastic gradient descent see [[@lecunEfficientBackProp2012 1]]
-   Shuffle in memory if samples are otherwise correlated. (see https://www.lesswrong.com/posts/b3CQrAo2nufqzwNHF/how-to-train-your-transformer)
-   Shuffle in memory if samples are otherwise correlated. (see https://www.lesswrong.com/posts/b3CQrAo2nufqzwNHF/how-to-train-your-transformer)

## Classical algorithms
- Implement as sklearn classifier for easier evaluation, ease of use, and re-usability.

## Baseline 
- Start with something simple e. g., Logistic Regression or Gradient Boosted Trees, due to being well suited for tabular data. Also  [[@grauerOptionTradeClassification2022]] could be a baseline.
-   Train a simple fully connected network as baseline and sanity check. We could argue that TabTransformer callapses to one.

### Gradient Boosting
- Discuss overfitting and underfitting. What measures are taken to address the problem?
- Use early stopping
- Use sample weighting

### Transformer
- See tips in [[@tuningplaybookgithub]]
- Transformers are much more elaborate to train than gradient boosting approaches. Training of the transformer has been found non-trivial [[@liuUnderstandingDifficultyTraining2020]]
- Motivate the importance of regularized neural nets with [[@kadraWelltunedSimpleNets2021]] papers. Authors state, that the improvements from regularization of neural nets are very pronounced and highly significant. Discuss which regularization approaches are applied and why.  
- Similarly, [[@heBagTricksImage2018]] show how they can improve the performance of neural nets for computer vision through "tricks" like learning rate scheduling.
- Also see [[@shavittRegularizationLearningNetworks2018]] for regularization in neural networks for tabular data.
- Motivate different activation functions with [[@shazeerGLUVariantsImprove2020]]
- Search space is adapated from [[@huangTabTransformerTabularData2020]] and [[@gorishniyRevisitingDeepLearning2021]]. We make sure optimal solution is not on the borders of the search space.
- post norm / pre-norm / lr warm up [[@xiongLayerNormalizationTransformer2020]].  Use of Post-Norm (Hello [[🤖TabTransformer]]) has been deemed outdated in Transformers due to a more fragile training process (see [[@gorishniyRevisitingDeepLearning2021]]). 
- In case of diverged training, try gradient clipping and/or more warmup steps. (found in [[@popelTrainingTipsTransformer2018]])
- Use weight decay of 0.1 for a small amount of regularization [[@loshchilovDecoupledWeightDecay2019]].
- on the compute cost of transformers [[@ivanovDataMovementAll2021]]
- training tips for Transformer https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/
- Might use additional tips from here: ([[@liuRoBERTaRobustlyOptimized2019]] and [[@liuUnderstandingDifficultyTraining2020]])
- One commonly used technique for training a Transformer is learning rate warm-up. This means that we gradually increase the learning rate from 0 on to our originally specified learning rate in the first few iterations. Thus, we slowly start learning instead of taking very large steps from the beginning. In fact, training a deep Transformer without learning rate warm-up can make the model diverge and achieve a much worse performance on training and testing (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html).
- For training the transformer see: https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it

### Additional techniques 🍰
- Try out adverserial weight perturbation as done [here.][feedback-nn-train | Kaggle](https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook)
- Try out Stochastic weight averaging for neural net as done [here.](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx) or here [Stochastic Weight Averaging in PyTorch](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)
- Use weighting scheme for samples. See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html


## Visualizations 🖼️

![[visualization_of_bleu_over_time.png]]

![[bleu_no_of_gpus.png]]

Visualize model parameters:
![[viz-model-params.png]]
(from https://arxiv.org/pdf/2005.14165.pdf)













