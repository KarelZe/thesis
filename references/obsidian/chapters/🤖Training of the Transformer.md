#lr-warmup #lr-scheduling 

## Resources
- Do less Alchemy at NIPS: https://www.youtube.com/watch?v=Qi1Yry33TQE
- Practical guide for researchers by Google: https://github.com/google-research/tuning_playbook or [[@tuningplaybookgithub]]


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
- Introduce notion of effective batch size (batch size when training is split across multiple gpus; see [[🧠Deep Learning Methods/Transformer/@popelTrainingTipsTransformer2018]] p. 46)
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
- For random shuffling with stochastic gradient descent see [[@lecunEfficientBackProp2012]]
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
- In case of diverged training, try gradient clipping and/or more warmup steps. (found in [[🧠Deep Learning Methods/Transformer/@popelTrainingTipsTransformer2018]])
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





