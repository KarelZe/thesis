#lr-warmup #lr-scheduling 

**Ressources**
- Do less Alchemy at NIPS: https://www.youtube.com/watch?v=Qi1Yry33TQE
- Practical guide for researchers by Google: https://github.com/google-research/tuning_playbook


## Task to be done⛑️
- create an action plan for improving module results
- restructure all notes and info, that I've gathered so far and incorporate into this plan




- Find optimal batch size
- add learning rate scheduler
- decide on a optimizer
- implement a much simpler approach e. g., logistic regression
- complete evaluation pipeline
- advance experiment tracking https://www.learnpytorch.io/07_pytorch_experiment_tracking/

## Chapter structure
- Write about the general idea of training / tuning.
- Differentiate into exploration and exploitation. Do this in a structured way.
- Set up a simple baseline
- Write why we use a quasi-random search during exploration and Bayesian search during exploitation phase. (see https://github.com/google-research/tuning_playbook)
- Isolate the optimization of different hyperparameters e. g., optimizer, activation functions etc. during the exploration phase. Take into account nuance parameters. Decompose into smaller problems, that are manageable.
- Keep ideas simple and gradually add complexity and make it visible in the structure of the chapter. Helps with reasoning later. Possible steps could be:
	1. 

### Classical algorithms
- Implement as sklearn classifier. Explain why.

## Baseline 
- Start with something simple e. g., Logistic Regression or Gradient Boosted Trees, due to being well suited for tabular data. Also  [[@grauerOptionTradeClassification2022]] could be a baseline.

### Gradient Boosting
- Discuss overfitting and underfitting. What measures are taken to address the problem?

### Neural Nets
- Motivate the importance of regularized neural nets with [[@kadraWelltunedSimpleNets2021]] papers. Authors state, that the improvements from regularization of neural nets are very pronounced and highly significant. Discuss which regularization approaches are applied and why.  
- Similarly, [[@heBagTricksImage2018]] show how they can improve the performance of neural nets for computer vision through "tricks" like learning rate scheduling.
- Also see [[@shavittRegularizationLearningNetworks2018]] for regularization in neural networks for tabular data.
- Motivate different activation functions with [[@shazeerGLUVariantsImprove2020]]

### Additional techniques 🍰
- Try out adverserial weight perturbation as done [here.][feedback-nn-train | Kaggle](https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook)
- Try out Stochastic weight averaging for neural net as done [here.](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx) or here [Stochastic Weight Averaging in PyTorch](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)





## Learning rate
- try cyclic learning rates https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html
- cycling procedure was proposed in [[@loshchilovSGDRStochasticGradient2017]] and [[@smithCyclicalLearningRates2017]]



## Batch size
- The batch size should _not be_ treated as a tunable hyperparameter for validation set performance. (from https://github.com/google-research/tuning_playbook)
-   As long as all hyperparameters are well-tuned (especially the learning rate and regularization hyperparameters) and the number of training steps is sufficient, the same final performance should be attainable using any batch size (see[[@shallueMeasuringEffectsData]]. (from https://github.com/google-research/tuning_playbook)
- Smaller batch sizes introduce more noise into the training algorithm due to sample variance, and this noise can have a regularizing effect. Thus, larger batch sizes can be more prone to overfitting and may require stronger regularization and/or additional regularization techniques. (from https://github.com/google-research/tuning_playbook)
- Often, the largest batch size supported by the available hardware will be smaller than the critical batch size. Therefore, a good rule of thumb (without running any experiments) is to use the largest batch size possible.
- -   The hyperparameters that interact most strongly with the batch size, and therefore are most important to tune separately for each batch size, are the optimizer hyperparameters (e.g. learning rate, momentum) and the regularization hyperparameters.

## Incremental tuning strategy
-   The most effective way to maximize performance is to start with a simple configuration and incrementally add features and make improvements while building up insight into the problem.
    -   We use automated search algorithms in each round of tuning and continually update our search spaces as our understanding grows.
-   As we explore, we will naturally find better and better configurations and therefore our "best" model will continually improve.

At a high level, our incremental tuning strategy involves repeating the following four steps:

1.  Identify an appropriately-scoped goal for the next round of experiments.
2.  Design and run a set of experiments that makes progress towards this goal.
3.  Learn what we can from the results.
4.  Consider whether to launch the new best configuration. (from https://github.com/google-research/tuning_playbook)

-   Each round of experiments should have a clear goal and be sufficiently narrow in scope that the experiments can actually make progress towards the goal: if we try to add multiple features or answer multiple questions at once, we may not be able to disentangle the separate effects on the results.
-   Example goals include:
    -   Try a potential improvement to the pipeline (e.g. a new regularizer, preprocessing choice, etc.).
    -   Understand the impact of a particular model hyperparameter (e.g. the activation function)
    -   Greedily maximize validation error.

- Identify which hyperparameters are <mark style="background: #FFF3A3A6;">scientific, nuisance, and fixed hyperparameters</mark> for the experimental goal. Create a sequence of studies to compare different values of the scientific hyperparameters while optimizing over the nuisance hyperparameters. Choose the search space of nuisance hyperparameters to balance resource costs with scientific value.
- Scientific hyperparameters are those whose effect on the model's performance we're trying to measure.
- Nuisance hyperparameters are those that need to be optimized over in order to fairly compare different values of the scientific hyperparameters. This is similar to the statistical concept of [nuisance parameters](https://en.wikipedia.org/wiki/Nuisance_parameter).
- Fixed hyperparameters will have their values fixed in the current round of experiments. These are hyperparameters whose values do not need to (or we do not want them to) change when comparing different values of the scientific hyperparameters.
- By fixing certain hyperparameters for a set of experiments, we must accept that conclusions derived from the experiments might not be valid for other settings of the fixed hyperparameters. In other words, fixed hyperparameters create caveats for any conclusions we draw from the experiments.
- The purpose of the studies is to run the pipeline with different values of the scientific hyperparameters, while at the same time **"optimizing away"** (or "optimizing over") the nuisance hyperparameters so that comparisons between different values of the scientific hyperparameters are as fair as possible.
-   For example, if our goal is to select the best optimizer out of Nesterov momentum and Adam, we could create one study in which `optimizer="Nesterov_momentum"` and the nuisance hyperparameters are `{learning_rate, momentum}`, and another study in which `optimizer="Adam"` and the nuisance hyperparameters are `{learning_rate, beta1, beta2, epsilon}`. We would compare the two optimizers by selecting the best performing trial from each study.
- In the simplest case, we would make a separate study for each configuration of the scientific parameters, where each study tunes over the nuisance hyperparameters.
- We can use any gradient-free optimization algorithm, including methods such as Bayesian optimization or evolutionary algorithms, to optimize over the nuisance hyperparameters, although [we prefer](https://github.com/google-research/tuning_playbook#why-use-quasi-random-search-instead-of-more-sophisticated-black-box-optimization-algorithms-during-the-exploration-phase-of-tuning) to use quasi-random search in the [exploration phase](https://github.com/google-research/tuning_playbook#exploration-vs-exploitation) of tuning because of a variety of advantages it has in this setting. [After exploration concludes](https://github.com/google-research/tuning_playbook#after-exploration-concludes), if state-of-the-art Bayesian optimization software is available, that is our preferred choice.
- Extracting insights from experimental results: Ultimately, each group of experiments has a specific goal and we want to evaluate the evidence the experiments provide toward that goal.
-   However, if we ask the right questions, we will often find issues that need to be corrected before a given set of experiments can make much progress towards their original goal. If we don’t ask these questions, we may draw incorrect conclusions.
- Since running experiments can be expensive, we also want to take the opportunity to extract other useful insights from each group of experiments, even if these insights are not immediately relevant to the current goal
- Before analyzing a given set of experiments to make progress toward their original goal, we should ask ourselves the following additional questions:
	- Is the search space large enough? e. g., close to the boundary of search space?
	- Did we sample enough points?
	- Does the model exhibit optimization issues?
	- What can we learn from the training curves for the best trials?
## Examine the training curves
-   Although in many cases the primary objective of our experiments only requires considering the validation error of each trial, we must be careful when reducing each trial to a single number because it can hide important details about what’s going on below the surface.
-   For every study, we always look at the **training curves** (training error and validation error plotted versus training step over the duration of training) of at least the best few trials.
-   Even if this is not necessary for addressing the primary experimental objective, examining the training curves is an easy way to identify common failure modes and can help us prioritize what actions to take next.
- If any of the best trials exhibits problematic overfitting, we usually want to re-run the experiment with additional regularization techniques and/or better tune the existing regularization parameters before comparing the values of the scientific hyperparameters.
- Is there high step-to-step variance in the training or validation error late in training?
- Are the trials still improving at the end of training?
- Has performance on the training and validation sets saturated long before the final training step?
-   **Training procedure variance**, **retrain variance**, or **trial variance**: the variation we see between training runs that use the same hyperparameters, but different random seeds.
## After exploration concludes
-  At some point, our priorities will shift from learning more about the tuning problem to producing a single best configuration to launch or otherwise use.
-  Our exploration work should have revealed the most essential hyperparameters to tune (as well as sensible ranges for them) that we can use to construct a search space for a final automated tuning study using as large a tuning budget as possible.
- Since we no longer care about maximizing our insight into the tuning problem, many of [the advantages of quasi-random search](https://github.com/google-research/tuning_playbook#why-use-quasi-random-search-instead-of-more-sophisticated-black-box-optimization-algorithms-during-the-exploration-phase-of-tuning) no longer apply and Bayesian optimization tools should be used to automatically find the best hyperparameter configuration.
-   At this point, we should also consider checking the performance on the test set.
- <mark style="background: #FFF3A3A6;">In principle, we could even fold the validation set into the training set and retraining the best configuration found with Bayesian optimization.</mark> However, this is only appropriate if there won't be future launches with this specific workload (e.g. a one-time Kaggle competition).

## Deciding how long to train when training is compute bound
-   As a starting point, we recommend two rounds of tuning:
    -   Round 1: Shorter runs to find good model and optimizer hyperparameters.
    -   Round 2: Very few long runs on good hyperparameter points to get the final model.
-   The biggest question going from `Round i` → `Round i+1` is how to adjust learning rate decay schedules.
    -   One common pitfall when adjusting learning rate schedules between rounds is using all the extra training steps with too small of a learning rate.
- In round 1 (very likely to transfer): warmup length, intialization, (likely to transfer) model architecture, (might transfer) optimizer, data augmentation, regularization (unlikely to transfer) learning rate schedule
- E.g. if linear schedule then keep the length of the decay fixed from Round 1 and extend the period of constant lr in the beginning.
- For cosine decay, just keep the base lr from Round 1 and extend `max_train_steps` as in [Chinchilla paper](https://arxiv.org/abs/2203.15556).
- Supporting prospective early stopping is usually not necessary, since we’re pre-specifying a trial budget and are preserving the N best checkpoints seen so far.

## Experiment tracking
-   We've found that keeping track of experiment results in a spreadsheet has been helpful for the sorts of modeling problems we've worked on. It often has the following columns:
    -   Study name
    -   A link to wherever the config for the study is stored.
    -   Notes or a short description of the study.
    -   Number of trials run
    -   Performance on the validation set of the best checkpoint in the study.
    -   Specific reproduction commands or notes on what unsubmitted changes were necessary to launch training.
-   Find a tracking system that captures at least the information listed above and is convenient for the people doing it. 

## Best learning rate decay schedule family
-   Although we don't know the best schedule family, we're confident that it’s important to have some (non-constant) schedule and that tuning it matters.
-   Different learning rates work best at different times during the optimization process. Having some sort of schedule makes it more likely for the model to hit a good learning rate.
- Our preference is either linear decay or cosine decay, and a bunch of other schedule families are probably good too.

## How should Adam be tuned?
- As discussed above, making general statements about search spaces and how many points one should sample from the search space is very difficult. Note that not all the hyperparameters in Adam are equally important. The following rules of thumb correspond to different "budgets" for the number of trials in a study.
- If $<10$ trials in a study, only tune the (base) learning rate.
- If 10-25 trials, tune learning rate and $\beta_1$.
- If $25+$ trials, tune the learning rate, $\beta_1$ and $\epsilon$.
- If one can run substantially more than 25 trials, additionally tune $\beta_2$.

## Quasi random search in exploration and Baysian search in Random search
-   Quasi-random search (based on [low-discrepancy sequences](https://en.wikipedia.org/wiki/Low-discrepancy_sequence)) is our preference over fancier black box optimization tools when used as part of an iterative tuning process intended to maximize insight into the tuning problem (what we refer to as the "exploration phase"). Bayesian optimization and similar tools are more appropriate for the exploitation phase.
-   Quasi-random search based on randomly shifted low-discrepancy sequences can be thought of as "jittered, shuffled grid search", since it uniformly, but randomly, explores a given search space and spreads out the search points more than random search.
-   The advantages of quasi-random search over more sophisticated black box optimization tools (e.g. Bayesian optimization, evolutionary algorithms) include:
    1.  Sampling the search space non-adaptively makes it possible to change the tuning objective in post hoc analysis without rerunning experiments.
        -   For example, we usually want to find the best trial in terms of validation error achieved at any point in training. But the non-adaptive nature of quasi-random search makes it possible to find the best trial based on final validation error, training error, or some alternative evaluation metric without rerunning any experiments.
    2.  Quasi-random search behaves in a consistent and statistically reproducible way.
        -   It should be possible to reproduce a study from six months ago even if the implementation of the search algorithm changes, as long as it maintains the same uniformity properties. If using sophisticated Bayesian optimization software, the implementation might change in an important way between versions, making it much harder to reproduce an old search. It isn’t always possible to roll back to an old implementation (e.g. if the optimization tool is run as a service).
    3.  Its uniform exploration of the search space makes it easier to reason about the results and what they might suggest about the search space.
        -   For example, if the best point in the traversal of quasi-random search is at the boundary of the search space, this is a good (but not foolproof) signal that the search space bounds should be changed. [This section](https://github.com/google-research/tuning_playbook#identifying-bad-search-space-boundaries) goes into more depth. However, an adaptive black box optimization algorithm might have neglected the middle of the search space because of some unlucky early trials even if it happens to contain equally good points, since it is this exact sort of non-uniformity that a good optimization algorithm needs to employ to speed up the search.
    4.  Running different numbers of trials in parallel versus sequentially will not produce statistically different results when using quasi-random search (or other non-adaptive search algorithms), unlike with adaptive algorithms.
    5.  More sophisticated search algorithms may not always handle infeasible points correctly, especially if they aren't designed with neural network hyperparameter tuning in mind.
    6.  Quasi-random search is simple and works especially well when many tuning trials will be running in parallel.
        -   Anecdotally[1](https://github.com/google-research/tuning_playbook#user-content-fn-3-9853a2357fd18b27e5cfd5bb3779f74d), it is very hard for an adaptive algorithm to beat a quasi-random search that has 2X its budget, especially when many trials need to be run in parallel (and thus there are very few chances to make use of previous trial results when launching new trials).
        -   Without expertise in Bayesian optimization and other advanced black box optimization methods, we might not achieve the benefits they are, in principle, capable of providing. It is hard to benchmark advanced black box optimization algorithms in realistic deep learning tuning conditions. They are a very active area of current research, and the more sophisticated algorithms come with their own pitfalls for inexperienced users. Experts in these methods are able to get good results, but in high-parallelism conditions the search space and budget tend to matter a lot more.
-   That said, if our computational resources only allow a small number of trials to run in parallel and we can afford to run many trials in sequence, Bayesian optimization becomes much more attractive despite making our tuning results harder to interpret.

## Potential fixes for instability patterns 
-   Apply learning rate warmup
    -   Best for early training instability.
-   Apply gradient clipping
    -   Good for both early and mid training instability, may fix some bad inits that warmup cannot.
-   Try a new optimizer
    -   Sometimes Adam can handle instabilities that Momentum can’t. This is an active area of research.
-   We can ensure that we’re using best practices/initializations for our model architecture (examples below).
    -   Add residual connections and normalization if the model doesn't contain it already.
-   Normalization should be the last operation before the residual. E.g. x + Norm(f(x)).
-   Norm(x + f(x)) known to cause issues.
-   Try initializing residual branches to 0 (e.g. [ReZero init](https://arxiv.org/abs/2003.04887)).
-   Lower the learning rate
    -   This is a last resort.





- practical tips for deep learning: http://yyue.blogspot.com/2015/01/abrief-overview-of-deep-learning.html
- post norm / pre-norm / lr warm up [[@xiongLayerNormalizationTransformer2020]]

- training of the transformer has been found non-trivial[[@liuUnderstandingDifficultyTraining2020]]
- introduce notion of effective batch size (batch size when training is split across multiple gpus; see [[🧠Deep Learning Methods/Transformer/@popelTrainingTipsTransformer2018]] p. 46)
- report or store training times?
- In case of diverged training, try gradient clipping and/or more warmup steps. (found in [[🧠Deep Learning Methods/Transformer/@popelTrainingTipsTransformer2018]])
- use a higher learning rate e. g. lr=0.2
- results are the same when trained on multiple gpus, if batch size across all gpus remains the same. [[@poppeSensitivityVPINChoice2016]] confirmed this empirically.
- One might has to adjust the lr when scaling across multiple gpus [[@poppeSensitivityVPINChoice2016]] contains a nice discussion.
- Use weight decay of 0.1 for a small amount of regularization [[@loshchilovDecoupledWeightDecay2019]].
- on the compute cost of transformers [[@ivanovDataMovementAll2021]]

- On activation function see [[@shazeerGLUVariantsImprove2020]]


- On activation function see [[@shazeerGLUVariantsImprove2020]]

-  https://wandb.ai/site/articles/debugging-neural-networks-with-pytorch-and-w-b-using-gradients-and-visualizations

- log gradients and loss using `wandb.watch` as shown here https://www.youtube.com/watch?v=k6p-gqxJfP4 with `wandb.log({"epoch":epoch, "loss":loss}, step)` (nested in `if ((batch_ct +1) % 25) == 0:`) and `wandb.watch(model, criterion, log="all", log_freq=10)`
- watch out for exploding and vanishing gradients
- distillation, learning rate warmup, learning rate decay (not used but could improve training times and maybe accuracy) ([[@gorishniyRevisitingDeepLearning2021]])
- Use of Post-Norm (Hello [[🤖TabTransformer]]) has been deemed outdated in Transformers due to a more fragile training process (see [[@gorishniyRevisitingDeepLearning2021]]). May swap (?).
- Tips for training deep neural networks on categorical data: https://www.youtube.com/watch?v=E8C_obO1HfY 
- Mind the double descent effect https://openai.com/blog/deep-double-descent/
- https://blog.ml6.eu/how-a-pretrained-tabtransformer-performs-in-the-real-world-eccb12362950
- https://www.borealisai.com/research-blogs/tutorial-14-transformers-i-introduction/
- Might use additional tips from here: [[@liuRoBERTaRobustlyOptimized2019]])
- https://paperswithcode.com/paper/revisiting-deep-learning-models-for-tabular/review/
- [[@liuUnderstandingDifficultyTraining2020]]

## Tipps from reddit 🤖
https://www.reddit.com/r/MachineLearning/comments/z088fo/r_tips_on_training_transformers/
1.  Bigger architectures learn better and train faster
2.  Layer norms are very important
3.  Apply high learning rates to top layers and smaller rates to lower layers
4.  The batch size should be as high as possible -> write script -> keep gpu busy
5. Transformers are data hungry (must be stated in the [[@vaswaniAttentionAllYou2017]] paper)

## Notes from Borealis AI
- detailed tips: https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/

## Notes from Neptune AI
(https://neptune.ai/blog/tips-to-train-nlp-models)
- use layer-wise learning rate
- reinitialize layers
- use pre-training :-)
- code provided here: https://github.com/flowerpot-ai/stabilizer

## Notes from less wrong on training Transformers
(https://www.lesswrong.com/posts/b3CQrAo2nufqzwNHF/how-to-train-your-transformer)
-   Read the relevant literature and take note of all tricks
-   Use Colab for free GPU time.
-   Rezero or Dynamic Linear Combinations for scaling depth.
-   Shuffle data and create train, validation, test sets from the beginning.
-   Shuffle in memory if samples are otherwise correlated.
-   Train a simple fully connected network as baseline and sanity check.
-   Establish scaling laws to find bottlenecks and aim at the ideal model size.
-   Use gradient accumulation to fit larger models on a small GPU.
-   Lower the learning rate when the model stagnates, but don't start too low. (Better yet, use a cyclic learning rate schedule.)


## Notes from Huang et al paper
See [[@huangTabTransformerTabularData2020]] (p. 12)
- AdamW optimizer
- const learning rate
- early stopping after 15 epochs
- hidden (embedding) dim, no of layers, no attention heads. MLP sizes are 4x and 2x the input
- 2,4,8 attention heads
- hid dim 32, 64, 128, 256
- no layers 1, 2, 3, 6, 12
- mlp selu, batchnorm, 2 hidden layers, capacity 8* l with l = d/8, second layer m*l with m in (1,3)

## Notes from uvadlc
(https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)

- One commonly used technique for training a Transformer is learning rate warm-up. This means that we gradually increase the learning rate from 0 on to our originally specified learning rate in the first few iterations. Thus, we slowly start learning instead of taking very large steps from the beginning. In fact, training a deep Transformer without learning rate warm-up can make the model diverge and achieve a much worse performance on training and testing.
- Clearly, the warm-up is a crucial hyperparameter in the Transformer architecture. Why is it so important? There are currently two common explanations.
- For instance, the original Transformer paper used an exponential decay scheduler with a warm-up. However, the currently most popular scheduler is the cosine warm-up scheduler, which combines warm-up with a cosine-shaped learning rate decay. We can implement it below, and visualize the learning rate factor over epochs.
- Some intuition on learning rate warm-up can be found in [[@liuVarianceAdaptiveLearning2021]] and https://stackoverflow.com/a/55942518/5755604




For training of transformers see [[@popelTrainingTipsTransformer2018]]
Intuition behind sample weights: https://m.youtube.com/watch?v=68ABAU_V8qI
Smart batching for transformers: https://mccormickml.com/2020/07/29/smart-batching-tutorial/#why-we-pad
For training the transformer see: https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it


Ilya Sutskever. A Brief Overview of Deep Learning. http://yyue.blogspot.com/2015/01/abrief-overview-of-deep-learning.html, January 2015.

Research broader theory behind sample weighting. Research if there is a broader theory / concept to decay e. g. exponential smooting or weighted regression etc.
For random shuffling with stochastic gradient descent see [[@lecunEfficientBackProp2012]]

- Use weighting scheme for samples. See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- tips for training transformers https://huggingface.co/docs/transformers/v4.18.0/en/performance
- learning rate warmup https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
- optimizer schedulers for transformers: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules


![[visualization_of_bleu_over_time.png]]

![[bleu_no_of_gpus.png]]

Visualize model parameters:
![[viz-model-params.png]]
(from https://arxiv.org/pdf/2005.14165.pdf)

