This chapter documents the basic training setup, and defines a baseline before applying hyperparameter tuning.

Practical guide for researchers by Google: https://github.com/google-research/tuning_playbook or [[@tuningplaybookgithub]]


## Research Framework

![[research-framework.png]]



## Gradient-Boosting
Our implementation of gradient-boosted trees is based on *CatBoost* ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]]5--6) due to its native support for categorical variables and its efficient implementation on gls-gpu. Though, we expect the chosen library to have marginal effects on performance as discussed earlier in cref-gradient-boosting. 

![[gbm-log-loss-accuracy.png]]
(One iteration equals one tree added to the ensemble. )

Use to motivate?💡
<mark style="background: #FFB8EBA6;">- Write why we use a quasi-random search during exploration and Bayesian search during exploitation phase. (see https://github.com/google-research/tuning_playbook)
- Isolate the optimization of different hyperparameters e. g., optimizer, activation functions etc. during the exploration phase. Take into account nuance parameters. Decompose into smaller problems, that are manageable.</mark>

gap between training and validation loss💡Though there is some gap between training and validation performance, the gap grows only minimally with model size and training time, suggesting that most of the gap comes from a difference in difficulty rather than overfitting (From gpt-3 paper)

Fig-learning-curves visualizes the learning curves of the default implementation on the ISE training and validation set / feature set classical. The complete configuration is documented in cref-appendix-loss. Several conclusions can be drawn from this plot (...) First, the model is overfitting the training data, as indicated by the significant gap between training and validation accuracies. As the training performance does not transfer to the validation set, we apply regularization techniques to improve generalization performance. Second, the validation loss spikes for larger ensembles while validation accuracy continues to improve. In other words, the correctness of the predicted class improves, but the ensemble becomes less confident in the correctness of the prediction. Part of this behaviour may be explained with the log-loss being unbound, whereby single incorrect predictions can lead to an explosion in loss. Consider the case where the ensemble learns to confidently classify a trade based on the training samples, but the label is different for the validation set (...)

*Improvements / Configuration:*
-> train loss decreases -> predictions become more accurate, but model is less certain about the predictions
https://stats.stackexchange.com/questions/282160/how-is-it-possible-that-validation-loss-is-increasing-while-validation-accuracy
https://github.com/keras-team/keras/issues/3755

We leverage the following architectural changes to improve the performance of gradient-boosting. The effects on validation accuracy and log-loss over the default configuration are visualized in cref-fig. To derive the plots, all other parameters are kept at defaults and a single parameter is varied. While this approach neglects inter-changes between parameters, it can still provide guidance for the optimal training configuration. The training is performed on the ISE training set with FS1 and metrics are reported on the validation set.

*Growth Strategy:*
By default, *CatBoost* grows oblivious regression trees ([[@dorogushCatBoostGradientBoosting]]4). In this configuration, trees are symmetric and grown level-wise and splits are performed on the same feature and split values across all nodes of a single level. This strategy is computationally efficient, but may sacrifice performance, as the split is performed on the same features or split values. Following ([[@chenXGBoostScalableTree2016]]4) we switch to a leaf-wise growth strategy, whereby terminal nodes are selected for splitting that provide the largest improvement in loss. Nodes within the same level may result from different feature splits and thereby fit data more closely. Leaf-wise growth also matches our intuition of split finding in cref-[[🎄Decision Trees]]. Changing to a leaf-wise growth improves validation accuracy by perc-num, but hardly improves the loss.

*Sample Weighting:*
The work of ([[@grauerOptionTradeClassification2022]]36--38) suggests a strong temporal shift in the data, with the performance of classical trade classification rules to deteriorate over time. In consequence, the predictability of features defined on top of classical rules diminishes over time and patterns learned on old observations loose their relevance for predictions of test samples. In absence of an update mechanism, we extend the log loss by a sample weighting scheme to assign higher weights to recent training samples and gradually decay weights over time. Validation and test samples are equally-weighted. Sample weighting turns out to be vital for high validation performance. It positively affects the correctness and the confidence in the prediction. 

$$
\frac{-\sum_{i=1}^N w_i\left(c_i \log \left(p_i\right)+\left(1-c_i\right) \log \left(1-p_i\right)\right)}{\sum_{i=1}^N w_i}
$$
(from https://catboost.ai/en/docs/concepts/loss-functions-classification 🔢) (logloss)

*Border Count:*
Split finding in regression trees of gradient-boosting is typically approximated through quantization, whereby all numeric features are discretised first into a fixed number of buckets through histogram building and splits are evaluated at the border of the buckets ([[@dorogushCatBoostGradientBoosting]]4) and ([[@keLightGBMHighlyEfficient2017]]2). We raise the border count to $254$, which increases the number of split candidates per feature through a finer quantization. In general, accuracy increases at the cost of computational efficiency. In the experiment above, the improvements in validation loss and accuracy are minor compared to the previous changes.

*Early Stopping:*
To avidly fight overfitting, we monitor the training and validation accuracies when adding new trees to the ensemble and suspend training once validation accuracy decreases for 100 iterations. The ensemble is then pruned to achieve highest validation accuracy. In consequence, the maximum size of the ensemble may not be fully exhausted. For the experiment above, early stopping does not apply, as validation accuracy continues to improve for larger ensembles. We employ additional measures to address overfitting, but treat them as a tunable hyperparameter. More details are provided in cref-hyperparameter-tuning.

We combine these ideas to leverage the improvements for our large-scale studies in cref-hyperparameter-tuning. 

**Classical Rules**
Classical trade classification rules serve as a benchmark in our work. We implement them as a generic classifier which combines arbitrary trade classification rules through stacking, as covered in cref-stacking. The implementation conforms to the interface of *scikit-learn* proposed by ([[@pedregosaScikitlearnMachineLearning2018]]).

In case where no classification is not feasible due to missing data or the definition of the rules itself, we resort to a random classification, which achieves an average accuracy of perc-50. In this regard we deviate from ([[@grauerOptionTradeClassification2022]]29--32), who treat unclassified trades as falsely classified resulting in perc-0 accuracy. In our setting, this procedure would introduce a bias towards machine learning classifiers.

**Gradient-Boosting + Self-Training**
To incorporate unlabelled trades into the training procedure, we combine gradient boosting with a self-training classifier, as derived in cref-self-training-gbm. We repeat self-training for 2 iterations and require the predicted class probability to exceed $\tau=0.9$. As the entire ensemble is rebuilt three times, the relatively low number of iterations and high confidence threshold, is a compromise to balance computational requirements and the need for high-quality predictions. The base classifier is otherwise identical to supervised gradient boosting from cref-supervised-training.

**Transformer + Pre-Training**
As derived in cref-pretraining, we
- For pre-training objectives see: https://github.com/puhsu/tabular-dl-pretrain-objectives/
- For implementation of masked language modelling see https://nn.labml.ai/transformers/mlm/index.html

Following ([[@rubachevRevisitingPretrainingObjectives2022]]14) we set the hidden dimension of the classification head to 512. The configuration of the Transformer with pre-training objective is otherwise identical to the Transformer trained from scratch.

“For deep models with transfer learning, we tune the hyperparameters on the full upstream data using the available large upstream validation set with the goal to obtain the best performing feature extractor for the pre-training multi-target task. We then fine-tune this feature extractor with a small learning rate on the downstream data. As this strategy offers considerable performance gains over default hyperparameters, we highlight the importance of tuning the feature extractor and present the comparison with default hyperparameters in Appendix B as well as the details on hyperparameter search spaces for each model.” ([Levin et al., 2022, p. 6](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/QCVUFCDQ?page=6&annotation=PICSZEZU)) [[@levinTransferLearningDeep2022]]
When finetuning 

“Pretraining. Pretraining is always performed directly on the target dataset and does not exploit additional data. The learning process thus comprises two stages. On the first stage, the model parameters are optimized w.r.t. the pretraining objective. On the second stage, the model is initialized with the pretrained weights and finetuned on the downstream classification or regression task. We focus on the fully-supervised setup, i.e., assume that target labels are provided for all dataset objects. Typically, pretraining stage involves the input corruption: for instance, to generate positive pairs in contrastive-like objectives or to corrupt the input for reconstruction in self-prediction based objectives. We use random feature resampling as a proven simple baseline for input corruption in tabular data [4, 42]. Learning rate and weight decay are shared between the two stages (see Table 11 for the ablation). We fix the maximum number of pretraining iterations for each dataset at 100k. On every 10k-th iteration, we compute the value of the pretraining objective using the hold-out validation objects for early-stopping on large-scale WE, CO and MI datasets. On other datasets we directly finetune the current model every 10k-th iteration and perform early-stopping based on the target metric after finetuning (we do not observe much difference between early stopping by loss or by downstream metric, see Table 12).” (Rubachev et al., 2022, p. 4)

**Transformer**
As derived in cref-supervised selection, we rely on FT-Transformer of ([[@gorishniyRevisitingDeepLearning2021]]) as our second model. Training Transformers has been found non-trivial and requires a carefully designed training setup of model, optimizer, and learning rate schedule ([[@liuUnderstandingDifficultyTraining2020]]1). We investigate minor modifications to the default FT-Transformer to stabilize training and improve overall performance. 

The loss and accuracy of the default FT-Transformer trained for 10 epochs on gls-ise dataset  are visualized in cref-x. The configuration is listed in the cref-appendix.

![[train-val-loss-acc-transformer.png]]

(One step equals one batched gradient update. Training is performed for 1000 steps) (similarily in https://arxiv.org/pdf/2005.14165.pdf)



Our implementation is based on *PyTorch* ([[@paszkePyTorchImperativeStyle2019]]).



(What can be seen? General overview for neural nets in [[@melisStateArtEvaluation2017]]. Also, [[@kadraWelltunedSimpleNets2021]])

The best results of the efficient Transformer models can be obtained with the BigBird and LED models with a performance of 53.58% and 53.7% respectively. Using the Longformer encodings yields a comparably low accuracy of 48.02%. <mark style="background: #FF5582A6;">Of note is</mark>, however, that the classification model which receives the Longformer encodings fits the training data nearly perfectly. The models receiving BigBird and LED encodings on the other hand merely yield a training accuracy of 81.41% and 75.47% respectively.<mark style="background: #FFB8EBA6;"> There appears to be a clear trade-off relationship</mark> between the degree to which the models are able to fit to the training data and the results obtained on the test data. This is also evident in the performance of the baseline models. See Figure 6.2. for an illustration of the described trade-off.

- See tips in [[@tuningplaybookgithub]]
- Transformers are much more elaborate to train than gradient boosting approaches. Training of the transformer has been found non-trivial [[@liuUnderstandingDifficultyTraining2020]]
- Motivate the importance of regularized neural nets with [[@kadraWelltunedSimpleNets2021]] papers. Authors state, that the improvements from regularization of neural nets are very pronounced and highly significant. Discuss which regularization approaches are applied and why.  
- Similarly, [[@heBagTricksImage2018]] show how they can improve the performance of neural nets for computer vision through "tricks" like learning rate scheduling.
- Also see [[@shavittRegularizationLearningNetworks2018]] for regularization in neural networks for tabular data.
- post norm / pre-norm / lr warm up [[@xiongLayerNormalizationTransformer2020]].  Use of Post-Norm (Hello [[🤖TabTransformer]]) has been deemed outdated in Transformers due to a more fragile training process (see [[@gorishniyRevisitingDeepLearning2021]]). 

- training tips for Transformer https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/
- Might use additional tips from here: ([[@liuRoBERTaRobustlyOptimized2019]] and [[@liuUnderstandingDifficultyTraining2020]])

- For training the transformer see: https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it

2.2 Architecture Following recent work on large language models, our network is based on the transformer architecture (Vaswani et al., 2017). We leverage various improvements that were subsequently proposed, and used in different models such as PaLM. Here are the main difference with the original architecture, and where we were found the inspiration for this change (in bracket): Pre-normalization [GPT3]. To improve the training stability, we normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing function, introduced by Zhang and Sennrich (2019). SwiGLU activation function [PaLM]. We replace the ReLU non-linearity by the SwiGLU activation function, introduced by Shazeer (2020) to improve the performance. We use a dimension of 2 3 4d instead of 4d as in PaLM. Rotary Embeddings [GPTNeo]. We remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE), introduced by Su et al. (2021), at each layer of the network. The details of the hyper-parameters for our different models are given in Table 2.

Clearly, overfitting is evident, (...)

**Learning rate schedule**
We apply a 

We use a cosine learning rate schedule, such that the final learning rate is equal to 10% of the maximal learning rate. We use a weight decay of 0.1 and gradient clipping of 1.0. We use 2, 000 warmup 0 200 400 600 800 1000 1200 1400 Billion of tokens 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 Training loss LLaMA 7B LLaMA 13B LLaMA 33B LLaMA 65B Figure 1: Training loss over train tokens for the 7B, 13B, 33B, and 65 models. LLaMA-33B and LLaMA65B were trained on 1.4T tokens. The smaller models were trained on 1.0T tokens. All models are trained with a batch size of 4M tokens. steps, and vary the learning rate and batch size with the size of the model (see Table 2 for details)

From preliminary tests, we observed that the use of a learning rate schedule with a short learning rate warm-up phase both stabilizes training and improves accuracy as derived in \cref{sec:training-of-supervised-models}. Their constant learning rate and our decayed learning rate may thus not be entirely comparable. Additionally, we implement early stopping and halt training after \num{15} consecutive decreases in validation accuracy, affecting the effective number of epochs. Both techniques have not been used by the original authors to provide a conservative baseline, for the sake of a fair comparison in our work, both techniques should be used.

Their constant learning rate and our decayed learning rate may thus not be entirely comparable. Additionally, we employ early stopping and halt training after 15 consecutive decreases in validation accuracy, affecting the effective number of epochs. Both techniques have not been used by the orginal author's to provide a conservative baseline ([[@gorishniyRevisitingDeepLearning2021]]5), for the sake of a fair comparison in our context both techniques should be used.

- One commonly used technique for training a Transformer is learning rate warm-up. This means that we gradually increase the learning rate from 0 on to our originally specified learning rate in the first few iterations. Thus, we slowly start learning instead of taking very large steps from the beginning. In fact, training a deep Transformer without learning rate warm-up can make the model diverge and achieve a much worse performance on training and testing (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html).

- Lower the learning rate when the model stagnates, but don't start too low.  Try cyclic learning rates https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html
- cycling procedure was proposed in [[@loshchilovSGDRStochasticGradient2017]] and [[@smithCyclicalLearningRates2017]]
- for intuitive explanation on learning rate warm up see https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
- One might has to adjust the lr when scaling across multiple gpus contains a nice discussion.
- Use learning rate warmup for post-LN transformers and maybe also for other
- Some intuition on learning rate warm-up can be found in [[@liuVarianceAdaptiveLearning2021]] and https://stackoverflow.com/a/55942518/5755604
- In case of diverged training, try gradient clipping and/or more warmup steps. (found in [[@popelTrainingTipsTransformer2018]])


*Activation Function:*
Motivated by previous research, we conducted an experiment by replacing the $\operatorname{ReLU}$ activation with the $\operatorname{GELU}$ activation function ([[@hendrycksGaussianErrorLinear2020]]) in the classification head and the gated variant $\operatorname{ReGLU}$ with the gated variant $\operatorname{GEGLU}$ ([[@shazeerGLUVariantsImprove2020]]2) in the glspl-FFN. However, we observe no advantage in terms of validation accuracy or loss. As a result, we decided to stick with the default configuration, as the performance is comparable.

**Label Smoothing:**
We experiment with label smoothing () to mitigate the networks tendency to overfit the training data  

**Label Smoothing** is a regularization technique that introduces noise for the labels. This accounts for the fact that datasets may have mistakes in them, so maximizing the likelihood of log⁡�(�∣�) directly can be harmful. Assume for a small constant �, the training set label � is correct with probability 1−� and incorrect otherwise. Label Smoothing regularizes a model based on a [softmax](https://paperswithcode.com/method/softmax) with � output values by replacing the hard 0 and 1 classification targets with targets of ��−1 and 1−� respectively.

Label smoothing has been used successfully to improve the accuracy of deep learning models across a range of tasks, including image classification, speech recognition, and machine translation (Table 1). Szegedy et al. [6] originally proposed label smoothing as a strategy that improved the performance of the Inception architecture on the ImageNet dataset, and many state-of-the-art image classification models have incorporated label smoothing into training procedures ever since [7, 8, 9]

In machine learning or deep learning, we usually use a lot of regularization techniques, such as L1, L2, dropout, etc., to prevent our model from overfitting. In classification problems, sometimes our model would learn to predict the training examples extremely confidently. This is not good for generalization.
In this blog post, I am going to talk about label smoothing as a regularization technique for classification problems to prevent the model from predicting the training examples too confidently.

but find no advantage with regard to validation performance. 

**Sample weighting:**
We transfer the idea of sample weighting from gls-gbm. Again, the contribution of individual training samples to the cross-entropy loss is scaled by a sample weight which is higher for more recent observations. As the loss in this configuration shows spurious patterns of early overfitting, we equally weight all samples instead.

**Batch Size:**
We maximize the batch size on 

- Estimate the maximum batch size early on, as many parameters depend on it. (see https://github.com/google-research/tuning_playbook)

We do not distribute batches across multiple accelerators, as it would require

*Early Stopping and Checkpointing:*
Similar to the gls-gbm, we prematurely halt training based on an consecutive decrease in validation accuracy. We set the patience to 10 epochs and restore the best model in terms of validation accuracy through checkpointing. Checkpointing is performed at the end of each epoch. 


*Optimizer*
We optimize our models using AdamW optimizer ([[@loshchilovDecoupledWeightDecay2019]]) and treat the weight decay term in AdamW as a hyperparameter. 

- Use weight decay of 0.1 for a small amount of regularization [[@loshchilovDecoupledWeightDecay2019]].


To this end, we extend the training setup of ([[@gorishniyRevisitingDeepLearning2021]]) with a learning rate schedule and early stopping



- Introduce notion of effective batch size (batch size when training is split across multiple gpus; see [[@popelTrainingTipsTransformer2018]] p. 46)
- For random shuffling with stochastic gradient descent see [[@lecunEfficientBackProp2012]]
-   Shuffle in memory if samples are otherwise correlated. (see https://www.lesswrong.com/posts/b3CQrAo2nufqzwNHF/how-to-train-your-transformer)

As found in [KMH+20, MKAT18], larger models can typically use a larger batch size, but require a smaller learning rate. We measure the gradient noise scale during training and use it to guide our choice of batch size [MKAT18]. Table 2.1 shows the parameter settings we used. To train the larger models without running out of memory, we use a mixture of model parallelism within each matrix multiply and model parallelism across the layers of the network. All models were trained on V100 GPU’s on part of a high-bandwidth cluster provided by Microsoft. Details of the training process and hyperparameter settings are described in Appendix B. (Found in gpt paper)
- Base implementation on https://github.com/BlackHC/toma or this blog post https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1 (nice idea with the dummy data.)

We do exploit model parallelism



We scale the *effective batch size* 


Visualize the decision of 


Our models are trained using the AdamW optimizer (Loshchilov and Hutter, 2017), with the following hyper-parameters: β1 = 0.9, β2 = 0.95. We use a cosine learning rate schedule, such that the final learning rate is equal to 10% of the maximal learning rate. We use a weight decay of 0.1 and gradient clipping of 1.0. We use 2, 000 warmup 0 200 400 600 800 1000 1200 1400 Billion of tokens 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 Training loss LLaMA 7B LLaMA 13B LLaMA 33B LLaMA 65B Figure 1: Training loss over train tokens for the 7B, 13B, 33B, and 65 models. LLaMA-33B and LLaMA65B were trained on 1.4T tokens. The smaller models were trained on 1.0T tokens. All models are trained with a batch size of 4M tokens. steps, and vary the learning rate and batch size with the size of the model (see Table 2 for details).

To train all versions of GPT-3, we use Adam with β1 = 0.9, β2 = 0.95, and  = 10−8 , we clip the global norm of the gradient at 1.0, and we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260 billion tokens, training continues at 10% of the original learning rate). There is a linear LR warmup over the first 375 million tokens. We also gradually increase the batch size linearly from a small value (32k tokens) to the full value over the first 4-12 billion tokens of training, depending on the model size. Data are sampled without replacement during training (until an epoch boundary is reached) to minimize overfitting. All models use weight decay of 0.1 to provide a small amount of regularization [LH17]. During training we always train on sequences of the full nctx = 2048 token context window, packing multiple documents into a single sequence when documents are shorter than 2048, in order to increase computational efficiency. Sequences with multiple documents are not masked in any special way but instead documents within a sequence are delimited with a special end of text token, giving the language model the information necessary to infer that context separated by the end of text token is unrelated. This allows for efficient training without need for any special sequence-specific masking.

We make several optimizations to improve the training speed of our models. First, we use an efficient implementation of the causal multi-head attention to reduce memory usage and runtime. This implementation, available in the xformers library,2 is inspired by Rabe and Staats (2021) and uses the backward from Dao et al. (2022). This is achieved by not storing the attention weights and not computing the key/query scores that are masked due to the causal nature of the language modeling task. To further improve training efficiency, we reduced the amount of activations that are recomputed during the backward pass with checkpointing. More precisely, we save the activations that are expensive to compute, such as the outputs of linear layers. This is achieved by manually implementing the backward function for the transformer layers, instead of relying on the PyTorch autograd. To fully benefit from this optimization, we need to  reduce the memory usage of the model by using model and sequence parallelism, as described by Korthikanti et al. (2022). Moreover, we also overlap the computation of activations and the communication between GPUs over the network (due to all_reduce operations) as much as possible. When training a 65B-parameter model, our code processes around 380 tokens/sec/GPU on 2048 A100 GPU with 80GB of RAM. This means that training over our dataset containing 1.4T tokens takes approximately 21 days.


### Logistic regression

- KISS 💘

[[@somepalliSAINTImprovedNeural2021]] use logistic regression. I really like the fact they also compare a simple logistic regression to these models, because if you’re not able to perform notably better relative to the simplest model one could do, then why would we care? The fact that logistic regression is at times competitive and even beats boosting/SAINT methods occasionally gives me pause though. Perhaps some of these data are not sufficiently complex to be useful in distinguishing these methods? It is realistic though. While it’s best not to assume as such, sometimes a linear model is appropriate given the features and target at hand.

Start with something simple e. g., Logistic Regression or Gradient Boosted Trees, due to being well suited for tabular data. Also  [[@grauerOptionTradeClassification2022]] could be a baseline.

Train a simple fully connected network as baseline and sanity check. We could argue that TabTransformer callapses to one.



