This chapter documents the basic training setup, and sets up a baseline before tuning

## Gradient-Boosting

- early stopping
- exponential weighting
- quantization
- weight decay
- 

Visualize model parameters:
![[viz-model-params.png]]
(from https://arxiv.org/pdf/2005.14165.pdf)


deas to try out in CatBoost
- look into search space, where did I choose the search space poorly
- use custom weighting factor
- make relatative distance to quote categorical / cut-off at threshold
- Try out `lossguided` growing strategy. Add it to objective.
- Try out `XGBoost` or `lightgbm`
- Think about indicator variables
- Think about interaction features, which e. g.,which features could be summed / multiplied etc.?
- Perform an error analysis. For which classes does CatBoost do so poorly? See some ideas here. https://elitedatascience.com/feature-engineering-best-practices
- Check time consistency (found idea here: https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600/notebook)
> We added 28 new feature above. We have already removed 219 V Columns from correlation analysis done [here](https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id). So we currently have 242 features now. We will now check each of our 242 for "time consistency". We will build 242 models. Each model will be trained on the first month of the training data and will only use one feature. We will then predict the last month of the training data. We want both training AUC and validation AUC to be above `AUC = 0.5`. It turns out that 19 features fail this test so we will remove them. Additionally we will remove 7 D columns that are mostly NAN. More techniques for feature selection are listed [here](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308)
> 

- Could try featuretools. Not sure, if it actually has some advantage. The thing is features are not to fancy here. (Check out this kernel https://www.kaggle.com/code/vbmokin/titanic-featuretools-automatic-fe-fs/notebook?scriptVersionId=43519589)

- Go for even greater ensembles, if stopped out early e. g., 20000. See kaggle notebooks e. g.,:
```python
# https://www.kaggle.com/code/kyakovlev/ieee-fe-for-local-test/notebook
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'n_estimators':80000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                }
```
- add statistics /  group features like min, max, std dev etc.
```python
# found at: https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/108575
temp = df.groupby('card1')['TransactionAmt'].agg(['mean'])   
    .rename({'mean':'TransactionAmt_card1_mean'},axis=1)
df = pd.merge(df,temp,on='card1',how='left')```
```
- add frequency encoding
```python
# found at: https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/108575
temp = df['card1'].value_counts().to_dict()
df['card1_counts'] = df['card1'].map(temp)
```
- Fill missing values with something obvious e. g., `train_df.fillna(-999,inplace=True)`

```python
# found at: https://www.kaggle.com/code/kooaslansefat/tips-tricks-catboost-version
# create new features from newly created categorical variables
age_group_feat_eng = True
bmi_group_feat_eng = True

if age_group_feat_eng:
    train_X, test_X = _group_feature_eng(combined_df, n_train, 'age_group', num_feats)

if bmi_group_feat_eng:
    train_X, test_X = _group_feature_eng(combined_df, n_train, 'bmi_group', num_feats)

def _group_feature_eng(combined_df, n_train, group_var, num_feats):
    
    """
    combined_df: the combined train & test datasets.
    n_train: number of training observations
    group_var: the variable we'd like to group by
    num_feat: numerical features
    
    This function loops through all numerical features, 
    group by the variable and compute new statistics of the numerical features.
    """
    
    grouped = combined_df.groupby(group_var)

    for nf in num_feats:

        combined_df[group_var + '_' + nf + '_max'] = grouped[nf].transform('max')
        combined_df[group_var + '_' + nf + '_min'] = grouped[nf].transform('min')
        combined_df[group_var + '_' + nf + '_mean'] = grouped[nf].transform('mean')
        combined_df[group_var + '_' + nf + '_skew'] = grouped[nf].transform('skew')
        combined_df[group_var + '_' + nf + '_std'] = grouped[nf].transform('std')

    train_X = combined_df.iloc[:n_train]
    test_X = combined_df.iloc[n_train:]
    
    return train_X, test_X
```

Study feature interactions


```python
feature_interaction = [[X.columns[interaction[0]], X.columns[interaction[1]], interaction[2]] for i,interaction in interactions.iterrows()]
feature_interaction_df = pd.DataFrame(feature_interaction, columns=['feature1', 'feature2', 'interaction_strength'])
feature_interaction_df.head(10)
```

Restructure TOC conforming to CRISPDM:

```
1.  Project Scoping / Data Collection
2.  Exploratory Analysis
3.  Data Cleaning
4.  **Feature Engineering**
5.  Model Training (including cross-validation to tune hyper-parameters)
6.  Project Delivery / Insights
```




As found in [KMH+20, MKAT18], larger models can typically use a larger batch size, but require a smaller learning rate. We measure the gradient noise scale during training and use it to guide our choice of batch size [MKAT18]. Table 2.1 shows the parameter settings we used. To train the larger models without running out of memory, we use a mixture of model parallelism within each matrix multiply and model parallelism across the layers of the network. All models were trained on V100 GPU’s on part of a high-bandwidth cluster provided by Microsoft. Details of the training process and hyperparameter settings are described in Appendix B.


![[Pasted image 20230428134902.png]]


Our search space is reported in Table-X, which we laid out based on the recommendations in ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]]20) and ([[@gorishniyRevisitingDeepLearning2021]]18) and ([[@rubachevRevisitingPretrainingObjectives2022]]., 2022, p. 4) with minor deviations. For gradient-boosting we raise the border count to $256$, which increases the number of split candidates per feature through a finer quantization, Expectedly, accuracy increases at the cost of computational efficiency. The size of the ensemble $M$ may not be fully exhausted. Acknowleding the observations ([[@friedmanGreedyFunctionApproximation2001]]14), that the learning rate $\lambda$ the learning rate and the size of the ensemble have a strong interdependence, we only tune the learning rate and stop adding new trees to the ensemble, once the validation accuracy decreases for consecutive (...) steps.

We grow symmetric trees, which acts as a regularizer.

The hyperparameter search for the FT-Transformer is identical to ([[@gorishniyRevisitingDeepLearning2021]]18) variant (b). From preliminary tests, we observed that the use of a learning rate schedule with a short learning rate warm-up phase both stabilizes training and improves accuracy (cp. cref-training-of-supervised). Their constant learning rate and our decayed learning rate may thus not be entirely comparable. Additionally, we employ early stopping and halt training after 15 consecutive decreases in validation accuracy, affecting the effective number of epochs. Both techniques have not been used by the orginal author's to provide a conservative baseline ([[@gorishniyRevisitingDeepLearning2021]]5), for the sake of a fair comparison in our context both techniques should be used.

**Failed attempts:**
We experimented with label smoothing, 

Visualize the decision of 

To train all versions of GPT-3, we use Adam with β1 = 0.9, β2 = 0.95, and  = 10−8 , we clip the global norm of the gradient at 1.0, and we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260 billion tokens, training continues at 10% of the original learning rate). There is a linear LR warmup over the first 375 million tokens. We also gradually increase the batch size linearly from a small value (32k tokens) to the full value over the first 4-12 billion tokens of training, depending on the model size. Data are sampled without replacement during training (until an epoch boundary is reached) to minimize overfitting. All models use weight decay of 0.1 to provide a small amount of regularization [LH17]. During training we always train on sequences of the full nctx = 2048 token context window, packing multiple documents into a single sequence when documents are shorter than 2048, in order to increase computational efficiency. Sequences with multiple documents are not masked in any special way but instead documents within a sequence are delimited with a special end of text token, giving the language model the information necessary to infer that context separated by the end of text token is unrelated. This allows for efficient training without need for any special sequence-specific masking.

Visualize effect 

![[Pasted image 20230428110412.png]]


Classical trade signing algorithms, such as the tick test, are also impacted by missing values. In theses cases, we defer to a random classification or a subsequent rule, if rules can not be computed. Details are provided in section [[💡Training of models (supervised)]].

## Gradient-Boosting
- Refer to discussion. Gradient-boosted trees are prone to overfit
- Employ early stopping
- Visualize loss on training set evaluation set
- What is the configuration
- What can be inferred from the training / validation loss?
- What is the loss function used in my gradient-boosting 
- How to handle categoricals?



## Transformer
[[🤖Training of the Transformer]]

- what parameters
- how to handle caregoricals
- early stopping
- gradient-checkpointing
- adam with weight decay
- attention dropout / feed forward drop out
- visualize lr decay

## Logistic regression
- Think about simple baseline e. g., logistic regression


![[Pasted image 20230427155407.png]]


Look into grooking: https://arxiv.org/pdf/2201.02177.pdf
![[grocking.png]]

![[Pasted image 20230427161504.png]]

- Research Question 1: Which methods and models to encode long text sequences are most suited for downstream machine learning tasks?

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



The best results of the efficient Transformer models can be obtained with the BigBird and LED models with a performance of 53.58% and 53.7% respectively. Using the Longformer encodings yields a comparably low accuracy of 48.02%. <mark style="background: #FF5582A6;">Of note is</mark>, however, that the classification model which receives the Longformer encodings fits the training data nearly perfectly. The models receiving BigBird and LED encodings on the other hand merely yield a training accuracy of 81.41% and 75.47% respectively.<mark style="background: #FFB8EBA6;"> There appears to be a clear trade-off relationship</mark> between the degree to which the models are able to fit to the training data and the results obtained on the test data. This is also evident in the performance of the baseline models. See Figure 6.2. for an illustration of the described trade-off.

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













