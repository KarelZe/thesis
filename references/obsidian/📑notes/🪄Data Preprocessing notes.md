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


**Filter:**
- What preprocessing have been applied. Minimal set of filters. See [[@grauerOptionTradeClassification2022]].

**Class imbalances:**
- Data set is slightly imbalanced. Would not treat, as difference is only minor and makes it harder. Could make my final decision based on [[@japkowiczClassImbalanceProblem2002]] [[@johnsonSurveyDeepLearning2019]]. Some nice background is also in [[@huyenDesigningMachineLearning]]
- [[@huyenDesigningMachineLearning]] discusses different sampling strategies. -> Could make sense to use stratified sampling or weighted sampling. 

**Time consistency:**
- Check time consistency (found idea here: https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600/notebook)
> We added 28 new feature above. We have already removed 219 V Columns from correlation analysis done [here](https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id). So we currently have 242 features now. We will now check each of our 242 for "time consistency". We will build 242 models. Each model will be trained on the first month of the training data and will only use one feature. We will then predict the last month of the training data. We want both training AUC and validation AUC to be above `AUC = 0.5`. It turns out that 19 features fail this test so we will remove them. Additionally we will remove 7 D columns that are mostly NAN. More techniques for feature selection are listed [here](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308)
> 