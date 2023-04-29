**Filter:**
- What preprocessing have been applied. Minimal set of filters. See [[@grauerOptionTradeClassification2022]].

**Class imbalances:**
- Data set is slightly imbalanced. Would not treat, as difference is only minor and makes it harder. Could make my final decision based on [[@japkowiczClassImbalanceProblem2002]] [[@johnsonSurveyDeepLearning2019]]. Some nice background is also in [[@huyenDesigningMachineLearning]]
- [[@huyenDesigningMachineLearning]] discusses different sampling strategies. -> Could make sense to use stratified sampling or weighted sampling. 

**Time consistency:**
- Check time consistency (found idea here: https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600/notebook)
> We added 28 new feature above. We have already removed 219 V Columns from correlation analysis done [here](https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id). So we currently have 242 features now. We will now check each of our 242 for "time consistency". We will build 242 models. Each model will be trained on the first month of the training data and will only use one feature. We will then predict the last month of the training data. We want both training AUC and validation AUC to be above `AUC = 0.5`. It turns out that 19 features fail this test so we will remove them. Additionally we will remove 7 D columns that are mostly NAN. More techniques for feature selection are listed [here](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308)
> 