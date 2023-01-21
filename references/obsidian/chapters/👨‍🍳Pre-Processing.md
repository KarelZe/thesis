- infer minimimal data types to minimize memory requirements. No data loss happening as required resolution for e. g., mantisse is considered.  (see [here](https://github.com/KarelZe/thesis/blob/main/notebooks/1.0-mb-data_preprocessing_mem_reduce.ipynb) and [here](https://www.kaggle.com/code/gemartin/load-data-reduce-memory-usage/notebook)(not used) or [here](https://www.kaggle.com/code/wkirgsn/fail-safe-parallel-memory-reduction) (not used))

**Filter:**
- What preprocessing have been applied. Minimal set of filters. See [[@grauerOptionTradeClassification2022]]

**Class imbalances:**
- Data set is slightly imbalanced. Would not treat, as difference is only minor and makes it harder. Could make my final decision based on [[@japkowiczClassImbalanceProblem2002]] [[@johnsonSurveyDeepLearning2019]]. Some nice background is also in [[@huyenDesigningMachineLearning]]
- [[@huyenDesigningMachineLearning]] discusses different sampling strategies. -> Could make sense to use stratified sampling or weighted sampling. With weighted sampling we could give classes that are notoriously hard to learn e. g., large trades or index options.

- To evaluate the performance of trade classification algoriths the true side of the trade needs to be known. To match LiveVol data, the total customer sell volume or total or total customer buy volume has to match with the transactions in LiveVol. Use unique key of trade date, expiration date, strike price, option type, and root symbol to match the samples. (see [[@grauerOptionTradeClassification2022]]) Notice, that this leads to an imperfect reconstruction!
- The approach of [[@grauerOptionTradeClassification2022]] matches the LiveVol data set, only if there is a matching volume on buyer or seller side. Results in 40 % reconstruction rate
- **fuzzy matching:** e. g., match volumes, even if there are small deviations in the volumes e. g. 5 contracts. Similar technique used for time stamps in [[@savickasInferringDirectionOption2003]]. Why might this be a bad idea?
- Discuss that only a portion of data can be reconstructed. Previous works neglected the unlabeled part. 
- Discuss how previously unused data could be used. This maps to the notion of supervised and semi-supervised learning
- Pseudo Labeling?
- We assign sells the label `0` and buys the label `1`. This has the advantage that the calculation from the logloss doesn't require any mapping. Easy interpretation as probability.

Adversarial validation can help balance training and testing sets, and improve the model performance on testing set. 

feature importance is used in adversarial validation to filter the most inconsistently distributed features sequentially. However, there is a trade-off for this method between the improvement of generalization performance and losing information by dropping features from the model. The proposed method in this paper can improve the generalization performance of the model to the testing set without losing information