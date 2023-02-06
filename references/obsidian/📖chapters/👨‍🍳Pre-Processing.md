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

“The LR, EMO, and CLNV algorithms require assigning one bid and ask quote to each trade in order to classify it. In an ideal data environment where at the time of the trade we record only one quote change, we know that the quotes in effect at the time of the trade are the last ones recorded before the time of the trade. With several quote changes occurring at the same time as the trade, however, the choice is less clear. For example, with one trade and three quote changes recorded at the same millisecond, the quotes corresponding to the trade could be the last quotes from before the millisecond or one of the first two recorded at the millisecond. The convention in such a case is to take the last ask and bid from before the time of the trade.” (Jurkatis, 2022, p. 7) -> not sure how Caroline did the matching


“As the probability of observing only buy trades or only sell trades decreases with an increasing number of trades, the number of trades per option day is lower and the time between two trades is higher in our matched samples compared to their full sample equivalents. Because tick tests depend on the information from preceding or succeeding trades as a precise signal for the fair option price, our results might therefore underestimate their performance.” ([[@grauerOptionTradeClassification2022]]., 2022, p. 9)

**What might not apply:**

“The LR, EMO, and CLNV algorithms require assigning one bid and ask quote to each trade in order to classify it. In an ideal data environment where at the time of the trade we record only one quote change, we know that the quotes in effect at the time of the trade are the last ones recorded before the time of the trade. With several quote changes occurring at the same time as the trade, however, the choice is less clear. For example, with one trade and three quote changes recorded at the same millisecond, the quotes corresponding to the trade could be the last quotes from before the millisecond or one of the first two recorded at the millisecond. The convention in such a case is to take the last ask and bid from before the time of the trade.” (Jurkatis, 2022, p. 7)

“With several quote changes occurring at the same time as the trade, however, the choice is less clear. For example, with one trade and three quote changes recorded at the same millisecond, the quotes corresponding to the trade could be the last quotes from before the millisecond or one of the first two recorded at the millisecond. The convention in such a case is to take the last ask and bid from before the time of the trade.” (Jurkatis, 2022, p. 7)

“An alternative suggested by Holden and Jacobsen (2014) to circumvent the problem of imprecise timestamps is to transform timestamps to a higher precision. This is done by interpolating the recorded times according to: t = s + 2i − 1 2I , i = 1, ... , I, where t is the interpolated timestamp and s is the originally recorded time. I isthenumberoftradesorthenumberofchanges at the ask or bid at time s depending on which timestamp to interpolate. The algorithm then proceeds as described above using the last ask and bid price from before the time of the trade according to the interpolated time.” (Jurkatis, 2022, p. 7)