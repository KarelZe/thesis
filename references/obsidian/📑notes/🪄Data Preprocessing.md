**Filter:**
- What preprocessing have been applied. Minimal set of filters. See [[@grauerOptionTradeClassification2022]].

**Class imbalances:**
- Data set is slightly imbalanced. Would not treat, as difference is only minor and makes it harder. Could make my final decision based on [[@japkowiczClassImbalanceProblem2002]] [[@johnsonSurveyDeepLearning2019]]. Some nice background is also in [[@huyenDesigningMachineLearning]]
- [[@huyenDesigningMachineLearning]] discusses different sampling strategies. -> Could make sense to use stratified sampling or weighted sampling. With weighted sampling we could give classes that are notoriously hard to learn e. g., large trades or index options.
