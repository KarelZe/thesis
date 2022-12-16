- infer minimimal data types to minimize memory requirements. No data loss happening as required resolution for e. g., mantisse is considered.  (see [here](https://github.com/KarelZe/thesis/blob/main/notebooks/1.0-mb-data_preprocessing_mem_reduce.ipynb) and [here](https://www.kaggle.com/code/gemartin/load-data-reduce-memory-usage/notebook)(not used) or [here](https://www.kaggle.com/code/wkirgsn/fail-safe-parallel-memory-reduction) (not used))

**Filter:**
- What preprocessing have been applied. Minimal set of filters. See [[@grauerOptionTradeClassification2022]]
**Scaling:**
- TODO: Why do we perform feature scaling at all?
- TODO: Try out robust scaler, as data contains outliers. Robust scaler uses the median which is robust to outliers and iqr for scaling. 
- TODO: Try out different IQR thresholds and report impact. Similarily done here: https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/
- We scale / normalize features to a $\left[-1,1\right]$  scale using statistics estimated on the training set to avoid data leakage. This is also recommended in [[@huyenDesigningMachineLearning]]. Interestingly, she also writes that empirically the interval $\left[-1,1\right]$ works better than $\left[0,1\right]$. Also read about this on stackoverflow for neural networks, which has to do with gradient calculation.
- Scale to an arbitrary range $\left[a,b\right]$ using the formula from [[@huyenDesigningMachineLearning]]:
$$
x^{\prime}=a+\frac{(x-\min (x))(b-a)}{\max (x)-\min (x)}
$$
- Feature scaling theoretically shouldn't be relevant for gradient boosting due to the way gbms select split points / not based on distributions. Also in my tests it didn't make much of a difference for gbms but for transformers. (see https://github.com/KarelZe/thesis/blob/main/notebooks/3.0b-mb-comparsion-transformations.ipynb) 
- [[@ronenMachineLearningTrade2022]] performed no feature scaling.
- [[@borisovDeepNeuralNetworks2022]] standardize numerical features and apply ordinal encoding to categorical features, but pass to the model which ones are categorical features. 
- [[@gorishniyRevisitingDeepLearning2021]] (p. 6) use quantile transformation, which is similar to the robust scaler, see https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-pyf) Note that [[@grinsztajnWhyTreebasedModels2022]] only applied quantile transformations to all features, thus not utilize special implementations for categorical variables.
**Class imbalances:**
- Data set is slightly imbalanced. Would not treat, as difference is only minor and makes it harder. Could make my final decision based on [[@japkowiczClassImbalanceProblem2002]] [[@johnsonSurveyDeepLearning2019]]. Some nice background is also in [[@huyenDesigningMachineLearning]]
- [[@huyenDesigningMachineLearning]] discusses different sampling strategies. -> Could make sense to use stratified sampling or weighted sampling. With weighted sampling we could give classes that are notoriously hard to learn e. g., large trades or index options.

**imputation:**
- Best do a large-scale comparsion as already done for some transformations in  https://github.com/KarelZe/thesis/blob/main/notebooks/3.0b-mb-comparsion-transformations.ipynb. My feeling is that differences are neglectable. 
- Different imputation appraoches are listed in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]]. Basic observation use approaches that can inherently handle missing values. This is a good tradeoff between performance and prediction quality.
- simple overview for imputation https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779
- Refer to previous chapters e. g., on gradient boosting, where handling of missing values should be captured.
