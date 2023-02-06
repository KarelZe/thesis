
title: Benchmarking missing-values approaches for predictive models on health databases
authors: Alexandre Perez-Lebel, Gaël Varoquaux, Marine Le Morvan, Julie Josse, Jean-Baptiste Poline
year: 2022
tags :  #data-preprocessing #imputation #gbm #nan #missing-value #decision-trees 
status : #📦 
related: 
- [[@breimanBaggingPredictors1996]] (Bagging is used in some of the approaches)
- [[@friedmanGreedyFunctionApproximation2001]] (technique used for classification / prediction)
- [[@josseConsistencySupervisedLearning2020]] (provides some theoretical backgorund on imputation)
- [[@rubinInferenceMissingData1976]] (This paper uses their classification which is very common).
code:
- https://github.com/dirty-data/supervised_missing

## Notes:

- Their comparsion of imputation strategies considers both two step approaches (impute first, then train) asw ell as tree-based models with intrinsic support for missing values (MIA). They use gradient-boosted trees, due to their sota performance.
- Native support for missing values in supervised machine learning predicts better than sota imputation with computational cost. Authors give examples how missing values can be treated in decision trees.
- For prediction after imputation the authors find, that adding an indicator (referred to as missing mask) to express which values have been imputed is important, suggesting that data is not missing at random. 
- Complex missing-value imputation e. g. $k$-nn can improve upon simpler approaches such as median fill, but the computational effort is often not justified.  An indicator which entries were imputed must be added to the completed data
- Following [[@rubinInferenceMissingData1976]] they differentiate missing completely at random, missing at random, probability of missing value only depends on the observed values of other variables, and missing not at random.
- They use the Friedman test and the Wilcoxon signed-rank test to test statistical significance.
- **Comparison of strategies:**
	- Delete missing values. Leads to loss in information in high and even moderate dimensions.
	- Impute missing values first, then apply predictive model. Popular due to simplicity. Allows to apply many off-the-shelf learners, that could not handle missing values otherwise.
	- Handle missing values natively in the machine learning model e. g., gradient boosting. Allows model to learn directly from incomplete data.
**Single imputation:**
- impute with *constant* mean, median, mode etc. Recent theoretical results suggests taht powerful learns such as tree-based ensembles can recognize imputed values and thus give the best possible predictions. Imputation must happen with a constant learned on the training set e. g., mean of training set.
- impute with *conditional imputation* e. g. MICE or $k$-nn. Relies on conditional dependencies between features to fill missing vlaues. as conditional imputation is hard for the predictor to differentiate from real values, a mask or missingness indicator should be provided.
**Multiple Imputation:**
- Multiple imputation reflects the uncertainty in missing values. Authors use a bagging approach. For each task they draw 100 bootstrap replicates, single impute and obtain 100 predictors  on each of the replicates. The final predictor is obtained by voting or averaging.
- Multiple imputation comes at a high computational cost. 
- Theoretically flexible learners can reach optimal performance with a single imputation pass, whatever the missing data mechanmism is. [[@josseConsistencySupervisedLearning2020]]
**Intrinsic support for missing vlaues**
- MIA has the benefit that all samples can can be used including ones with missing data. 
- In hte decision tree for each split based on a variable $j$ all samples with a missing value in variable $j$ are sent either to the left or to the right child node, on which option leads to the lowest risk. .... As such missing information can be harnessed.
- As trees with MIA support learn on training data with missing values they can also handle missing values in the test set. Useful for the missingnot-at-random (MNAR) setting, as missing values follow a certain pattern. 
- **Findings:**
	- Iterative + mask + bagging achieves the highest mean rank but is also among the most computationally demanding.
	- MIA provides a good tradeoff between performance and computational tractability. MIA gives excellent prediction performance.
	- Speedups range from 200x when combined with bagging to 2 when combined with bagging.
	- k-nn + mask and iterative + mask have algorithmic scalability issues.
	- Imputation with additional variables representing the mask perform systematically better than the mean of the same imputers without a mask.
	- Iterative or $k$-nn imputers do not perform better than imputation with constant values.
	- GBMs outperform linear models.
	- Good imputation does not imply good prediction, due to different trade-offs.
	- Missing values should be directly handled in the model (MIA) and not be prior to training.
	- Features with high missing rates are also **important** and should not simply be removed. They study the importance with feature permutation [[@breimanRandomForests2001]]. Predictions do not only rely on features with few missing data but also on features with a large degree of missing values (e. g., > 80 %). 
## Annotations:
“For prediction after imputation, we find that adding an indicator to express which values have been imputed is important, suggesting that the data are missing not at random. Elaborate missing-values imputation can improve prediction compared to simple strategies but requires longer computational time on large data. Learning trees that model missing values—with missing incorporated attribute—leads to robust, fast, and well-performing predictive modeling” ([Perez-Lebel et al., 2022, p. 1](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=1&annotation=J7A28PPT))

“Conclusions: Native support for missing values in supervised machine learning predicts better than state-of-the-art imputation with much less computational cost. When using imputation, it is important to add indicator columns expressing which values have been imputed.” ([Perez-Lebel et al., 2022, p. 1](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=1&annotation=G8C26DQ2))

“For such a problem, an important distinction between missing data mechanisms was introduced by Rubin [3]: missing completely at random (MCAR), where the probability of having missing data does not depend on the covariates; missing at random (MAR), where the probability of a missing value only depends on the observed values of other variables; and missing not at random (MNAR), which covers all other cases. MNAR corresponds to cases where the missingness carries information.” ([Perez-Lebel et al., 2022, p. 1](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=1&annotation=LQ7BCVZY))

“The simplest one is to delete all observations containing missing values. However, leaving aside the possible biases that this practice may induce, it often leads to considerable loss of information in high and even moderate dimensions” ([Perez-Lebel et al., 2022, p. 2](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=2&annotation=Z8J2TAQN))

“To deal with arbitrary subsets of input features, the most common practice currently consists in first imputing the missing values and then learning a predictive model (e.g., regression or classification) on the completed data.” ([Perez-Lebel et al., 2022, p. 2](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=2&annotation=HANE9F6Z))

“Recent theoretical results show that applying a supervised-learning regression on imputed data can asymptotically recover the optimal prediction function; however most imputation strategies, including the common imputation by the conditional expectation, create discontinuities in the regression function to learn [16].” ([Perez-Lebel et al., 2022, p. 2](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=2&annotation=IRPT6MTH))

“A small number of machine learning models can natively handle missing values, in particular popular tree-based methods. Trees greedily partition the input space into subspaces in order to minimize a risk. This non-smooth optimization scheme enables them to be easily adapted to directly learn from incomplete data.” ([Perez-Lebel et al., 2022, p. 2](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=2&annotation=B9ENAT3Q))

“High-quality conditional imputation gives good prediction provided that a variable indicating which entries were imputed is added to the completed data. However, its algorithmic complexity makes it prohibitively costly on large data” ([Perez-Lebel et al., 2022, p. 2](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=2&annotation=PX9HWKVN))

“Rather, tree-based methods with integrated support for missing values (MIA) perform as well or better, at a fraction of the computational cost” ([Perez-Lebel et al., 2022, p. 2](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=2&annotation=PMXRJN6K))

“Our experiments compare 2-step procedures based on imputation followed by regression or classification, as well as tree-based models with an intrinsic support for missing values thanks to MIA.” ([Perez-Lebel et al., 2022, p. 2](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=2&annotation=RYICNTNM))

“Constant imputation: mean and median.The simplest approach to imputation is to replace missing values by a constant such as the mean, the median, or the mode of the corresponding feature. This is frowned upon in classical statistical practice because the resulting data distribution is severely distorted compared to that of fully observed data. Yet, in a supervised setting, the goal is different from that of inferential tasks. Recent theoretical results have established that powerful learners such as those based on trees can learn to recognize such imputed values and give the best possible predictions [7].” ([Perez-Lebel et al., 2022, p. 2](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=2&annotation=8XCRSN5F))

“Conditional imputation: MICE and KNN. Powerful imputation approaches rely on conditional dependencies between features to fill in the missing values. Adapting machine learning techniques gives flexible estimators of these dependencies.” ([Perez-Lebel et al., 2022, p. 2](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=2&annotation=DQF9NRD2))

“For these reasons, it can be useful after imputation to add new binary features that encode whether a value was originally missing or not: the “mask” or “missingness indicator” [6, 7, 21]” ([Perez-Lebel et al., 2022, p. 3](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=3&annotation=7TR2NZDC))

“When estimating model parameters, it is important to reflect the uncertainty due to the missing values.” ([Perez-Lebel et al., 2022, p. 3](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=3&annotation=VBUBJ7JI))

“Indeed, it has been shown recently that a sufficiently flexible learner reaches optimal performance asymptotically with single imputation, whatever the missing data mechanism and whatever the choice of imputation function [16].” ([Perez-Lebel et al., 2022, p. 3](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=3&annotation=XXC5CGRQ))

“Because these methods all come with a substantial computing cost, we focus on the most promising approach: bagging single imputation” ([Perez-Lebel et al., 2022, p. 3](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=3&annotation=LI2QSPXU))

“More precisely, for each task we draw 100 bootstrap replicates. We then fit the single imputation and the predictive model on each of these replicates to obtain 100 predictors. Final predictions are made either by voting or by averaging (see Table A4)” ([Perez-Lebel et al., 2022, p. 3](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=3&annotation=UCIJX67S))

“It has the benefit of using all samples, including incomplete ones, to produce the splits of the input space. More precisely for each split based on variable j, all samples with a missing value in variable j are sent either to the left or to the right child node, depending on which option leads to the lowest risk.” ([Perez-Lebel et al., 2022, p. 3](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=3&annotation=B3QRTEJF))

“That makes MIA particularly suited to MNAR settings because it can harness the missingness information.” ([Perez-Lebel et al., 2022, p. 3](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=3&annotation=Z6TJRETI))

“Moreover, because trees with MIA directly learn with missing values, they provide a straightforward way of dealing with missing values in the test set” ([Perez-Lebel et al., 2022, p. 3](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=3&annotation=HLH6782E))

“We applied supervised learning to the imputed data for the imputation-based methods. We also used the tree models with their support of MIA for a direct handling of missing values” ([Perez-Lebel et al., 2022, p. 3](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=3&annotation=H8UJGIA5))

“Gradient-boosted trees are state-of-the art predictors for tabular data [24–26] and thus constitute a strong baseline” ([Perez-Lebel et al., 2022, p. 3](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=3&annotation=QQIK7HES))

“Iterative+mask+bagging obtains the best overall mean rank (2.6) across all tasks and sizes in terms of prediction score, closely followed by MIA+bagging (2.8) as shown on Fig. 1A and Table A7B.” ([Perez-Lebel et al., 2022, p. 4](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=4&annotation=U4B8YT8P))

“MIA makes it possible to navigate a trade-off between prediction performance and computational tractability: with bagging it comes close to iterative+mask with one-half the computational cost on large databases” ([Perez-Lebel et al., 2022, p. 4](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=4&annotation=VPR55X9M))

“In terms of computing time, beyond the fact that bagging multiplies the cost of every method by 100, MIA is almost always the fastest (Fig. 1B), although it gives excellent prediction performance.” ([Perez-Lebel et al., 2022, p. 4](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=4&annotation=WYZBEM4L))

“performance—doubles their computing times. At the other end of the spectrum, iterative+mask and KNN+mask are the slowest non-bagged methods. The gaps between training times of the methods increase with the size of the database, revealing the difference in algorithmic scalability.” ([Perez-Lebel et al., 2022, p. 4](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=4&annotation=UEXA3CKP))

“The Friedman test compares the mean ranks of several algorithms run on several datasets. The null hypothesis assumes that all algorithms are equivalent, i.e., their rank should be equal. Table A2 shows that the null hypothesis is rejected, with P-values much less than the 0.05 level for the sizes 2,500, 10,000, and 25,000. This indicates that ≥1 algorithm has significantly different performances from 1 other on these sizes.” ([Perez-Lebel et al., 2022, p. 4](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=4&annotation=ZV4MK7G2))

“We run a complementary analysis with a 1-sided Wilcoxon signed-rank test, used for non-parametric tests comparing algorithms pairwise. We compare MIA with every other method. The null hypothesis claims that the median of the score differences between the 2 methods is positive (respectively, negative) for the 1-sided right (1-sided left) test” ([Perez-Lebel et al., 2022, p. 4](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=4&annotation=E28C3764))

“mputations with the additional variable representing the mask perform systematically better in terms of mean prediction score than their counterpart without mask (Fig. 1A, Table A7b)” ([Perez-Lebel et al., 2022, p. 4](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=4&annotation=DY2A9MXG))

“Figure 1 shows that conditional imputation using iterative or KNN imputers does not perform consistently better than constant imputation. The overall mean rank of iterative and KNN are 9.0 and 11.5 versus 7.5 and 9.2 for mean and median, respectively (Fig. 1A and Table A7), and a similar delta is visible on the masked version.” ([Perez-Lebel et al., 2022, p. 4](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=4&annotation=HEUVCPLN))

“Indeed, bagging in itself is known to improve generalization. To answer whether the good performance of multiple imputation can be attributed to ensembling (averaging multiple predictors) or capturing the conditional distribution, we performed an additional experiment with mean+mask+bagging (see Fig. A8)” ([Perez-Lebel et al., 2022, p. 5](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=5&annotation=K2Q2Z3G7))

“It may be surprising at first that a sophisticated conditional imputation does not outperform constant imputation. Indeed, it contradicts the intuition that better imputation should lead to better prediction. Theoretical work shows that this intuition is not always true [16]: even in MAR settings, it may not hold for strongly non-linear mechanisms and little dependency across features.” ([Perez-Lebel et al., 2022, p. 5](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=5&annotation=CDSM7R2R))

“MIA, the missing-values support inside gradient-boosted trees, appears as a method of choice to deal with missing values. Once put aside the prohibitively costly bagged methods, MIA was on average the best in terms of performance in our extensive benchmark while having a low computational cost.” ([Perez-Lebel et al., 2022, p. 6](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=6&annotation=WFAED3PU))

“For imputation-based pipelines, prediction significantly improves with the missingness mask added as input features” ([Perez-Lebel et al., 2022, p. 6](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=6&annotation=4T5P7LP5))

“We checked that features’ missing rates and predictive importance were not associated. For this, we measured permutation features: the drop in a model score after shuffling a feature, thereby canceling its contribution to the model performance.” ([Perez-Lebel et al., 2022, p. 6](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=6&annotation=3HPS6HM7))

“We ran this experiment for each task and each feature using scikit-learn’s implementation (see Table A4). We found no association between a feature’s missing rate and its importance (Fig. A7). Predictions do not only rely on features with few missing values. Moreover, even features with a very high level of missing values (e.g., >80%) seem to be as important as the others.” ([Perez-Lebel et al., 2022, p. 6](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=6&annotation=FSUT2CZV))

“irst, directly incorporating missing values in treebased models with MIA gives a small but systematic improvement in prediction performance over prior imputation. Second, the computational cost of imputation using MICE or KNN becomes intractable for large datasets. Third, gradient-boosted trees give better predictions than linear models. Fourth, bagging increases predictive performance but with a severe computational cost. Fifth, good imputation does not imply good prediction because both have different trade-offs. Finally, the experiments reveal that the missingness is informative.” ([Perez-Lebel et al., 2022, p. 7](zotero://select/library/items/WCMHL2JY)) ([pdf](zotero://open-pdf/library/items/A937KLQK?page=7&annotation=ZJ5BHUYM))