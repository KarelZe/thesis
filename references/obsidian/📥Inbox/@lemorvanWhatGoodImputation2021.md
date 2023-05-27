*title:* What’s a good imputation to predict with missing values?
*authors:* Marine Le Morvan, Julie Josse, Erwan Scornet, Gael Varoquaux
*year:* 2020
*tags:* #imputation #gbm #bayes-optimal #missing-value #regression 
*status:* #📦 
*related:*
- [[@rubinInferenceMissingData1976]] (the classification is used in the paper)
*code:*
- None
## Notes 📍

- Commonly impute-then-regress (apply imputation first and then perform regression) is  used to train regression models on data with missing values. This precedure lags a theoretical foundation. Imputation is commonly done using "off-the-shelf" methods.
- Authors can show that for almost all imputation approaches this two step procedure is Bayses optimal for all missing data mechanisms e. g., missing at random etc.. They only consider the regression setting only. Not sure if it would transfer to classification. Results depend on the powerfulness of the predictor.
- One of the few supervised-learning models that support learning on partially-observed data, are tree-based approaches (see [[@twalaGoodMethodsCoping2008]] (*Missing Incorporated Attribute strategy*) and [[@chenXGBoostScalableTree2016]].

## Annotations 📖

“Yet, this widespread practise has no theoretical grounding. Here we show that for almost all imputation functions, an impute-then-regress procedure with a powerful learner is Bayes optimal.” ([Le Morvan et al., 2021, p. 1](zotero://select/library/items/U5TTSA2S)) ([pdf](zotero://open-pdf/library/items/XAHMRU4X?page=1&annotation=DN4MMQYN))

“regression function will generally be discontinuous, which makes it hard to learn. Crafting instead the imputation so as to leave the regression function unchanged simply shifts the problem to learning discontinuous imputations.” ([Le Morvan et al., 2021, p. 1](zotero://select/library/items/U5TTSA2S)) ([pdf](zotero://open-pdf/library/items/XAHMRU4X?page=1&annotation=PCER8YFT))

“en simple data-generating mechanisms lead to complex decision rules. To date, there are few supervised-learning models natively suited for partially-observed data. A notable 35th Conference on Neural Information Processing Systems (NeurIPS 2021)” ([Le Morvan et al., 2021, p. 1](zotero://select/library/items/U5TTSA2S)) ([pdf](zotero://open-pdf/library/items/XAHMRU4X?page=1&annotation=ZVHKCFYD))

“exception is found with tree-based models [[@twalaGoodMethodsCoping2008]], widely used in data-science practise.” ([Le Morvan et al., 2021, p. 2](zotero://select/library/items/U5TTSA2S)) ([pdf](zotero://open-pdf/library/items/XAHMRU4X?page=2&annotation=YKLJHH8C))

“The most common practise however remains by far to use off-the-shelf methods first for imputation of missing values and second for supervised-learning on the resulting completed data.” ([Le Morvan et al., 2021, p. 2](zotero://select/library/items/U5TTSA2S)) ([pdf](zotero://open-pdf/library/items/XAHMRU4X?page=2&annotation=EQNKWHAD))

“We contribute a systematic analysis of Impute-the-Regress procedures in a general setting: non-linear response function and any missingness mechanism (no MAR assumptions). We show that:” ([Le Morvan et al., 2021, p. 2](zotero://select/library/items/U5TTSA2S)) ([pdf](zotero://open-pdf/library/items/XAHMRU4X?page=2&annotation=QL7X9VCQ))

“Impute-then-Regress procedures are Bayes optimal for all missing data mechanisms and for almost all imputation functions, whatever the number of variables that may be missing. This very general result gives theoretical grounding to such widespread procedures.” ([Le Morvan et al., 2021, p. 2](zotero://select/library/items/U5TTSA2S)) ([pdf](zotero://open-pdf/library/items/XAHMRU4X?page=2&annotation=RX34ADUM))

“GBRT: Gradient boosted regression trees (Scikit-learn’s HistGradientBoostingRegressor with default parameters). This predictor readily supports missing values: during training, missing values on the decision variable for a given split are sent to the left or right child depending on which provides the largest gain. This is know as the Missing Incorporated Attribute strategy [Twala et al., 2008]” ([Le Morvan et al., 2021, p. 9](zotero://select/library/items/U5TTSA2S)) ([pdf](zotero://open-pdf/library/items/XAHMRU4X?page=9&annotation=UXYKI88D))

“Impute-then-regress procedures assemble standard statistical routines to build predictors suited for data with missing values. However, we have shown that seeking the best prediction of the outcome leads to different tradeoffs compared to inferential purposes. Given a powerful learner, almost all imputations lead asymptotically to the optimal prediction, whatever the missingness mechanism.” ([Le Morvan et al., 2021, p. 10](zotero://select/library/items/U5TTSA2S)) ([pdf](zotero://open-pdf/library/items/XAHMRU4X?page=10&annotation=749VILY9))