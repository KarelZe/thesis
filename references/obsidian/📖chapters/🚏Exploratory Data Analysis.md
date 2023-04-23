
Get inspired by https://www.sciencedirect.com/science/article/pii/S2049080120305604


- high cardinality of underlyings
- insert stats from `2.0-mb-data_preprocessing_loading_splitting.ipynb`

- Adress leakage as part of eda. “Exploratory Data Analysis (EDA) can be a powerful tool for identifying leakage. EDA is the good practice of getting more intimate with the raw data, examining it through basic and interpretable visualization or statistical tools. Prejudice free and methodological, this kind of examination can expose leakage as patterns in the data that are surprising.” ([[@kaufmanLeakageDataMining2012]] p. 165)

- “On the very practical side, a good starting point for EDA is to look for any form of unexpected data properties. Common giveaways are found in identifiers, matching (or inconsistent matching) of identifiers (i.e., sample selection biases), surprises in distributions (spikes in densities of continuous values), and finally suspicious order in supposedly random data.” ([[@kaufmanLeakageDataMining2012]], p. 165)

- for analysis on CBOE data set see [[@easleyOptionVolumeStock1998]]. Could adapt their analysis of the trade times etc. for my own sample.

Why? Avoids data leakage: “Exploratory Data Analysis (EDA) can be a powerful tool for identifying leakage. EDA [Tukey 1977] is the good practice of getting more intimate with the raw data, examining it through basic and interpretable visualization or statistical tools. Prejudice free and methodological, this kind of examination can expose leakage as patterns in the data that are surprising.” (Kaufman et al., 2012, p. 165)
“On the very practical side, a good starting point for EDA is to look for any form of unexpected data properties. Common giveaways are found in identifiers, matching (or inconsistent matching) of identifiers (i.e., sample selection biases), surprises in distributions (spikes in densities of continuous values), and finally suspicious order in supposedly random data.” (Kaufman et al., 2012, p. 165)

- compare against [[@coxExploratoryDataAnalysis2017]]
- explain why we look just into the training set

- Involve domain experts: An example of the latter is using an opaque ML model to solve a problem where there is a strong need to understand how the model reaches an outcome, e.g. in making medical or financial decisions (see Rudin 2019). At the beginning of a project, domain experts can help you to understand the data, and point you towards features that are likely to be predictive. At the end of a project, they can help you to publish in domain-specific journals, and hence reach an audience that is most likely to benefit from your research. (from [[@lonesHowAvoidMachine2022]])
- To ignore previous studies is to potentially miss out on valuable information. For example, someone may have tried your proposed approach before and found fundamental reasons why it won’t work (and therefore saved you a few years of frustration), or they may have partially solved the problem in a way that you can build on. So, it’s important to do a literature review before you start work; leaving it too late may mean that you are left scrambling to explain why you are covering the) -> go back to related works and discuss them.

- Test if buys and sells are *really* imbalanced, as indicated by [[@easleyOptionVolumeStock1998]]. Might require up-or down-sampling.

- Describe interesting properties of the data set. How are values distributed?
- Examine the position of trade's prices relative to the quotes. This is of major importance in classical algorithms like LR, EMO or CLNV.
- Study if classes are imbalanced and require further treatmeant. The work of [[@grauerOptionTradeClassification2022]] suggests that classes are rather balanced.
- Study correlations between variables
- Remove highly correlated features as they also pose problems for feature importance calculation (e. g. feature permutation)
Perform EDA e. g., [AutoViML/AutoViz: Automatically Visualize any dataset, any size with a single line of code. Created by Ram Seshadri. Collaborators Welcome. Permission Granted upon Request. (github.com)](https://github.com/AutoViML/AutoViz) and [lmcinnes/umap: Uniform Manifold Approximation and Projection (github.com)](https://github.com/lmcinnes/umap)
- The approach of [[@grauerOptionTradeClassification2022]] matches the LiveVol data set, only if there is a matching volume on buyer or seller side. Results in 40 % reconstruction rate [[@grauerOptionTradeClassification2022]](p. 9). 
- In [[@easleyOptionVolumeStock1998]] CBOE options are more often actively bought than sold (53 %). Also, the number of trades at the midpoints is decreasing over time [[@easleyOptionVolumeStock1998]]. Thus the authors reason, that classification with quote data should be sufficient. Compare this with my sample!
- In adversarial validation it became obvious, that time plays a huge role. There are multiple options how to go from here:
	- Drop old data. Probably not the way to go. Would cause a few questions. Also it's hard to say, where to make the cut-off.
	- Dynamic retraining. Problematic i. e., in conjunction with pretrained models.
	- Use Weighting. Yes! Exponentially or linearily or date-based. Weights could be used in all models, as a feature or through penelization. CatBoost supports this through `Pool(weight=...)`. For PyTorch one could construct a weight tensor and used it when calculating the loss (https://stackoverflow.com/questions/66374709/adding-custom-weights-to-training-data-in-pytorch).

- Visualize behaviour over time e. g., appearing `ROOT`s and calculate statistics. How many of the clients / percentage are in the train set and how many are just in the test set?

- correlation binary vars, categorical vars https://www.kaggle.com/code/vmalyi/finding-the-most-correlating-variables/notebook

![[uuid_over_time.png]]
(found at https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284)

- See https://neptune.ai/blog/tabular-data-binary-classification-tips-and-tricks-from-5-kaggle-competitions for more ideas
- exploratory data analysis has been first introduced / coined by Tuckey. Found in [[@kuhnFeatureEngineeringSelection2020]]
- Cite [[@rubinInferenceMissingData1976]] for different patterns in missing data.
- on missingness see [[@kuhnFeatureEngineeringSelection2020]]


- Investigate skewness and outliers. See e. g., https://scientistcafe.com/ids/resolve-skewness.html and https://scientistcafe.com/ids/outliers.html.


“First, the percentage of trades going off at the midpoint of the spread is far lower in CBOE trades than in NYSE trades. Vijh offers the explanation that the market design of the CBOE—a competitive dealer system—might be the cause of this phenomenon as marketmakers offer their lowest quotes, and hence are not willing to bargain on transactions prices. An alternative explanation is that if informed trading occurs on the CBOE, and, if it is harder to detect given the multiplicity of dealers, then marketmakers protect themselves by trading at quoted prices more often” ([Easley et al., 1998, p. 454](zotero://select/library/items/593W67XA)) ([pdf](zotero://open-pdf/library/items/ZBEQIUNK?page=24&annotation=APMBYNEV))

“A second observation from Table II is that, over time, the percentage of trades executed at the spread midpoint shows a strong downward trend. This should make trade data more easily classifiable using quote data alone. Also, although studies of NYSE transactions report a roughly even split between buys and sells, it is clear that trades on the CBOE are increasingly buys. Hence, options are actively bought, rather than sold. This strengthens the argument against using transactions prices in studies of option marketstock market interactions, as these prices are more likely to be at the ask than at the bid and, hence, would bias upward the implied stock price.” ([Easley et al., 1998, p. 454](zotero://select/library/items/593W67XA)) ([pdf](zotero://open-pdf/library/items/ZBEQIUNK?page=24&annotation=2LW9T8MQ))

“Trading hours on the CBOE begin at 8:30 a.m. and end at 3:00 p.m., resulting in 78 5-minute intervals during each trading day.22 The option volume series reveal a distinct U-shaped intraday pattern. Both put and call option trading reach a peak about 45 minutes after the opening. The peak volume is almost 2 percent of daily traded volume for calls and about 2.75 percent of daily volume for puts. The volume falls to less than 0.5 percent of the daily volume by noon. It rises again in the afternoon and, although the level of the afternoon peak is not as high as the morning one for puts, it is marginally higher than the morning peak for calls. Trading volume then quickly falls off toward the close. These volume patterns have two interesting implications. First, because early morning and late afternoon are the periods of high volume, such periods may include more informed trades as it is easier for informed traders to "hide" in the volume at that time (see Admati and Pfleiderer (1988) and Foster and Viswanathan (1990) for discussion of such behavior). Second, because peak option volume lags peak stock volume (which occurs at the opening) by about 45 minutes, the multiple regression results of our study must take this into account.” (Easley et al., 1998, p. 455)

“As the probability of observing only buy trades or only sell trades decreases with an increasing number of trades, the number of trades per option day is lower and the time between two trades is higher in our matched samples compared to their full sample equivalents. Because tick tests depend on the information from preceding or succeeding trades as a precise signal for the fair option price, our results might therefore underestimate their performance.” ([[@grauerOptionTradeClassification2022]]., 2022, p. 9)
