
![[summary_statistics.png]]


- Describe interesting properties of the data set. How are values distributed?
- Examine the position of trade's prices relative to the quotes. This is of major importance in classical algorithms like LR, EMO or CLNV.
- Study if classes are imbalanced and require further treatmeant. The work of [[@grauerOptionTradeClassification2022]] suggests that classes are rather balanced.
- Study correlations between variables
- Remove highly correlated features as they also pose problems for feature importance calculation (e. g. feature permutation)
- Plot KDE plot of tick test, quote test...
![[kde-tick-rule 1.png]]
Perform EDA e. g., [AutoViML/AutoViz: Automatically Visualize any dataset, any size with a single line of code. Created by Ram Seshadri. Collaborators Welcome. Permission Granted upon Request. (github.com)](https://github.com/AutoViML/AutoViz) and [lmcinnes/umap: Uniform Manifold Approximation and Projection (github.com)](https://github.com/lmcinnes/umap)
- The approach of [[@grauerOptionTradeClassification2022]] matches the LiveVol data set, only if there is a matching volume on buyer or seller side. Results in 40 % reconstruction rate [[@grauerOptionTradeClassification2022]](p. 9). 
- In [[@easleyOptionVolumeStock1998]] CBOE options are more often actively bought than sold (53 %). Also, the number of trades at the midpoints is decreasing over time [[@easleyOptionVolumeStock1998]]. Thus the authors reason, that classification with quote data should be sufficient. Compare this with my sample!
- In adversarial validation it became obvious, that time plays a huge role. There are multiple options how to go from here:
	- Drop old data. Probably not the way to go. Would cause a few questions. Also it's hard to say, where to make the cut-off.
	- Dynamic retraining. Problematic i. e., in conjunction with pretrained models.
	- Use Weighting. Yes! Exponentially or linearily or date-based. Weights could be used in all models, as a feature or through penelization. CatBoost supports this through `Pool(weight=...)`. For PyTorch one could construct a weight tensor and used it when calculating the loss (https://stackoverflow.com/questions/66374709/adding-custom-weights-to-training-data-in-pytorch).

- Visualize behaviour over time e. g., appearing `ROOT`s and calculate statistics. How many of the clients / percentage are in the train set and how many are just in the test set?
![[uuid_over_time.png]]
(found at https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284)

- See https://neptune.ai/blog/tabular-data-binary-classification-tips-and-tricks-from-5-kaggle-competitions for more ideas
- exploratory data analysis has been first introduced / coined by Tuckey. Found in [[@kuhnFeatureEngineeringSelection2020]]
- Cite [[@rubinInferenceMissingData1976]] for different patterns in missing data.
- on missingness see [[@kuhnFeatureEngineeringSelection2020]]
- ![[visualization-of-missingness.png]]