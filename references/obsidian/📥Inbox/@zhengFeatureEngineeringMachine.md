*title:* Feature Engineering for Machine Learning
*authors:* Alice Zheng, Amanda Casari
*year:* 2018
*tags:* feature-engineering
*status:* #📦 
*related:*
- [[@banachewiczKaggleBookData2022]]
- [[@butcherFeatureEngineeringSelection2020]]

# Notes 

# Annotations  
(20/12/2022, 08:43:13)

“Feature Scaling or Normalisation Some features, such as latitude or longitude, are bounded in value. Other numeric features, such as counts, may increase without bound. Models that are smooth functions of the input, such as linear regression, logistic regression, or anything that involves a matrix, are affected by the scale of the input. Tree-based models, on the Feature Scaling or Normalisation | 2” ([Zheng and Casari, p. 29](zotero://select/library/items/C4NI3DEH)) ([pdf](zotero://open-pdf/library/items/TBEXXQDV?page=45&annotation=RY245LNZ))

“other hand, couldn’t care less. If your model is sensitive to the scale of input features, feature scaling could help. As the name suggests, feature scaling changes the scale of the feature. Sometimes people also call it feature normalisation. Feature scaling is usually done individually to each feature. Next, we will discuss several types of common scaling operations, each resulting in a different distribution of feature values.” ([Zheng and Casari, p. 30](zotero://select/library/items/C4NI3DEH)) ([pdf](zotero://open-pdf/library/items/TBEXXQDV?page=46&annotation=EIEYLRWZ))

“Min-Max Scaling Let x be an individual feature value (i.e., a value of the feature in some data point), and min(x) and max(x), respectively, be the minimum and maximum values of this feature over the entire dataset. Min-max scaling squeezes (or stretches) all feature values to be within the range of [0, 1]. Figure 2-15 demonstrates this concept. The formula for min-max scaling is: x ̃ = x – min(x) max(x) – min(x)” ([Zheng and Casari, p. 30](zotero://select/library/items/C4NI3DEH)) ([pdf](zotero://open-pdf/library/items/TBEXXQDV?page=46&annotation=RD2BXIR7))

“Standardisation (Variance Scaling) Feature standardisation is defined as: x ̃ = x – mean(x) sqrt(var(x)) It subtracts off the mean of the feature (over all data points) and divides by the variance. Hence, it can also be called variance scaling. The resulting scaled feature has a mean of 0 and a variance of 1. If the original feature has a Gaussian distribution, then the scaled feature does too. Figure 2-16 is an illustration of standardisation.” ([Zheng and Casari, p. 31](zotero://select/library/items/C4NI3DEH)) ([pdf](zotero://open-pdf/library/items/TBEXXQDV?page=47&annotation=X2U6WTSN))