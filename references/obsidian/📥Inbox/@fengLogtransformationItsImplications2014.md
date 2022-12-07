*title:* Log-transformation and its implications for data analysis
*authors:* Changyong Feng, Hongyue Wang, Naiji Lu, Tian Chen, Hua He, Ying Lu, Xin M Tu
*year:* 2013
*tags:* #log-transform #feature-enginering #interpretability #skewness #log-normal 
*status:* #📦 
*related:*
*code:*
*review:*
- published in strange journal

## Notes 📍
- Read only if needed. Log-transform is likely not needed and thus paper not really relevant. Also paper contradicts the common belief.
- Overall the say that a $\log$ transform does not always reduce skewness.
- Also $\log$-transformation can hamper interpretability. They consider the regression case only.

## Annotations 📖

“Despite the common belief that the log transformation can decrease the variability of data and make data conform more closely to the normal distribution, this is usually not the case. Moreover, the results of standard statistical tests performed on log-transformed data are often not relevant for the original, non-transformed data” ([Feng et al., 2014, p. 105](zotero://select/library/items/Q6EF7PHI)) ([pdf](zotero://open-pdf/library/items/PLRJD4ET?page=2&annotation=HP4BDLBW))

“The log transformation is, arguably, the most popular among the different types of transformations used to transform skewed data to approximately conform to normality.” ([Feng et al., 2014, p. 106](zotero://select/library/items/Q6EF7PHI)) ([pdf](zotero://open-pdf/library/items/PLRJD4ET?page=3&annotation=GZIJQIJJ))

“In general, for right-skewed data, the logtransformation may make it either right- or left-skewed. If the original data does follow a log-normal distribution,” ([Feng et al., 2014, p. 106](zotero://select/library/items/Q6EF7PHI)) ([pdf](zotero://open-pdf/library/items/PLRJD4ET?page=3&annotation=8VH96UH2))

“the log-transformed data will follow or approximately follow the normal distribution. However, in general there is no guarantee that the log-transformation will reduce skewness and make the data a better approximation of the normal distribution.” ([Feng et al., 2014, p. 106](zotero://select/library/items/Q6EF7PHI)) ([pdf](zotero://open-pdf/library/items/PLRJD4ET?page=3&annotation=D3R5KBYP))

“Another popular use of the log transformation is to reduce the variability of data, especially in data sets that include outlying observations” ([Feng et al., 2014, p. 106](zotero://select/library/items/Q6EF7PHI)) ([pdf](zotero://open-pdf/library/items/PLRJD4ET?page=3&annotation=QNAMTGXM))

“Once the data is log-transformed, many statistical methods, including linear regression, can be applied to model the resulting transformed data. For example, the mean of the log-transformed observations (log yi), LT =(1/n)\*Σi=1 log yi is often used to estimate the population mean of the original data by applying the anti-log (i.e., exponential) function to obtain exp( LT). However, this inversion of the mean log value does not usually result in an appropriate estimate of the mean of the original data.” ([Feng et al., 2014, p. 107](zotero://select/library/items/Q6EF7PHI)) ([pdf](zotero://open-pdf/library/items/PLRJD4ET?page=4&annotation=UUC4QP8C))