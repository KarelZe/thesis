*title:* Advances in financial machine learning
*authors:* Marcos {López de Prado}
*year:* 2017
*tags:* #cross-validation #train-test-split 
*status:* #📦 
*related:*
- [[@hastietrevorElementsStatisticalLearning2009]]
- [[@nagelMachineLearningAsset2021]]
*code:*
- in the book.
*review:*

## Notes 📍

## Annotations 📖

“CV is yet another instance where standard ML techniques fail when applied to financial problems. Overfitting will take place, and CV will not be able to detect it. In fact, CV will contribute to overfitting through hyper-parameter tuning” ([{López de Prado}, 2018, p. 103](zotero://select/library/items/UKQHETWP)) ([pdf](zotero://open-pdf/library/items/WCD8RZP8?page=131&annotation=HSNX8V7C))

“CV splits observations drawn from an IID process into two sets: the training set and the testing set. Each observation in the complete dataset belongs to one, and only one, set. This is done as to prevent leakage from one set into the other, since that would defeat the purpose of testing on unseen data. Further details can be found in the books and articles listed in the references section.” ([{López de Prado}, 2018, p. 103](zotero://select/library/items/UKQHETWP)) ([pdf](zotero://open-pdf/library/items/WCD8RZP8?page=131&annotation=LEEP94KV))

“By now you may have read quite a few papers in finance that present k-fold CV evidence that an ML algorithm performs well. Unfortunately, it is almost certain that those results are wrong. One reason k-fold CV fails in finance is because observations cannot be assumed to be drawn from an IID process. A second reason for CV’s failure is that the testing set is used multiple times in the process of developing a model, leading to multiple testing and selection bias. We will revisit this second cause of failure in Chapters 11–13. For the time being, let us concern ourselves exclusively with the first cause of failure. Leakage takes place when the training set contains information that also appears in the testing set. Consider a serially correlated feature X that is associated with labels Y that are formed on overlapping data” ([{López de Prado}, 2018, p. 104](zotero://select/library/items/UKQHETWP)) ([pdf](zotero://open-pdf/library/items/WCD8RZP8?page=132&annotation=PXPFDZA2))

“JWBT2318-c07 JWBT2318-Marcos February 13, 2018 14:59 Printer Name: Trim: 6in × 9in A SOLUTION: PURGED K-FOLD CV 105 Because of the serial correlation, Xt ≈ Xt+1. Because labels are derived from overlapping datapoints, Yt ≈ Yt+1” ([{López de Prado}, 2018, p. 105](zotero://select/library/items/UKQHETWP)) ([pdf](zotero://open-pdf/library/items/WCD8RZP8?page=133&annotation=7K3V6N7H))

“By placing t and t+1 in different sets, information is leaked. When a classifier is first trained on (Xt, Yt), and then it is asked to predict E[Yt+1|Xt+1] based on an observed Xt+1, this classifier is more likely to achieve Yt+1 = E[Yt+1|Xt+1]evenifX is an irrelevant feature.” ([{López de Prado}, 2018, p. 105](zotero://select/library/items/UKQHETWP)) ([pdf](zotero://open-pdf/library/items/WCD8RZP8?page=133&annotation=SGU8WKYS))

“If X is a predictive feature, leakage will enhance the performance of an already valuable strategy. The problem is leakage in the presence of irrelevant features, as this leads to false discoveries. There are at least two ways to reduce the likelihood of leakage: 1. Drop from the training set any observation i where Yi is a function of information used to determine Yj, and j belongs to the testing set. (a) For example, Yi and Yj should not span overlapping periods (see Chapter 4 for a discussion of sample uniqueness). 2. Avoid overfitting the classifier. In this way, even if some leakage occurs, the classifier will not be able to profit from it. Use: (a) Early stopping of the base estimators (see Chapter 6). (b) Bagging of classifiers, while controlling for oversampling on redundant examples, so that the individual classifiers are as diverse as possible. i. Set max_samples to the average uniqueness. ii. Apply sequential bootstrap (Chapter 4).” ([{López de Prado}, 2018, p. 105](zotero://select/library/items/UKQHETWP)) ([pdf](zotero://open-pdf/library/items/WCD8RZP8?page=133&annotation=ALRQDCKD))