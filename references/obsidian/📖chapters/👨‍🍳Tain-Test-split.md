eschews data leakage problems

Previous works 

The datasets (see cref ISE, CBOE) are split into three disjoint sets. The training set is used to fit our 

- What do previous works do? 

Parts of our trades are unlabelled, as .

Recall from earlier, that only for a subset of the dataset the labels are known.



Data may however be shuffled in the subsets.

Within the period, we filter out trades, where a true label can be inferred, to avoid duplicates with the supervised training set. For self-training this is essential, as labelled and unlabelled data is provided to the model simultaneously.

![[train-test-split.png]]


**Notes:**
[[👨‍🍳Train-Test-split notes]]