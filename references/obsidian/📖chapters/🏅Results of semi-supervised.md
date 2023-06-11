We compare the performance of pre-trained Transformers and self-trained gradient-boosting on the gls-ise and gls-cboe test set. Results are reported in cref-tab-semi-supervised-results. 

![[results-semis-supervised.png]]

Identical to the supervised case, our models consistently outperform their respective benchmarks. Gradient boosting with self-training surpasses $\operatorname{gsu}_{\mathrm{small}}$ by percentage-3.35 on gls-ise and percentage-5.44 on gls-cboe in accuracy. Improvements for larger feature sets over $\operatorname{gsu}_{\mathrm{large}}$ are marginally lower to the supervised model and range between percentage-4.55 and percentage-7.44.

The results do not support the hypothesis, that incorporating unlabelled trades into the training corpus improves the performance of the classifier. We explore this finding in detail.

**Finding 5: Unlabelled Trades Provide Poor Guidance**
todo()

To summarize, despite the significantly higher training costs, semi-supervised variants do not provide better generalisation performance than supervised approaches. We subsequently evaluate if semi-supervised learning improves robustness, if not performance.



While the performance of semi-supervised classifiers is competitive to the of supervised

Despite the strong performance of our classifiers, semi-supervised methods do not deliver 


Overall, out-of-sample performance is lower than for the supervised variants. Th 

Accuracy is not the sole criterion. Depends on whether error is systematic or not. Thus, we do application study. See reasoning in \textcite{theissenTestAccuracyLee2000}


Use $t$-SNE to assess the output of the supervised vs. the semi-supervised train models. See [[@leePseudolabelSimpleEfficient]] and [[@banachewiczKaggleBookData2022]] for how to use it.
See [[@vandermaatenVisualizingDataUsing2008]] for original paper.
![[t-sne-map.png]]




- Results for random classifier
- What would happen if the classical rules weren't stacked?
- Confusion matrix
- ROC curve. See e. g., [this thread](https://stackoverflow.com/a/38467407) for drawing ROC curves

![[visualise-classical-rules-vs-ml.png]]
(print heatmap with $y$ axis with ask, bid and mid, $x$-axis could be some other criteria e. g. the trade size or none. If LR rule was good fit for options, accuracy should be evenly distributed and green. Visualise accuracy a hue / colour)
- calculate $z$-scores / $z$-statistic of classification accuracies to assess if the results are significant. (see e. g., [[@theissenTestAccuracyLee2000]])
- provide $p$-values. Compare twitter / linkedin posting of S. Raschka on deep learning paper.
- When ranking algorithms think about using the onesided Wilcoxon signed-rank test and the Friedman test. (see e. g. , code or practical application in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- Study removal of features with high degree of missing values with feature permutation. (see idea / code done in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- How do classical rules compare to a zero rule baseline? Zero rule baseline predicts majority class. (variant of the simple heuristic). The later uses simple heuristics to perform a heuristic. 
- Compare against "existing solutions" e. g., LR algorithm, depth rule etc.