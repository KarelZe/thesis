
Use $t$-SNE to assess the output of the supervised vs. the semi-supervised train models. See [[@leePseudolabelSimpleEfficient 1]] and [[@banachewiczKaggleBookData2022]] for how to use it.
See [[@vandermaatenVisualizingDataUsing2008]] for original paper.
![[t-sne-map 1.png]]




- Results for random classifier
- What would happen if the classical rules weren't stacked?
- Confusion matrix
- ROC curve. See e. g., [this thread](https://stackoverflow.com/a/38467407) for drawing ROC curves

![[visualize-classical-rules-vs-ml 1.png]]
(print heatmap with $y$ axis with ask, bid and mid, $x$-axis could be some other criteria e. g. the trade size or none. If LR rule was good fit for options, accuracy should be evenly distributed and green. Visualize accuracy a hue / color)
- calculate $z$-scores / $z$-statistic of classification accuracies to assess if the results are significant. (see e. g., [[@theissenTestAccuracyLee2000]])
- provide $p$-values. Compare twitter / linkedin posting of S. Raschka on deep learning paper.
- When ranking algorithms think about using the onesided Wilcoxon signed-rank test and the Friedman test. (see e. g. , code or practical application in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- Study removal of features with high degree of missing values with feature permutation. (see idea / code done in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- How do classical rules compare to a zero rule baseline? Zero rule baseline predicts majority class. (variant of the simple heuristic). The later uses simple heuristics to perform a heuristic. 
- Compare against "existing solutions" e. g., LR algorithm, depth rule etc.