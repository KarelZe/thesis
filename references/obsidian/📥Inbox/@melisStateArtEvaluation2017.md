*title:* On the State of the Art of Evaluation in Neural Language Models
*authors:* Gábor Melis, Chris Dyer, Phil Blunsom
*year:* 2017
*tags:* 
*status:* #📥
*related:*

## Notes 📍


Validation loss vs. hyperparameters:
 ![[validation-loss-vs-hyperparam.png]]

## Annotations 📖

“nce hyperparameters have been properly controlled for, we find that LSTMs outperform the more recent models, contra the published claims. Our result is therefore a demonstration that replication failures can happen due to poorly controlled hyperparameter variation, and this paper joins other recent papers in warning of the under-acknowledged existence of replication failure in deep learning (Henderson et al., 2017; Reimers & Gurevych, 2017)” ([Melis et al., 2017, p. 1](zotero://select/library/items/AR7QEF3A)) ([pdf](zotero://open-pdf/library/items/HKGEY6YA?page=1&annotation=G8KD3KMD))

“However, we do show that careful controls are possible, albeit at considerable computational cost.” ([Melis et al., 2017, p. 1](zotero://select/library/items/AR7QEF3A)) ([pdf](zotero://open-pdf/library/items/HKGEY6YA?page=1&annotation=BF4EM8AC))

“Hyperparameters are optimised by Google Vizier (Golovin et al., 2017), a black-box hyperparameter tuner based on batched GP bandits using the expected improvement acquisition function (Desautels et al., 2014). Tuners of this nature are generally more efficient than grid search when the number of hyperparameters is small.” ([Melis et al., 2017, p. 3](zotero://select/library/items/AR7QEF3A)) ([pdf](zotero://open-pdf/library/items/HKGEY6YA?page=3&annotation=B7BGM2P4))

“On two of the three datasets, we improved previous results substantially by careful model specification and hyperparameter optimisation, but the improvement for RHNs is much smaller compared to that for LSTMs. While it cannot be ruled out that our particular setup somehow favours LSTMs, we believe it is more likely that this effect arises due to the original RHN experimental condition having been tuned more extensively (this is nearly unavoidable during model development).” ([Melis et al., 2017, p. 5](zotero://select/library/items/AR7QEF3A)) ([pdf](zotero://open-pdf/library/items/HKGEY6YA?page=5&annotation=5MYPMENZ))

“With a large number of hyperparameter combinations evaluated, the question of how much the tuner overfits arises. There are multiple sources of noise in play, (a) non-deterministic ordering of floating-point operations in optimised linear algebra routines, (b) different initialisation seeds, (c) the validation and test sets being finite samples from a infinite population. To assess the severity of these issues, we conducted the following experiment: models with the best hyperparameter settings for Penn Treebank and Wikitext-2 were retrained from scratch with various initialisation seeds and the validation and test scores were recorded. If during tuning, a model just got a lucky run due to a combination of (a) and (b), then retraining with the same hyperparameters but with different seeds would fail to reproduce the same good results.” ([Melis et al., 2017, p. 6](zotero://select/library/items/AR7QEF3A)) ([pdf](zotero://open-pdf/library/items/HKGEY6YA?page=6&annotation=NSS6DRVG))

“Third, the validation perplexities of the best checkpoints are about one standard deviation lower than the sample mean of the reruns, so the tuner could fit the noise only to a limited degree” ([Melis et al., 2017, p. 7](zotero://select/library/items/AR7QEF3A)) ([pdf](zotero://open-pdf/library/items/HKGEY6YA?page=7&annotation=NB464LTR))

“We have not explicitly dealt with the unknown uncertainty remaining in the Gaussian Process that may affect model comparisons, apart from running it until apparent convergence. All in all, our findings suggest that a gap in perplexity of 1.0 is a statistically robust difference between models trained in this way on these datasets. The distribution of results was approximately normal with roughly the same variance for all models, so we still report numbers in a tabular form instead of plotting the distribution of results, for example in a violin plot (Hintze & Nelson, 1998).” ([Melis et al., 2017, p. 7](zotero://select/library/items/AR7QEF3A)) ([pdf](zotero://open-pdf/library/items/HKGEY6YA?page=7&annotation=H6E92AIN))

“To further verify that the best hyperparameter setting found by the tuner is not a fluke, we plotted the validation loss against the hyperparameter settings. Fig. 2 shows one such typical plot, for a 4-layer LSTM. We manually restricted the ranges around the best hyperparameter values to around 15–25% of the entire tuneable range, and observed that the vast majority of settings in that neighbourhood produced perplexities within 3.0 of the best value.” ([Melis et al., 2017, p. 7](zotero://select/library/items/AR7QEF3A)) ([pdf](zotero://open-pdf/library/items/HKGEY6YA?page=7&annotation=J5ZBK4Z5))

“Still, we demonstrate how, with a huge amount of computation, noise levels of various origins can be carefully estimated and models meaningfully compared. This apparent tradeoff between the amount of computation and the reliability of results seems to lie at the heart of the matter. Solutions to the methodological challenges must therefore make model evaluation cheaper by, for instance, reducing the number of hyperparameters and the sensitivity of models to them, employing better hyperparameter optimisation strategies, or by defining “leagues” with predefined computational budgets for a single model representing different points on the tradeoff curve.” ([Melis et al., 2017, p. 8](zotero://select/library/items/AR7QEF3A)) ([pdf](zotero://open-pdf/library/items/HKGEY6YA?page=8&annotation=39BWY349))