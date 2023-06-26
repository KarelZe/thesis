We compare the performance of pre-trained Transformers and self-trained gradient-boosting on the gls-ise and gls-cboe test set. Results are reported in cref-tab-semi-supervised-results. 

Identical to the supervised case, our models consistently outperform their respective benchmarks. Gradient boosting with self-training surpasses $\operatorname{gsu}_{\mathrm{small}}$ by percentage-3.35 on gls-ise and percentage-5.44 on gls-cboe in accuracy. Improvements for larger feature sets over $\operatorname{gsu}_{\mathrm{large}}$ are marginally lower to the supervised model and range between percentage-4.55 and percentage-7.44.

The results do not support the hypothesis, that incorporating unlabelled trades into the training corpus improves the performance of the classifier. We explore this finding in detail.

**Finding 5: Unlabelled Trades Provide Poor Guidance**
todo()

To summarize, despite the significantly higher training costs, semi-supervised variants do not provide better generalisation performance than supervised approaches. We subsequently evaluate if semi-supervised learning improves robustness, if not performance.