
The goal of this study is to examine the performance of machine learning-based trade classification in the option market. In particular, we propose to model trade classification with Transformers and gradient boosting. Both approaches are supervised and suffice to learn on labelled trades. For settings, where labelled trades are scarce, we extend Transformers with a pre-training objective to learn on unlabelled trades as well as generate pseudo-labels for gradient-boosting through a self-training procedure.

Our models establish a new state-of-the-art for trade classification on gls-ISE and gls-CBOE, achieving 
Performance generalises well Most notably,
Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.
Notably, performance is even stronger 

Relative to the ubiquitous tick test, quote rule, and LR algorithm, improvements are percentage-23.88, percentage-17.11, and percentage-17.02, respectively on the gls-ISE dataset without additional data requirements. Performance improvements are particularly strong out-of-the-money options, options with late maturity, as well as trades executed at the quotes.

Considering, the semi-supervised setting, Transformers on gls-ISE dataset profit from pre-training on unlabelled trades with accuracies up to percentage-74.55, but the performance gains slightly diminish on the gls-CBOE test set. Vice versa, we observe no advantage with regard to performance or robustness from semi-supervised training of glspl-GBRT.

Consistent with ([[@grauerOptionTradeClassification2022]]27) and ([[@savickasInferringDirectionOption2003]]901) we find evidence that the performance of common trade classification rules deteriorates in the option market. In particular, tick-base methods marginally outperform a random guess.

Unlike previous studies, we can trace back the performance of our approaches as well as of trade classification rules to individual features and feature groups using the importance measure gls-SAGE. We find that both approaches attain largest performance improvements from classifying trades based on quoted sizes and prices, but machine learning-based classifiers attain higher performance gains and effectively exploit the data. The change in the trade price, decisive criteria to the (reverse) tick test, plays no rule for option trade classification. We identify the relative illiquidity of options to hamper the information content of the surrounding trade prices. Our classifiers profit from the inclusion of option-specific features, like moneyness and  time-to-maturity, unexploited in classical trade classification. 

By probing and visualising the attention mechanism inside the Transformer, we can establish connection to rule-based classification. Experimentally, our results show, that attention heads encode knowledge about rule-based classification. Whilst attention heads in earlier layers of the network broadly attend to all features, in later they focus on specific features jointly used in rule-based classification akin to the gls-LR algorithm, depth rule or others.  Furthermore embeddings encode knowledge about the underlyings. Our results show, that the Transformer learns to group similar underlyings in embedding space.

Our models deliver accurate predictions and improved robustness, which effectively reduce noise and bias in option's research reliant on good estimates for the trade initiator. When applied to the calculation of trading cost through effective spreads, the models dominate all rule-based approaches by approximating the true effective spread best. Concretely, the Transformer pre-trained on unlabelled trades estimates a mean spread of  \SI[round-precision=3]{0.013}[\$]{} versus \SI[round-precision=3]{0.005}[\$]{} actual spread at the gls-ISE.
(feature importances)

In conclusion, our work demonstrates that machine learning is superior to existing trade signing algorithms for classifying option trades, if partially-labelled or labelled trades are available for training. 

While we tested our models on option trades, we expect that similar results are possible for other modalities including equity trades. 

