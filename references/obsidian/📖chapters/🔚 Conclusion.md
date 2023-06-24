### Outlook
- stronger focus on pre-text task 
- 
Furthermore, the hedge rebalancing is estimated to alter the probability of daily absolute stock returns 
Graphically, our results show that specific attention heads in the Transformer specialise in patterns akin to classical trade classification rules. We are excited to explore this aspect systematically and potentially reverse engineer classification rules
from attention heads that are yet unknown. This way, we can transfer the superior classification accuracy of the Transformer to regimes where labelled training data is abundant or computational costs of training are not affordable.

### Conclusion

The contextassociative power of language models likely confers significant advantages over 

Generalises well
relative to two approaches (...) relative to (...)

We also, find evidence consistent with ([[@grauerOptionTradeClassification2022]]) and ([[@savickasInferringDirectionOption2003]]), that common trade classification rules perform poorly on option data.
, insights into model uncertainty, 

feature tick like features, ml is bound by features

Our models deliver accurate predictions and improved robustness, which effectively reduce noise and bias in option's research reliant on good estimates for the trade initiator. When applied to the calculation of trading cost through effective spreads, the models dominate all rule-based approaches by approximating the true effective spread best. Concretely, the Transformer pre-trained on unlabelled trades estimates a mean spread of  \SI[round-precision=3]{0.013}[\$]{} versus \SI[round-precision=3]{0.005}[\$]{} actual spread at the gls-ISE.

![[Pasted image 20230624180316.png]]

In conclusion, our work demonstrates that machine learning is superior to existing trade signing algorithms for classifying option trades, if partially-labelled or labelled trades are available for training. 

----



Other desirable properties are low cost at inference and uncertainty estimates through a probabilistic problem framing.

Unlike previous study, we can trace back the performance of our approaches as well as of classical trade classification rules to individual features and feature groups. Particulari

We illustrate the usefulness of our models by applying the trade direction suggested by each algorithm to compute effective spreads and price impacts of trades. Our algorithm provides better and unbiased estimates of both the actual effective spreads and price impacts compared to the other classification rules.

This study presents a comprehensive comparison of common trade classification algorithms
is considerably lower than for stocks


The goal of this study is to examine the performance (...) and determine (...).

In conclusion, we propose a (...) to (...).
In this paper, we present (...) that (...), and competitive with state-of-the-art foundation models. Most notably, 
Our approach achieves performance on par with SOTA
The method we introduced satisfies numerous desirable properties

correctly classifying 85 %
In particular, transactions inside the bid}ask spread, small transactions, and transactions in large or frequently traded stocks are often misclassifed.
Furthermore, the hedge rebalancing is estimated to alter the probability of daily absolute stock returns 

Thanks to (...), we can 
Unlike previous studies, we show that it is possible to achieve
Additionally, we observed like
Finally, we plan to (...) trained on larger pretraining corpora in the future, since we have seen a constant improvement in performance as we were scaling.
For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles
We leverage (...) to Model (...) and build efficent (...). 
We hope that releasing these models to the research community will accelerate the development of large language models, and help efforts to improve their robustness and mitigate known issues such as toxicity and bias.

Our perspective of quantifying predictive power shows that numerous existing methods
We provide evidence of a significant negative relation between
We also, find evidence consistent with return reversals
All of these results are consistent

We also demonstrated that language models can compose a
We fit models to a variety of algorithms and determined that a
 Our analysis highlights
These results generally carry over to the equity market and generally reconcile with previous findings. 
 The overall success rates of the LR, EMO, and tick rules are 74.42%, 75.80%, and 75.40%, respectively. 







We show that options prices are predictable at high frequency, and a large fraction of options traders exploit this predictability in timing their executions. The expected future quote midpoint computed from our best predictive model is a reasonable estimate of the option fair value. Traders who time executions when taking liquidity buy when this estimate of the fair value (the expected future midpoint) is close to but less than the quoted ask and sell when the estimate of the fair value is close to but greater than the quoted bid. Traders who exploit this predictability are able to take liquidity at low costs. Measuring the cost of taking liquidity as the difference between the trade price and the estimate of option fair value at the time of the trade, adjusted for trade direction, the effective spread of the traders who time executions is on average 37.4% of the magnitude of the conventional estimate of the effective spread that uses the midpoint as a proxy for the fair value and 29.6% of the quoted spread. The overall average adjusted effective spread, averaging over the traders who do and do not time executions, is on average less than three-quarters as large as the conventional estimate and less than 60% of the quoted spread. These estimates of the costs of taking liquidity in the options markets help resolve the puzzle of why options trading volume is so high despite the seemingly high costs of taking liquidity: for traders who have access to execution algorithms that enable execution timing, the costs of taking liquidity are much lower than conventional estimates. The high-frequency predictability of options prices also affects estimates of the price impact of options trades. Conventional measures of price impact that use the midpoint as a proxy for the fair value average about 1.5% of the option price for options on S&P 500 stocks. Taking account of the predictability of options prices by using the predicted future option price as the estimate of the option fair value, the adjusted estimates of price impact are less than half as large as the conventional estimates. Finally, the reduction in trading costs due to execution timing affects our conclusions about the net of trading cost profitability of options trading strategies. Two options strategies that sell volatility by writing at-the-money straddles and holding them until expiration produce high returns if transactions costs are disregarded. These strategies remain highly profitable, though of course less profitable, for traders who time executions and pay the algo effective half-spread. However, these strategies yield statistically insignificant returns for traders who pay the conventional effective half-spread. (Murajev)


