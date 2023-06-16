**Motivation:**
Every option trade has a buyer and seller side. For a plethora of problems in option research, it’s also crucial to determine the party that initiated the transaction.  Common applications include the study of option demand ([[@garleanuDemandBasedOptionPricing2009]]3), the informational content of option trading ([[@huDoesOptionTrading2014]]631) and ([[@panInformationOptionVolume2006]]882) or todo-(pin...) or todo-(effective spread...). 

For example, Mayhew, Sarin, and Shastri (1995) find evidence that informed traders migrate between stock and option markets in response to changes in the option margin requirement. Easley, O’Hara, and Srinivas (1998) and Pan and Poteshman (2003) find that signed trading volume in the option market can help forecast stock returns. Cao, Chen, and Griffin (2000) and others document abnormal trading volume in the options market prior to takeover announcements.

Despite the evident importance for empirical research, the true initiator of the trade is frequently missing in option data sets and must be inferred using trade classification algorithms ([[@easleyOptionVolumeStock1998]]453). In consequence, the correctness of empirical studies hinges with the algorithm's ability to correctly identify the trade initiator.  

Among the most prevailing variants to sign trades are the tick rule ([[@hasbrouckTradesQuotesInventories1988]]240), quote rule ([[@harrisDayEndTransactionPrice1989]]41), and hybrids thereof such as the gls-LR algorithm ([[@LeeInferringTradeDirection1991]]745), the gls-EMO algorithm ([[@ellisAccuracyTradeClassification2000]]536), and the gls-CLNV method ([[@chakrabartyTradeClassificationAlgorithms2007]]3809), that infer the trade initiator from adjacent prices and quotes. These algorithms have initially been proposed and tested for the stock market. 

For option markets, the works of ([[@grauerOptionTradeClassification2022]]10--13) and ([[@savickasInferringDirectionOption2003]]887) raise concerns about the transferability of standard trade signing rules due to deteriorating classification accuracies and systematic miss-classification of trades.  todo-Trade classification in option markets is a particularly difficult testing ground due to illiquidity / trading at different venues / https://www.sec.gov/news/studies/ordpay.htm...)
- https://onlinelibrary.wiley.com/doi/pdf/10.1111/1540-6261.00447
- https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1540-6261.2004.00661.x

The recent work of ([[@grauerOptionTradeClassification2022]]13--16) partly alleviates the concern by proposing explicit overrides for trade types and by combining multiple heuristics into deep-stacked rules, advancing the state-of-the-art performance in option trade classification. By this means, their approach enforces a more sophisticated decision boundary eventually leading to a more accurate classification. Beyond heuristics, however, it remains an open research problem in option markets, if classifiers *learned* on trade data can improve upon static classification rules with respect to performance.

In this thesis, we focus on state-of-the-art machine learning methods to infer the trade initiator. Approaching trade classification with machine learning is a logical choice, given its capability to handle high-dimensional trade data and learn complex decision boundaries. Against this backdrop, the question is, can an alternative machine learning-based classifier improve upon standard trade classification rules?

**Contributions:**


**Outline:**
The remainder of this paper is organized as follows. Cref-[[👪Related Work]] reviews publications on trade classification in option markets and using machine learning, thereby underpinning our research framework. Cref-[[🍪Selection Of Supervised Approaches]] discusses and introduces supervised methods for trade classification. Then, cref- [[🍪Selection Of Semisupervised Approaches]] extends the previously selected algorithms for the semi-supervised case. We test the models in cref-[[🌏Dataset]] in an empirical setting. In cref-[[🍕Application study]]  we apply our models to the problem of effective spread estimation. Finally, cref-[[🧓Discussion]] concludes.

The remainder of this article is organized as follows. In Section I, we review
some of the theoretical and empirical literature on the informational role of
option markets. In Section II, we summarize the Hasbrouck (1995) method and
describe the modifications necessary to apply it to the options market. Our data
sources are described in Section III. Section IV presents our main results on
price discovery in the stock and ATM call options. In Section V, we extend the
analysis to OTM and ITM options, and seek to explain cross-sectional variation
in the relative information share measures of ATM and OTM options. In Section
VI, we report some additional robustness tests. Section VII summarizes our
results and contains suggestions for future research.

**Notes:**
[[👶introduction notes]]

You should go with a so called “Feynman's method”.  
He described in one of his interviews, that he would read scientific papers in a following fashion:

- Read the abstract.
- Try to predict what are the results and conclusions of the paper.
- Go ahead to “Conclusions” section — check whether your prediction stands.
- If your prediction was successful don't bother reading the rest of the paper, go to the next one.
- If the outcome surprised you, then read the whole paper and carefully study the methods.
- Profit!










