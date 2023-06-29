
Visually, the performance differences between gradient boosting and transformers on the same feature sets are minor, which is in line with previous studies ([[@grinsztajnWhyTreebasedModels2022]], [[@gorishniyRevisitingDeepLearning2021]]) These studies conclude, generally for tabular modelling, that neither Transformers nor GBRTs are universally superior. Our results validate this observation, specifically for trade classification.

However, our findings contradict those of ([[@ronenMachineLearningTrade2022]]14--49), who benchmark tree-based ensembles in the form of gls-RF and neural networks in the form of gls-FFN for trade classification in the equity and bond market and find clear dominance of the tree-based approach. Beyond differences in the market under study and variants, two methodological differences are evident, that explain the diverging results. First, unlike gls-FFN, the FT-Transformer is tailored to learn on tabular data through being a rotationally-invariant learner. Second, the data pre-processing and feature engineering is tailored to the requirements of neural networks. Without these measures, tree-based approaches excel due to their robustness in handling skewed and missing data.


From our experiments we observed, th






So far it remain


Figure 5.11 illustrates that the AUC-score of outlier detection can increase in the static setting on high-dimensional data with GMD preprocessing. This confirms the results from [Keller et al., 2012, Trittenbach and Böhm, 2019]. The experiments show that outlier detection profits from subspace search in the streaming setting, too:


**Distribution in Sample: TTM, Trade Size, Moneyness** 
- deep in the money-options most frequently traded / liquidity


**Accuracy of Basic Rules Is Downward-Biased by Coverage** **Accuracy of 

Trade Size Rule Is High And Highly Variant** **Improvements Relative To Literature** 

How would a linear model do? 

**No Universal Superior Classifier**



**Supervised Classifier Performance on CBOE**

**Link Between Unlabelled Trades And Generalisation Performance** 
Why is pre-training successful? Why is self-training not successful? 

**Robustness of Index Options / Options Outside The Quotes**
**Importance of Moneyness and Time-to-Maturity** 
**The Elephant In The Room** 
Requires labelled data. Compute intensive.


- low accuracy for trades outside the quotes
	- see also [[@ellisAccuracyTradeClassification2000]] for trades inside and outside the spread
	- “On the one hand, we would expect that the greater (smaller) the transaction price relative to the midspread, the more likely that the transaction is a buy (sell) and occurs on an uptick (a downtick), implying higher classification success for outside-quote trades, especially for large trades in which the trade initiator is willing to pay a premium for the execution of his large order.” ([[@savickasInferringDirectionOption2003]] p. 888)
	- “On the other hand, however, the outside-quote trades may be the manifestation of stale quotes, which result in misclassification. Also, the effect of market makers’ hedging and rebalancing trades on the classification of outside-quote trades is unclear. Section IV.C contains a logit analysis of outside-quote trades.” ([[@savickasInferringDirectionOption2003]], p. 888)
- high gains for options for otm options and options with long maturity
	- Accuracy is not the sole criterion. Depends on whether error is systematic or not. Thus, we do application study. See reasoning in ([[@theissenTestAccuracyLee2000]])
	- “Specifically, one of the most noticeable regularities is that smaller trades are classified more precisely. This is because these trades are more likely to be executed at quotes and are less prone to reversed-quote trading (partially due to the fact that many small trades are executed on RAES)” (Savickas and Wilson, 2003, p. 889)
	- Moneyness levels are “Out-of-the-money options offer the highest leverage (exposure for a dollar invested) and thus are particularly attractive for informed investors. Consistent with this argument, the information price impact is decreasing and convex in absolute delta. Figure 3(D) shows that the impact decreases from 0.4% for out-of-the-money options to 0.15% for in-the-money options. Next, private information is often short-lived and is related to near-term events, and thus short-term options are better suited for informed investors in addition to providing higher leverage. Indeed, the price impact decreases by 0.12% if time-to-expiration decreases from 80 days to 20 days. Buyer-initiated trades have a higher price impact than sell trades, because these trades provide an opportunity to bet not only on future volatility but also on the underlying direction. These results are broadly consistent with Pan and Poteshman (2006), except that I do not find a significant difference between call and put options, perhaps because my sample consists of large stocks that are easy to sell short.” (Muravyev, 2016, p. 695)
“Since time to maturity is inversely related to trade size, we observe greater classification errors for shorter maturity options.” (Savickas and Wilson, 2003, p. 889)
- performance gap in classical rules
- strong performance of neural networks / tree-based ensembles
	- We identify missingess in data to be down-ward biasing the results of classical estimators. ML predictors are robust to this missingness, as they can handle missing values and potentially substitute.
- methodology
	- our study puts special emphasises on thoughtful tuning, data pre-processing.
- the elephant in the room: 
	- labelled data and cmputational data. 
	- Finetune. Low cost of inference
- which algorithm is no preferable? Do Friedman rank test


## Trade size
![[tsize-my-results.png]]
(Similar to [[@ellisAccuracyTradeClassification2000]]537)

## Moneyness / Trade Size

![[moneyness-vs-trade-size.png]]
(test set)

## Time-to-Maturity
![[time-to-maturity-tsize-result.png]]
(test set)

## Index Options
- only few index options in sample

![[index-options-results.png]]
(test set)

## Proximity To Quotes
![[proximity-quotes-results.png]]
(all sets)

- moneyness / time-to-maturity / how do both relate with trade classification / motives
- low accuracy for index options
	- Study sources of missclassification. See e. g., [[@savickasInferringDirectionOption2003]]
	- The extent to which inaccurate trade classification biases empirical research dependes on whether misclassifications occur randomly or systematically [[@theissenTestAccuracyLee2000]]. This document also contains ideas how to study the impact of wrong classifications in stock markets. Might different in option markets.
	- “Spreads are portfolios of options of the same type (either only calls or only puts). Combinations are portfolios of options of different types. Traders can form these complex trades by individually buying the component options or by trading standard packages. The advantage of the latter approach is that the trader is subject to only one bid-ask spread, while buying the component options individually results in paying the bid-ask spread for each option. The market maker determines how to allocate the bid-ask spread among all options in a complex trade. Thus, not all (if any) of the component options necessarily trade at their quotes. Therefore, complex trades are highly likely to produce RQ and outside-quote trades. Furthermore, labeling complex trades as buys or  sells is not straightforward. For example, a bull spread involves buying a call option and selling another call option with a higher strike price. Thus, a buy requires a sell, and it is not clear whether treating the two trades separately is appropriate. Index option trading involves many complex trades because taking covered positions in index options is not as easy (or possible) as in equity options. Frequently, the only alternatives to naked positions in index options are complex options. Therefore, one way to reduce the problem of complex trades is to exclude all index trades. As Table 1 indicates, this results in a significant increase in the classification precision of all methods, but loses roughly one quarter of the sample, which is unacceptable.” (Savickas and Wilson, 2003, p. 899) (Savickas and Wilson, 2003, p. 898)
	- Neither of the models can detect complex trades. It would require attention across rows and columns, which we outruled.
	- “In contrast to Pan and Poteshman (2006), we use a unique data set from the International Securities Exchange (ISE), which contains the complete daily record of buy and sell activity in index options over a 12-year period, together with details on whether a transaction is involved in opening or closing an options position. These options are actively traded; indeed, on the ISE, the notional volume in index options is about onefifth of the total notional volume in all individual stock options during our sample period.” (Chordia et al., 2021, p. 1)

“Savickas and Wilson 899 sells is not straightforward. For example, a bull spread involves buying a call option and selling another call option with a higher strike price. Thus, a buy requires a sell, and it is not clear whether treating the two trades separately is appropriate. Index option trading involves many complex trades because taking covered positions in index options is not as easy (or possible) as in equity options. Frequently, the only alternatives to naked positions in index options are complex options. Therefore, one way to reduce the problem of complex trades is to exclude all index trades. As Table 1 indicates, this results in a significant increase in the classification precision of all methods, but loses roughly one quarter of the sample, which is unacceptable.” (Savickas and Wilson, 2003, p. 899)

## time-to-maturity
- “Expiration dummies are particularly good instruments. Investors substitute expiring option positions with similar nonexpiring ones in the three-day window around the expiration day (every third Friday of a month). Because investors are short call and put equity options on average, the rollover creates unprecedentedly large selling pressure in the nonexpiring options. Option expirations create exogenous variation in order imbalance, and thus exogenous variation in market-maker inventories as investors open new positions to replace positions in expiring options. Volatility and returns of the underlying stocks change little around expiration. Thus, fundamentals and informed trading are not responsible for the order imbalance.” (Muravyev, 2016, p. 700)
- “Order imbalance is extremely negative around option expiration because investors are rolling over their positions to nonexpiring options. The selling pressure is particularly large on the postexpiration Monday when the abnormal order imbalance reaches −24%.” (Muravyev, 2016, p. 701)

## Quotes change after the trade
“With respect to the intraday analysis, the interaction between trades and quotes is key to understanding how and why prices change. The literature identifies two reasons why quoted prices increase after a buyer-initiated trade. First, market-makers adjust upward their beliefs about fair value as the trade may contain private information (e.g., Glosten and Milgrom (1985)). Second, market-makers require compensation for allowing their inventory position to deviate from the desired level, and thus a risk-averse market-maker will accommodate a subsequent buy order only at a higher price (e.g., Stoll (1978)).” (Muravyev, 2016, p. 674)

## Quotes NBBO / Exchange
- “Condition (d) also serves another purpose. Since the trade price is equal to the NBBO price quoted by at least two exchanges, this condition resolves ambiguity about trade direction as further discussed in the Internet Appendix.” (Muravyev, 2016, p. 689)

## Algorithm
2.3.7 How to Write the Discussion  Assessment of the results  Comparison of your own results with the results of other studies = Citation of already published literature!  Components  Principles, relationships, generalizations shown by the results = Discussion, not recapitulation of the results  Exceptions, lack of correlation, open points  Referring to published work: = Results and interpretations in agreement with or in contrast to your results  Our Recommendations: The writing of the chapter “Discussion” is the most difficult one. Compare your own data/results with the results from other already published papers (and cite them!). Outline the discussion part in a similar way to that in the Results section = consistency. Evaluate whether your results are in agreement with or in contrast to existing knowledge to date. You can describe why or where the differences occur, e.g. in methods, in sites, in special conditions, etc. Sometimes it is difficult to discuss results without repetition from the chapter “Results”. Then, there is the possibility to combine the “Results” and “Discussion” sections into one chapter. However, in your presentation you have to classify clearly which are your own results and which are taken from other studies. For beginners, it is often easier to separate these sections.


## Other
- calculate $z$-scores / $z$-statistic of classification accuracies to assess if the results are significant. (see e. g., [[@theissenTestAccuracyLee2000]])
- provide $p$-values. Compare twitter / linkedin posting of S. Raschka on deep learning paper.
- When ranking algorithms think about using the onesided Wilcoxon signed-rank test and the Friedman test. (see e. g. , code or practical application in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- To test these hypotheses it would be best if we had the precise motivation behind the trades. While such analysis is not feasible here, using trade classification algorithms, we are able to assign stock and option volume as buyer or seller initiated. Easley et al. (1998) show how this directional volume is more informative than raw volume, because signed volume provides important information about the motivation of the trade (bullish or bearish). ([[@caoInformationalContentOption2005]])
- [https://doi.org/10.1287/mnsc.2019.3398](https://doi.org/10.1287/mnsc.2019.3398)
- https://pdf.sciencedirectassets.com/271671/1-s2.0-S0304405X20X00067/1-s2.0-S0304405X19302831/am.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIDixxfTiKliJIuzoOXxlII71RLwniTDskEPKeGqAyItEAiEA4%2Fytxevo9ZXJNkxW1jrTnKzaaobWySQgbq68siGmgwQquwUI7f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDLKVDK%2Bh4Dg%2B7qRvKCqPBUP%2BVpVVJhJWgxEXvneeMcgDHwz3Q%2BFj4BacIV3D2cLphWRHZirMPOW1Scz6VIzOfzGnUYgdZXRNb0yT8KQur9GvN%2B1TwgULtgOLUlII49PNpfnhgo%2B5TFji2%2BRpB4Bs7BoBu6JZH6x2vrhjfqFGSsl19%2Bsyxe3zfS%2BYZzLkEUBwXTVS0Omt3AWowOaltN5qRbzjH0M16ijT3HTA3BTtQJLZe%2BNqqKsohziXZJ2GIC0I%2BswnrB9qpx8TplWGO62ITP0I4Xa4F2GhzByCl2nrGKeHUdJ03VUa3dYpyw4ml8n3E7ADheEZh4yhh8W3GS%2Btc2AkrpJkl9JpInWeTwijmC5rsQVtRZfYLCNFXdSZkPtWFWOBYM0WVIiRHMx8urSTYs%2FQ5XiP61nmWn%2BlIdyeLDYgg8uYcBCwciMCBdfBKu86mAK42snqIIJC8fHQ6RjZ0HkTxXK3ecfWG9ZD5LYrwOig7B30VufNzSvG%2FnJ8UxeUOPfXcX9Ob8OEUuaWzvTCSU3%2BIsw8vx4%2BpScmof5EwYvWDb4ndAD02RdDsAps9DjoZT0Fo6ezxpGZYsSzJ%2FRvzXoxrKIhTVugO2%2BDJubQ9sHIex7HBmGf1dM3j56ypwqghzdFmDohh4bPT3oYbBkQkIeojKcuifG6RPtAROKxHdSv0Htm9LZrdehnkehKyeFESJ8pcZ9IrTP5sejH9%2BHrVo8m7gjUaYTw6vWdsQxw1dCZ96jSuoANt2O8QvxR5S%2BG078zV0yhx76Y1nuhz3Dzgk%2FJCwLUwklQcsGDNZzXKhuXoZkpyE1sHDgqeSwDU7xYdJEZnQy60exHcdnjh5qQt1cY3ZCb4EMH8Y4yUtDwJgf5YOlxHJo8rElAkn5T3e0w2YDWpAY6sQGhOGrkIFN%2FaJBDi7pCA1DemSn4hZ5FnHX1%2Fh492NDWmIx5ojFpVIHW5WftmRayIG6R4hzuIank8S0YgtSwnLUQpKMXKQEBeTmLwzHqVg7sHjO8dIJ0dtTywPmoP%2Fz7Op4wRbImbrCavUEUdf9bIt8ywHhHPJowJ9OnPNLByKti3DmdKLe%2FjO1iMpJGKTCwRu4WN9EDOsW3Jxo1xPUe3l2EUBbOB5oy6z3j8m1qo4bMbDE%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230623T124336Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYUUDJI6EP%2F20230623%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f0e2119f981b01f0c6f371ba9815e791e0ff23c077563ba5b6e75152e2e77385&hash=97254ce55dfdd1c8959a1ac605bad32a3be5f6fff1c70209a668432c2a4bbeff&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0304405X19302831&tid=pdf-81ea8b48-352c-43fe-bfe6-12fecbdb988f&sid=d1961e3d9967514b7918a74-b56e3090c4eagxrqb&type=client
- https://doi.org/10.1287/mnsc.2019.3529
- https://www.dropbox.com/s/1i4zxc23qm00bv9/OptionMarketMakers.032623.pdf?dl=0
- https://dmurav.com/CV_Dmitry_Muravyev_202305.pdf
- for index options see [[@chordiaIndexOptionTrading2021]]
- To test these hypotheses it would be best if we had the precise motivation behind the trades. While such analysis is not feasible here, using trade classification algorithms, we are able to assign stock and option volume as buyer or seller initiated. Easley et al. (1998) show how this directional volume is more informative than raw volume, because signed volume provides important information about the motivation of the trade (bullish or bearish). ([[@caoInformationalContentOption2005]])

7. Discussion 7.1. Towards Efficient Architectures In this work we have taken a well established architecture and pushed model scale. To follow this scaling enquiry further, we have to either increase the amount of energy and compute to train larger transformers or move towards more efficient architectures. 20 Scaling Language Models: Methods, Analysis & Insights from Training Gopher We break down the computational cost from training Gopher in Table A26 and Appendix F and observe the majority is spent in the linear maps. This motivated an investigation into sparseparameter training detailed in Appendix G, but did not yield an overall efficiency boost to date. An alternative approach to sparsifying the linear maps is to split them into separate, conditionallyactivated experts (Fedus et al., 2021; Lepikhin et al., 2021; Lin et al., 2021a). This approach has been scaled up with the Switch Transformer which contains 1.7T parameters but a smaller compute cost to Gopher (Fedus et al., 2021) and the more recent 1.2T GLaM (?) which outperforms GPT-3 across 29 language tasks whilst requiring 3X fewer FLOPs to train. We separately consider a retrieval mechanism searching over the training set for relevant extracts during pre-training (Borgeaud et al., 2021), partially avoiding the need to memorise knowledge into network weights. This approach reached GPT-3-level language model performance with a 7 billion parameter model and over a 10× reduction in training compute. Thus, whilst this paper focused on transformer models, this is likely a transitory stage as more efficient architectures are developed.

5. Discussion & Conclusion The trend so far in large language model training has been to increase the model size, often without increasing the number of training tokens. The largest dense transformer, MT-NLG 530B, is now over 3× larger than GPT-3’s 170 billion parameters from just two years ago. However, this model, as well as the majority of existing large models, have all been trained for a comparable number of tokens—around 300 billion. While the desire to train these mega-models has led to substantial engineering innovation, we hypothesize that the race to train larger and larger models is resulting in models that are substantially underperforming compared to what could be achieved with the same compute budget. We propose three predictive approaches towards optimally setting model size and training duration, based on the outcome of over 400 training runs. All three approaches predict that Gopher is substantially over-sized and estimate that for the same compute budget a smaller model trained on more data will perform better. We directly test this hypothesis by training Chinchilla, a 70B parameter model, and show that it outperforms Gopher and even larger models on nearly every measured evaluation task. Whilst our method allows us to make predictions on how to scale large models when given additional compute, there are several limitations. Due to the cost of training large models, we only have two comparable training runs at large scale (Chinchilla and Gopher), and we do not have additional tests at intermediate scales. Furthermore, we assume that the efficient computational frontier can be described by a power-law relationship between the compute budget, model size, and number of training tokens. However, we observe some concavity in log 𝑁𝑜𝑝𝑡 at high compute budgets (see Appendix E). This suggests that we may still be overestimating the optimal size of large models. Finally, the training runs for our analysis have all been trained on less than an epoch of data; future work may consider the multiple epoch regime. Despite these limitations, the comparison of Chinchilla to Gopher validates our performance predictions, that have thus enabled training a better (and more 1 lightweight) model at the same compute budget. Though there has been significant recent work allowing larger and larger models to be trained, our analysis suggests an increased focus on dataset scaling is needed. Speculatively, we expect that scaling to larger and larger datasets is only beneficial when the data is high-quality. This calls for responsibly collecting larger datasets with a high focus on dataset quality. Larger datasets will require extra care to ensure train-test set overlap is properly accounted for, both in the language modelling loss but also with downstream tasks. Finally, training for trillions of tokens introduces many ethical and privacy concerns. Large datasets scraped from the web will contain toxic language, biases, and private information. With even larger datasets being used, the quantity (if not the frequency) of such information increases, which makes dataset introspection all the more important. Chinchilla does suffer from bias and toxicity but interestingly it seems less affected than Gopher. Better understanding how performance of large language models and toxicity interact is an important future research question. While we have applied our methodology towards the training of auto-regressive language models, we expect that there is a similar trade-off between model size and the amount of data in other modalities. As training large models is very expensive, choosing the optimal model size and training steps beforehand is essential. The methods we propose are easy to reproduce in new settings.