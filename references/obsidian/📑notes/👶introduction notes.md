## Motivation

**What is the problem?**
- The validity of many economic studies hinges on the ability to properly classify trades as buyer or seller-initiated. ([[@odders-whiteOccurrenceConsequencesInaccurate2000]])
- “Such indication made directly from the data is nowadays in mostly cases inaccessible, since the majority of public databases including transaction data do not contain information of trade initiators and trade direction.” ([[@nowakAccuracyTradeClassification2020]], p. 65)
- “Who is buying and who is selling are important elements in determining the information content of trades, the order imbalance and inventory accumulation of liquidity providers, the price impact of large trades, the effective spread, and many other related questions. Unfortunately, commonly available high frequency databases do not provide in? formation on trade direction. Consequently, empirical researchers have relied on trade direction algorithms to classify trades as buyer or seller motivated.” ([[@ellisAccuracyTradeClassification2000]], p. 529)
- “The Berkeley Options Data Base does not classify trades as buyer-initiated or seller-initiated. This classification must be done using quote and trade information.” ([Easley et al., 1998, p. 453](zotero://select/library/items/593W67XA)) ([pdf](zotero://open-pdf/library/items/ZBEQIUNK?page=23&annotation=GXHQMKIW))
- Despite the overall importance for option research, the trade initiator is commonly not provided and inferred using simple simple heuristics.
- Commonly heuristics were proposed for the stock market and are thouroughly teste

**Applications that require the trade initiator**
- From expose: Determining whether a trade is buyer or seller-initiated is ubiquitous for many problems in option research. Typical applications include the study of option demand [[@garleanuDemandBasedOptionPricing2009]] or the informational content of option trading [[@huDoesOptionTrading2014]] or [[@panInformationOptionVolume2006]]. Despite the overall importance for empirical research, the true initiator of the trade is often missing in option data sets and must be inferred using trade classification algorithms[[@easleyDiscerningInformationTrade2016]].
- From Grauer paper. Make sure examples do not overlap! Particularly, the trade direction is required to determine the information content of trades, the price impact of customer transactions, as well as the order imbalance and inventory accumulation of intermediaries. Important examples are studies on option demand (Gârleanu, Pedersen, and Poteshman (2009); Muravyev and Ni (2020)), option order flow (Muravyev (2016)), and option price pressures (Goyenko and Zhang (2021)).
- trade site classification matters for several reasons, market liquidity measures, short sells, study of bid-ask-spreads.
- Option order flow [[@muravyevOrderFlowExpected2016]]
- Order imbalances [[@huDoesOptionTrading2014]] (option order flow contains valuable information about the underlying stock)
- Find some more (...)
- Possible application of the lee-ready algorithm -> market sideness? ([[@sarkarMarketSidednessInsights2023]])
- “The information which party to a trade is a trade initiator is indispensable to specify the trade indicator models used to investigate the intraday price formation (Glosten and Harris 1988; Huang and Stoll 1997; Madhavan 1992; McGroarty et al. 2007; Hagströmer et al. 2016). Moreover, the identification of party to a trade which is responsible for initiating a particular transaction is advantageous to clarify many important issues related to the market microstructure. First of all, it may be used to ascertain the information content of trades. Second, it can help to figure out the magnitude of the order imbalance as well as the proportion of the inventory accumulation made by the liquidity suppliers. Third, it helps to assess the price impact of large in volume transactions as well as the magnitude of effective spread (Ellis et al. 2000).” ([[@nowakAccuracyTradeClassification2020]], 2020, p. 66)
- “Various papers illustrate the consequences of inaccurate trade classification in empirical finance. For example, Boehmer et al. (2007) show analytically and empirically that inaccurate classification of trades leads to downward-biased PIN (probability of informed trade) estimates and that the magnitude of the bias is related to a security’s trading intensity. Using two separate periods around the NYSE’s change to a tick size of $1/16 in June 1997, Peterson and Sirri (2003) report that actual execution costs are overstated by up to 17% using effective spread estimates that incorporate errors in trade direction and benchmark quote assignments, and that the highest biases occur for small trades and for trades of larger firms.” ([[@aktasTradeClassificationAccuracy2014]], 2014, p. 260)
- “Much of market microstructure analysis is built on the concept that traders learn from market data. Some of this learning is prosaic, such as inferring buys and sells from trade execution. Other learning is more complex, such as inferring underlying new information from trade executions. In this paper, we investigate the general issue of how to discern underlying information from trading data. We examine the accuracy and efficacy of three methods for classifying trades: the tick rule, the aggregated tick rule, and the bulk volume classification methodology. Our results indicate that the tick rule is a reasonably good classifier of the aggressor side of trading, both for individual trades and in aggregate. Bulk volume is shown to also be reasonably accurate for classifying buy and sell trades, but, unlike the tick-based approaches, it can also provide insight into other proxies for underlying information.” ([Easley et al., 2016, p. 284](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=16&annotation=VC98DC2N))
- Despite the second observation, the trade direction of the liquidity demanding side of the order flow remains a popular indicator of informed trading (see, e.g., Bernile et al., 2016; Chordia et al., 2017; Hu, 2014, 2017; Muravyev, 2016) the appropriateness of which is context specific but particularly sensible when informed traders demand immediacy for their transactions in order to gain most from their informational advantage. In these cases, studies rely on the classical classification algorithms, most prominently the Lee and Ready (1991) algorithm, to obtain the indicator of the liquidity demanding side of the transaction, the trade initiator, as do traditional measures of market liquidity (Huang and Stoll, 1996; Fong et al., 2017). (https://dauphine.psl.eu/fileadmin/mediatheque/chaires/fintech/articles/1_UPDATE_Simon_Jurkatis_YFS2019.pdf)
- Empirical market microstructure research often requires knowledge about whether a transaction was initiated by a buyer or a seller. Examples include, but are not limited to, accurate calculation of effective spreads (Lightfood et. al. 1999), the identification of the components of the bid-ask spreads using methods based on a trade indicator variable (Huang / Stoll 1997) and the estimation of certain structural microstructure models (e.g. Easley et. al. 1996). (found in [[@theissenTestAccuracyLee2000]])
- The importance of identifying liquidity demanders in studies of financial markets is well established, given that demanding liquidity is known to require the liquidity demander to pay a fee (Grossman and Miller, 1988). Indeed, much work has gone into accurately identifying the initiators of trades (i.e., liquidity demanders) in equity markets (Lee and Ready, 1991)—with Easley, de Prado and O’Hara (2016) accurately inferring trade initiation in modern equity markets. Yet, as will be elaborated in what follows, methods used for assigning trade initiation in equity markets are not sufficiently applicable to OTC markets—and thus, perhaps surprisingly, liquidity demanders (vs. providers) are typically not identified in these markets.

**Transition**
- Why is there a need for alternatives? What happend and how this transfer / motivate the use of machine learning?
- Extant methods are adapted from the stock market (...).  Commonly stock trade classification algorithms are used, that have been transferred to the option market
- An initiative of ([[@grauerOptionTradeClassification2022]]) proposed new rules, tested in the option market
- “Although extant algorithms are adequate to the basic job of sorting trades, our work suggests that a refinement to the extant methods of classifying trades will do even better.” (Ellis et al., 2000, p. 539) -> nice word "extant methods"
- over time proposed methods applied more filters / got more sophisticated but didn't substantially improve im some cases. See e. g., [[@finucaneDirectTestMethods2000]] 
- Is it time to switch / test to another paradigm and let the data speak?
- Methods have become more sophisticated resulting in more complex decision boundaries
- Crisp sentence of what ML is and why it is promising here. 
- From expose: "The work of ([[@grauerOptionTradeClassification2022]]) and ([[@savickasInferringDirectionOption2003]]) raises concerns about the applicability of standard trade signing algorithms to the option market due to deteriorating classification accuracies."

**Research Question**
- What question do I solve? (SMART)
- From Expose: "Against this backdrop, the question is, can an alternative, machine learning-based classifier improve upon standard trade classification rules?"

## Contributions
- from expose: In the introduction, we provide motivation and present our key findings. The contributions are three-fold: (I) We employ state-of-the-art machine learning algorithms i.~e., gradient-boosted trees and transformer networks, for trade classification. Tree-based approaches outperform state-of-the-art trade classification rules in out-of-sample tests. (II) As part of semi-supervised approaches, we study the impact of incorporating unlabelled trades into the training procedure on trade classification accuracy. (III) We consistently interpret feature contributions to classical trade classification rules and machine learning models with a game-theoretic approach.

Our contributions are n-fold:
- Our paper contributes to at least two strands of literature. First, it is
- We compare our streaming algorithm to the original... new state-of-the-art in terms of out-of-sample accuracy without additional data requirements. Stable results in out-of-sample test on CBOE dataset. What are results with and without additional data requirements?
- test gradient-boosting and tabular transformers for the problem of trade classification
- game theoretic approach to study the effect of features on the prediction
- new framing as semi-supervised learning problem. Enables to learn on learn on unlabelled and labelled trades simultaneously
- we test the algorithms for the purpose of estimating effective spreads purpose of calculating effective spreads


## Outline
- The remainder of this paper is organized as follows:


**Examples:**
Chapter 2 introduces the related work. In Chapter 3, we present the theoretical basis for the design of our algorithm. Chapter 4 details the algorithm design and the resulting trade-offs. We evaluate our algorithm in Chapter 5 and conclude in Chapter 6. The Appendix provides extra information about the effect of various parameter settings.

**Examples:**
The rest of the paper is organized as follows. Section 2 describes
the data and defines the variables used in the regression analyses.
Section 3 examines the trading patterns around earnings announcements and shows that the surge in volume is mostly due to small
investors engaged in speculative trading. A refined test is offered
to prove that the apparent predictability of option turnovers for future returns is simply spurious. Section 4 presents empirical results
based on cross-section and time-series regression analyses employing various proxies of information asymmetry and opinion dispersion. Additional tests and robustness checks are presented in
Section 5. Section 6 offers some reconciliation with intuition and
the existing literature. Section 7 concludes the paper. The appendix
is relegated to the end.

**Examples:**
The remainder of this paper is organized as follows. Section 2 gives a literature overview how consumers react to price changes. In Section 3, publications related to the financial benefits from Demand Response are reviewed.
Afterwards, Section 4 identifies parameters that govern decisions in Demand
Response programs to pioneer a mathematical problem such that Demand Response decisions of retailers are optimized. Finally, Section 5 evaluates the
decisions derived by the model in a simulation based on historic data and analyzes their financial benefits.
