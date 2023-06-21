 Supply of sufficient background information  Definition of the (general) problem  State of knowledge about the problem (in detail)  Carving out the gap of research and development  Objectives of the work/paper  Overall purpose/goal: to fill the gap in R&D  Stating specific objectives: as objectives, as hypotheses or as research questions.  Our Recommendations: Begin the chapter “Introduction” with the problem statement. Set the context of your research and capture the reader's attention. Explain the background of your study, starting from a broad perspective narrowing it down to your own research goal. Go into more detail when presenting the state of knowledge to date. Review what is known about your research topic as far as it is relevant. Then, develop the research or development gap from the existing knowledge. This is an important part, because your research work starts with the “gap” and should result in new knowledge or new findings. Therefore, the goal of your research work is to close the gap which you have discovered. Then, specify your objectives. The easiest way is to set some special research questions, which you will directly answer later in the respective chapter “Conclusion”. Thus, you do not mix up/confuse this part with working steps. For example, “to make life cycle assessment”, “to make soil analysis” are working steps, and not objectives as such


Our paper contributes....

Motivated by these considerations, we investigate how the predictability documented in our main test varies across option contracts with differing degrees of leverage. We find that option signals constructed from deep out-of-the-money options, which are highly leveraged contracts, exhibit the greatest level of predictability, while the signals from contracts with low leverage provide very little, if any, predictability.3 ([[@panInformationOptionVolume2006]])

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


## Outline
The remainder of this paper is organized as follows. Cref-[[👪Related Work]] reviews publications on trade classification in option markets and using machine learning, thereby underpinning our research framework. Cref-[[🍪Selection Of Supervised Approaches]] discusses and introduces supervised methods for trade classification. Then, cref- [[🍪Selection Of Semisupervised Approaches]] extends the previously selected algorithms for the semi-supervised case. We test the models in cref-[[🌏Dataset]] in an empirical setting. In cref-[[🍕Application study]]  we apply our models to the problem of effective spread estimation. Finally, cref-foo concludes.




## Contributions
- from expose: In the introduction, we provide motivation and present our key findings. The contributions are three-fold: (I) We employ state-of-the-art machine learning algorithms i.~e., gradient-boosted trees and transformer networks, for trade classification. Tree-based approaches outperform state-of-the-art trade classification rules in out-of-sample tests. (II) As part of semi-supervised approaches, we study the impact of incorporating unlabelled trades into the training procedure on trade classification accuracy. (III) We consistently interpret feature contributions to classical trade classification rules and machine learning models with a game-theoretic approach.
- through visualising attention we are able to establish a theoretical link between rule-based classification and machine learning

Our contributions are n-fold:
- Our paper contributes to at least two strands of literature. First, it is
- We compare our streaming algorithm to the original... new state-of-the-art in terms of out-of-sample accuracy without additional data requirements. Stable results in out-of-sample test on CBOE dataset. What are results with and without additional data requirements?
- test gradient-boosting and tabular transformers for the problem of trade classification
- game theoretic approach to study the effect of features on the prediction
- new framing as semi-supervised learning problem. Enables to learn on learn on unlabelled and labelled trades simultaneously
- we test the algorithms for the purpose of estimating effective spreads purpose of calculating effective spreads
- based on a unified framework compare feature importances of rule-based approaches

**Examples**
Impressed by the superiority of tree-based models on tabular data, we strive to understand which inductive biases make them well-suited for these data. By transforming tabular datasets to modify the performances of different models, we uncover differing biases of tree-based models and deep learning algorithms which partly explain their different performances: neural networks struggle to learn irregular patterns of the target function, and their rotation invariance hurt their performance, in particular when handling the numerous uninformative features present in tabular data. Our contributions are as follow: 1. We create a new benchmark for tabular data, with a precise methodology for choosing and preprocessing a large number of representative datasets. We share these datasets through OpenML [Vanschoren et al., 2014], which makes them easy to use. 2. We extensively compare deep learning models and tree-based models on generic tabular datasets in multiple settings, accounting for the cost of choosing hyperparameters. We also share the raw results of our random searches, which will enable researchers to cheaply test new algorithms for a fixed hyperparameter optimization budget. 3. We investigate empirically why tree-based models outperform deep learning, by finding data transformations which narrow or widen their performance gap. This highlights desirable biases for tabular data learning, which

**Examples:**
applications in biomarker detection [Climente-González et al., 2019] and wind power prediction [Bouche et al., 2022], clustering [Song et al., 2007, Climente-González et al., 2019], and causal discovery [Mooij et al., 2016, Pfister et al., 2018, Chakraborty and Zhang, 2019, Schölkopf et al., 2021]. Various estimators for HSIC and other dependence measures exist in the literature, out of which we summarize the most closely related ones to our work in Table 1. The classical V-statistic based HSIC estimator (V-HSIC; Gretton et al. [2005], Quadrianto et al. [2009], Pfister et al. [2018]) is powerful but its runtime increases quadratically with the number of samples, which limits it applicability in largescale settings. To tackle this severe computational bottleneck, approximations of HSIC (N-HSIC, RFF-HSIC) have been proposed [Zhang et al., 2017], relying on the Nyström [Williams and Seeger, 2001] and the random Fourier feature (RFF; Rahimi and Recht [2007]) method, respectively. However, these estimators (i) are limited to two components, (ii) their extension to more than two components is not straightforward, and (iii) they lack theoretical guarantees. The RFF-based approach is further restricted to finitedimensional Euclidean domains and to translation-invariant kernels. The normalized finite set independence criterion (NFSIC; Jitkrittum et al. [2017]) replaces the RKHS norm of HSIC with an L2 one which allows the construction of linear-time estimators. However, NFSIC is also limited to two components, requires R d -valued input, and analytic kernels [Chwialkowski et al., 2015]. A novel complementary approach is the kernel partial correlation coefficient (KPCC; [Huang et al., 2022]) but when applied to kernel-enriched domains its runtime complexity is cubic in the sample size. The restriction of existing HSIC approximations to two components is a severe limitation in recent applications like causal discovery which require independence tests capable of handling more than two components. Furthermore, the emergence of large-scale data sets necessitates algorithms that scale well in the sample size. To alleviate these bottlenecks, we make the following contributions.

We propose Nyström M-HSIC, an efficient HSIC estimator, which can handle more than two components and has runtime O ´ Mn13 ` Mn1n ¯ , where n denotes the number of samples, n 1 ! n stands for the number of Nyström points, and M is the number of random variables whose independence is measured. 2. We provide theoretical guarantees for Nyström MHSIC: we prove that our estimator converges with rate O ` n ´1{2 ˘ for n 1 „ ? n, which matches the convergence of the quadratic-time estimator. 3. We perform an extensive suite of experiments to demonstrate the efficiency of Nyström M-HSIC. These applications include dependency testing of media annotations


We recall the existing HSIC estimator V-HSIC in Section 3.1, and its Nyström approximation for two compo- nents in Section 3.2. We present our proposed Nyström approximation for more than two components in Section 4.


To answer this question, we model trade classification throughTransformers and gradient boosting. We consider the cases, where labelled trades 
## Contributions
- from expose: In the introduction, we provide motivation and present our key findings. The contributions are three-fold: (I) We employ state-of-the-art machine learning algorithms i.~e., gradient-boosted trees and transformer networks, for trade classification. Tree-based approaches outperform state-of-the-art trade classification rules in out-of-sample tests. (II) As part of semi-supervised approaches, we study the impact of incorporating unlabelled trades into the training procedure on trade classification accuracy. (III) We consistently interpret feature contributions to classical trade classification rules and machine learning models with a game-theoretic approach.
- through visualising attention we are able to establish a theoretical link between rule-based classification and machine learning

Our contributions are n-fold:
- Our paper contributes to at least two strands of literature. First, it is
- We compare our streaming algorithm to the original... new state-of-the-art in terms of out-of-sample accuracy without additional data requirements. Stable results in out-of-sample test on CBOE dataset. What are results with and without additional data requirements?
- test gradient-boosting and tabular transformers for the problem of trade classification
- game theoretic approach to study the effect of features on the prediction
- new framing as semi-supervised learning problem. Enables to learn on learn on unlabelled and labelled trades simultaneously
- we test the algorithms for the purpose of estimating effective spreads purpose of calculating effective spreads
- based on a unified framework compare feature importances of rule-based approaches

Our work makes the following contributions:
1. 
2. We establish a link between classical trade classification rules and  mach consistently interpret feature attributions of classical trade classification rules and machine learning models with a game-theoretic approach.
3. 

Our experiments compare SAGE to several baselines and demonstrate that SAGE’s feature importance values are more representative of the predictive power associated with each feature. We also show that when a model’s performance is unexpectedly poor, SAGE can help identify corrupted features.

Our experiments document a superiority of machine learning-based 

as it threatens

Whilst we reach the same conclusion, we estimate that large models should be trained for many more training tokens than recommended by the authors.

![[Pasted image 20230621094017.png]]

![[Pasted image 20230621091250.png]]

![[Pasted image 20230620171109.png]]
![[Pasted image 20230620181450.png]]

![[Pasted image 20230621074457.png]]

![[Pasted image 20230621074845.png]]

Overall, however, the tick rule still outperforms BVC in terms of pure classification accuracy, a point analyzed in detail by Andersen and Bondarenko (2015). The major argument brought forward in favor of bulk classification is not raw buy-sell classification accuracy, though.

Inconsistent with Easley et al. (2012a) is a result in Chakrabarty et al. (2012b) indicating the bulk tick rule to be a better The Sensitivity of VPIN to the Choice of Trade Classification Algorithm | 73 indicator of order imbalance for all volume and time bars. The divergence of their conclusions could arise from the different methods used to estimate the accuracy of order flow imbalance. Easley et al. (2012a) use the correlation to the high-low spread, Chakrabarty et al. (2012b) use actual order imbalance. Overall, Chakrabarty et al. (2012b) focus on answering “who is right” and “who is faster” in classifying trades and less on the implications for calculating and applying VPIN, as we do in the current study. Further, Chakrabarty et al. (2012b) consider only the tick rule, but not more advanced trade classification algorithms as we do in this chapter.

option marg O'Hara, and Srinivas (1998) and Pan and Poteshman ( trading volume in the option market can help forecast s and Griffin (2000) and others document abnormal tradi market prior to takeover


(b)Option moneyness measured by absolute option delta (i.e., the sensitivity oftheoption price to changes in the underlying price)is between 0.2 and 0.8, that is, options with at least some “optionality” are selecte

https://onlinelibrary.wiley.com/doi/epdf/10.1111/jofi.12380

.2610. We also performed additional analysis on a sample of put options. Overall we found the information shares based on put options to be roughly comparab to those based on call options. Finally, in our main analysis, we estimated the VMA model using 300 lags. We re-estimated the model using up to 600 lag with no significant change in the r