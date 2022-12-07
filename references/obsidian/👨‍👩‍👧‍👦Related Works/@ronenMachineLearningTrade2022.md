title: Machine Learning and Trade Direction Classification: Insights From the Corporate Bond Market
authors: Tavy Ronen, Mark A. Fedenia, Seunghan Nam
year: 2021
tags :  #trade-classification #lr #tick-rule #quote-rule #random-forest #supervised-learning #bvc
#bonds #stocks 
status : #📥
related: 
- [[@fedeniaMachineLearningCorporate2021]] (samey)
- [[@rosenthalModelingTradeDirection2012]]
- [[@easleyDiscerningInformationTrade2016]] (paper contains some notes on this paper)
- [[@leeInferringTradeDirection1991]] (serves as a benchmark)

## Notes
### Problems
- Neural net is rather deep 15 layers. (?) Important hyperparameters (e. g., learning rate) or configurations (e. g., capacity) are unknown. Also, its not known if neural nets were regularized, which is important. (see e. g., [[@kadraWelltunedSimpleNets2021]])
- Similarily for RF only the number of trees is known, not how it was set.
- Hyperparam search is unclear. No validation set used? -> Tuned on test set?
- Standardization / normalization of data is unclear -> is known to hinder to hinder learning in neural networks.
- They do not handle imbalancies in the data.
- Unclear which algorithm posed practical problems to handle the amount of data.
- At time of reading its inclear if one should perform a 70-30 random split, as they did. Read on bulk classification (e. g. [[@chakrabartyEvaluatingTradeClassification2015 1]]). Shouldn't surrounding trades inform other trades? It's unclear what is meant by 10-fold-cv is not significantly dfferent. Split is obviously not maintained for the equity data, where 2/3s are used for training and the remainder for testing.
- It's confusing that the feature set is not maintained throughout the work. See e. g., "We include oder flow information ... by augmenting ... with quote features".
- They do some transfer learning, where the entire model is transferred -> What about parameter transfer.
### Content
- Authors apply machine learning methods to understand improve trade direction classification. Algorithms include decision trees, discriminant anlysis, logistic regression, SVMs, $k$ nearest neighbour, and neural networks.
- They conduct their study in the bond and equity market.
- They train both bulked and single observation versions.
- They emphasize the importance of optimizing the feature set and study the performance of trading rules under different market conditions e. g., trading frequencies.
- **Important findings:**
	- ML models at least matches or improves upon the accuracy of classical rules in the bond and stock market.
	- The random forest trained on additional features improves over the tick rule by 8.3 % in the bond market. Thus, they conlcude random forest is superior.
	- In their work the accuracy is higher for small trades in the equity market, which contradicts [[@chakrabartyTradeClassificationAlgorithms2012]]
	- In their case more features help to improve performance of the random forest.
	- RF on feature set 5 outperforms the tick rule on all trading frequency ranges, even if there is just one trade at that day.
	- The accuracy of the random forest on feature set five is lowest during the morning after no or low trading activity and peaks during the evening.
	- For more liquid bonds, random forests increase even further.
	- In the stock market their random forest improves upon trade size rule by 3.3 % and LR by 3.6 % when trained on quote features.
	- Accuracy for smaller trades is higher than for larger trades. Model performs best if price uncertainty is low.
	- The apply transfer learning to of a RF trained in the bond market to the equity market with a poor accuracy. The accuracy is just 61.2 % whereas the trick rule achieves 71.7 %. They conclude "we can report that machine learning does improve when trained on in-market data"
- They raise several questions:
	- Can ML discover simple rules like the TR rule?
	- Can ML improve when TR is used as a feature?
- They perform some sort of transfer learning. They train a model in the corporate bond market and apply the model in the stock market.
- Models are trained using a moving window with a train-test set and are not tuned.
- They say: "... We cannnot train ad model on the data used to compute existing classification rules without human intervention. .... machine learning must be paired with the intuition..." -> isn't feature engineering a normal thing?
- **Future work:
	- use additional features
	- invetigate models that can handle time series components in the panel data -> why not use better feature engineering?
- **Data sets:**
	- TRACE (bonds) between July 1, 2002 and December 31, 2019. 
	- ITCH (stocks) from December 9, 2013 - December 13, 2013.  Sample consists of roughly 1.4 million non-cross trades.
## Annotations

“As trade initiation information is generally not provided in most intraday transaction databases, trade direction is often inferred from local price and/or quote behavior,” ([Ronen et al., 2022, p. 2](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=3&annotation=Q46E2CX6))

“The efficacy of these rules has been hotly debated, and their relative accuracy in different markets has been studied extensively.” ([Ronen et al., 2022, p. 2](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=3&annotation=ZNCJC95A))

“Our paper contributes to the market microstructure literature by examining the applicability of machine learning methods to better understand and improve trade direction classification. In particular, we introduce a machine learning model that outperforms traditional classifiers. Moreover, we illustrate the importance of optimizing the feature set in correctly classifying trade direction and provide new insights on the efficacy of trading rules in different market conditions.” ([Ronen et al., 2022, p. 3](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=4&annotation=GHNUUP4C))

“but also allows us to discover market characteristics that may affect our understanding of existing trade classification rules in general which may extend to other markets” ([Ronen et al., 2022, p. 4](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=5&annotation=XF3X4DIZ))

“We find that machine learning models offer trade classification accuracy that at least matches and often exceeds the accuracy of the most commonly used models in the bond and stock markets.” ([Ronen et al., 2022, p. 4](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=5&annotation=PW6FW8ZS))

“The Random Forest (RF) algorithm is found to be superior to a number of other machine learning choices as well as to traditional classifiers. The combination of additional features and a more flexible form of the classification function improves classification accuracy. For example, improvements over the tick rule (TR) are roughly 8.3%.” ([Ronen et al., 2022, p. 4](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=5&annotation=EFUCCYWT))

“The accuracy of trade direction predictors depends on the trading and information environment in the market at the time of the trade. Consistent with Ellis et al. (2000), Finucane (2000), Chakrabarty et al. (2007), and Chakrabarty et al. (2012) who find that trade direction classifier accuracy is generally lower for larger trades in the equity markets, we find that the accuracy for smaller trades seem to be higher than for larger ones.” ([Ronen et al., 2022, p. 4](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=5&annotation=2MTDQDP2))

“Further, as bond liquidity increases, the gap between RF and TR accuracy widens” ([Ronen et al., 2022, p. 4](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=5&annotation=HSLQ7N4S))

“Using the inferences we glean from the bond market regarding model and feature choice, we examine the efficacy of an equity-trained machine learning model in our TotalView-ITCH sample of Nasdaq stocks trading in 2013 and find that it is on a par with both TR and LR. When we supplement our feature set with quotes for comparability with classifiers that employ order information, RF outperforms TR by 3.3% and LR by 3.6%.” ([Ronen et al., 2022, p. 5](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=6&annotation=JKDP7GF8))

“Indeed, the positive correlation between the RF algorithm and trade frequency across all ranges renders it more suitable for today’s faster markets, particularly in light of recent papers on new trade classification algorithms (Easley et al. (2016), Panayides et al. (2019), Chakrabarty et al. (2015), and Carrion and Kolay (2021)), which call in to question the ability of traditional trade direction classifiers to achieve high accuracy rates as the number of trades per unit time increases.” ([Ronen et al., 2022, p. 6](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=7&annotation=FYAG6MB2))

“BVC constitutes a robust alternative to traditional trade-level classification methods, particularly useful in low latency fragmented markets, where it is challenging to ensure that the exact sequencing of orders and trades are correctly reported” ([Ronen et al., 2022, p. 7](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=8&annotation=FIXI85BW))

“Since BVC aggregates trading activity across volume, time or trade intervals (bars), it is not directly comparable to trade-based rules, and studies examining relative efficacy generally compute bulk versions of TR and LR.” ([Ronen et al., 2022, p. 8](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=9&annotation=UXRF37MB))

“The classifiers described above rely entirely on the intuition of the researcher to define an estimator that captures the trading dynamics.” ([Ronen et al., 2022, p. 9](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=10&annotation=3IJIFQMM))

“By the same token, we admit that a myriad of models and combinations of models (ensemble models) are not explored. Our purpose here is not to identify the best prediction model possible. Instead, our goal is to see if an inductive approach can deliver insights into a classic problem in market microstructure” ([Ronen et al., 2022, p. 10](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=11&annotation=DGJN3L7R))

“Can machine learning inductively discover a very simple but useful trade classification rule, namely TR?” ([Ronen et al., 2022, p. 10](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=11&annotation=9J2WWWRK))

“Additionally, we can test whether machine learning improves the TR by adding the tick rule prediction as a feature to see if the resulting ensemble forecast accuracy is more accurat” ([Ronen et al., 2022, p. 10](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=11&annotation=JYQLBL6N))

“For this inquiry, we train a model in the corporate bond market and use the model to predict trade direction in the stock market.” ([Ronen et al., 2022, p. 10](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=11&annotation=4NBFJ4VU))

“We examine all corporate bonds disseminated on the Enhanced Trade Reporting and Compliance Engine (TRACE) between July 1, 2002 and December 31, 2019.” ([Ronen et al., 2022, p. 11](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=12&annotation=2DVV2WE2))

“We employ five days of trade data from the TotalView-ITCH database provided by Nasdaq (December 9, 2013-December 13, 2013) for 1,984 NASDAQ stocks (1,493,298 non-cross trades). Our ITCH data include signed orders and trades, time-stamped to the nanosecond. For each trade, we capture ticker, price, size, timestamp, and the buy/sell trade flag.” ([Ronen et al., 2022, p. 13](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=14&annotation=LV5YH7UN))

“We consider several algorithms including Decision Trees, Discriminant Analysis, Logistic Regression, Support Vector Machines, k-Nearest Neighbor and a Neural Network (NN)” ([Ronen et al., 2022, p. 13](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=14&annotation=QNRVPD62))

“Some 10The imbalance in the number of buys versus sells in our sample is likely not a concern, since we do not train the model on this sub-sample. 11Since we aggregate ITCH trades to the millisecond, approximately 14% of ITCH trades are discarded in this matching procedure. An alternative matching procedure is considered, in which we compare each ITCH trade with the next or last available millisecond and retain an additional 13.5% of trades, which yields stronger results regarding the dominance of our RF model, rendering our choice the conservative one. 13 Electronic copy available at: https://ssrn.com/abstract=421331” ([Ronen et al., 2022, p. 13](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=14&annotation=AQ75C8E2))

“of these algorithms were not well suited to the quantity of data we employed or they did not improve forecasts enough to be useful.” ([Ronen et al., 2022, p. 14](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=15&annotation=F4YHCJZ9))

“We explicitly report results for RF, NN, and Logistic Regression. However, we focus our attention on RF as it is the dominant performer” ([Ronen et al., 2022, p. 14](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=15&annotation=X564UFEC))

“RF was implemented with 100 trees and increasing that number did little to improve out-of-sample accuracy. We increased the number of leaves in a tree and varied the pruning parameters. These changes did not affect the accuracy significantly. Additionally, we employed 10-Fold cross-validation to check if our 70/30 random partition is appropriate. In 10-Fold cross-validation, we divide the sample into ten equal sizes and ten iterations of Random Forest using 90/10 partition are performed.” ([Ronen et al., 2022, p. 14](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=15&annotation=ZX5WIRFP))

“The algorithms we employ do not efficiently incorporate lagged information. Therefore we include relevant lagged data in each observation.” ([Ronen et al., 2022, p. 15](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=16&annotation=7AIG6L33))

“We apply the classification algorithms to seven feature sets labeled {1} - {7} described below and shown in Table 4.” ([Ronen et al., 2022, p. 15](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=16&annotation=Q7FPD7RH))

“Our first task is to invoke a number of popular classification algorithms and evaluate their efficacy in fifteen rolling windows of three years each. In each window we use 70% of the data to train the classifier and then measure the out-of-bag prediction accuracy on the remaining 30% of the observations.14” ([Ronen et al., 2022, p. 16](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=17&annotation=M7VJSUMN))

“We find that a Neural Network performs on par with most other decision tree models (not reported) except for the Random Forest. Logistic Regression performs about as well as TR (the most widely used classifier in markets without pre-trade transparency) but not as well as either a Neural Network or Random Forest. As mentioned earlier, the Random Forest approach (Breiman (2001) has an advantage over other algorithms we tried partly because it is an ensemble method.” ([Ronen et al., 2022, p. 16](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=17&annotation=7HJFAIYK))

“Across all random forest models we consider, adding more inclusive feature sets improve classification accuracy” ([Ronen et al., 2022, p. 16](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=17&annotation=TGVG55IM))

“Choices regarding the rolling window periods and percentages of training and testing batches are somewhat ad-hoc by nature. Our results are robust to all other choices made regarding the testing-to-training ratio (85%/15%, and 60%/40%), as well as to fixing the ratios to the first and last data points (as opposed to using random batches that maintain these ratios). In addition, we consider one and two year rolling windows and obtain similar results.” ([Ronen et al., 2022, p. 16](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=17&annotation=S2GRNP7D))

“On average, RF{4} (RF{5}) accuracy level is 81.1% (80.6%) compared with accuracy rates of 77.1%, 77.2%, and 78.1% for RF{1}, RF{2}, and RF{3}, respectively.” ([Ronen et al., 2022, p. 17](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=18&annotation=VQZVD4HY))

“We now examine how well the models shown in Table 5 hold up out-of-sample” ([Ronen et al., 2022, p. 17](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=18&annotation=XG5RJJ6P))

“Compared to out-of-bag accuracy levels shown in Table 5, the out of sample prediction rates are mildly lower, although the ordinal properties across model accuracy rates remains similar. RF{4} still outperforms NN{4}, LG{4}, and TR. Since RF{4} and RF{5} are effectively equally accurate, and since RF{5} is more universally applicable in both the bond and stock markets, we focus on RF{5} and de-emphasize RF{4} going forward.” ([Ronen et al., 2022, p. 17](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=18&annotation=BD523LTG))

“Table 7 reports the incremental out-of-sample prediction accuracy provided by each of the feature sets described in Table 4.” ([Ronen et al., 2022, p. 18](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=19&annotation=5LYGRNEC))

“We graph the change in RF{5} and TR accuracy as a function of the number of transactions per day, for the entire sample. The figure shows improved accuracy rates for both models as trading activity increases, with RF{5} uniformly outperforming TR across all trading frequency ranges, even in the extreme case of one trade per day.” ([Ronen et al., 2022, p. 19](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=20&annotation=QVYXMYWK))

“As the time elapsed since the last trade decreases, accuracy rates for model RF{5} and TR improve. Prediction rates for RF{5} (TR) range from 71.57% (68.05%) in decile 1 to 81.43% (72.33%) in decile 10.” ([Ronen et al., 2022, p. 19](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=20&annotation=6S9UAUPU))

“Figure 2a shows that trade direction accuracy is lowest for both RF{5} and TR in the morning hours after long periods of low or no trading information. Conversely, trade accuracy levels peak for both RF{5} and TR in the middle of the trading day and plummet at 6PM.” ([Ronen et al., 2022, p. 20](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=21&annotation=JBU62STJ))

“Bulk RF{5}.25 Table 11 reports that across all bars (volume, time, and trade), Bulk RF{5} outperforms Bulk TR, which in turn outperforms the BVC.26” ([Ronen et al., 2022, p. 25](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=26&annotation=26NGALTG))

“In previous sections we found that in the bond market RF{5} and its bulk counterpart outperform the TR and BVC” ([Ronen et al., 2022, p. 28](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=29&annotation=MWJC7MUP))

“We train RF{5} and RF{7} over the first three days (using 842,494 observations) in the sample, test on the remaining days (434,910 transactions), and compute the variables required by RF{5} and RF{7}.” ([Ronen et al., 2022, p. 29](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=30&annotation=DF9L4548))

“The corresponding accuracy rates for RF{5}, LR, and TR are 74.6%, 74.0%, and 74.3%, respectively.31” ([Ronen et al., 2022, p. 30](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=31&annotation=CA24IS54))

“The combined results from our study, which spans a large range of trading frequencies, including high latency trades in the corporate bond markets (when interpreted alongside those in earlier papers) reveal that most classifiers’ accuracy is maximized when the number of daily trades is highest (with several local maxima in the equity market), when trade is fast, and when trade sizes are smaller.” ([Ronen et al., 2022, p. 31](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=32&annotation=G9RIFRB5))

“Our finding that trade frequency is positively related to RF{7} accuracy across all ranges is notable in light of recent literature on new trade classification algorithms (Easley et al. (2016), Panayides et al. (2019), Chakrabarty et al. (2015), and Carrion and Kolay (2021)), which generally suggest that as markets become increasingly faster (more trades per unit time), existing algorithms perform worse, exacerbating the need for new, more effective ones, that are more compatible with these faster markets. Thus, one clear advantage of the machine learning algorithm we propose is that it performs better as the number of trades increases” ([Ronen et al., 2022, p. 31](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=32&annotation=QGHPFQBU))

“Overall, the results in Table 13 as well as those in Section 4 reveal that RF{5} and RF{7} outperform LR and TR across all samples and for different types of trades” ([Ronen et al., 2022, p. 32](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=33&annotation=T2428C88))

“As an alternative to the ITCH trained RF{7}, we also test the ability of the TRACEtrained RF{5} to classify equity market trades and find that its accuracy in the ITCH data is an unimpressive 61.2%, trailing that of TR (71.7%). While it is disappointing that we could not train the model in one market and predict well in another, we harvested useful insights from the TRACE dataset regarding the relative efficacy of different machine learning and traditional trade direction classifiers. The TRACE trained models led us to RF{5} and ultimately RF{7} in the equity market. Despite the lackluster transfer efficacy of cross-market trained models, we can report that machine learning does improve accuracy when trained on in-market data.” ([Ronen et al., 2022, p. 32](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=33&annotation=5RTZ8ZW3))

“One caveat is that machine learning and other data driven methods achieve limited success with incomplete data. Since equity markets are fragmented, even if researchers obtain full sets of data, the inability to ensure correct sequencing of trades due to fragmentation and high frequency trading in the current equity markets presents problems for constructing machine learning models.” ([Ronen et al., 2022, p. 32](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=33&annotation=TCH89IX7))

“We fit models to a variety of algorithms and determined that a Random Forest model is the superior machine learning algorithm for this problem.” ([Ronen et al., 2022, p. 33](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=34&annotation=9SK9BI59))

“Looking at the specific features that improve forecasting accuracy we found the biggest improvements came from including the Tick Rule forecast and a variety of rolling price statistics. These features were important in both the stock and bond markets.” ([Ronen et al., 2022, p. 33](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=34&annotation=MRD2V4IP))

“Accuracy for smaller trades is higher than for larger trades. Our model is also sensitive to information as it performs best when price uncertainty is lower.” ([Ronen et al., 2022, p. 33](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=34&annotation=IJTRCWPJ))

“We find that we cannot train a model only on the data used to compute existing classification rules without human intervention. At least for the algorithms we employ, it appears that machine learning must be paired with intuition provided by the financial economist to improve the accuracy of traditional estimation methods.” ([Ronen et al., 2022, p. 34](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=35&annotation=SCVAHR85))

“Future work might include many other features that we did not investigate. Although we employed a number of machine learning algorithms for our models, methods that handle the time-series component of panel data in a more sophisticated way may yield even better results” ([Ronen et al., 2022, p. 34](zotero://select/library/items/9BA47YWD)) ([pdf](zotero://open-pdf/library/items/SK56ALN9?page=35&annotation=R6TR4TCU))