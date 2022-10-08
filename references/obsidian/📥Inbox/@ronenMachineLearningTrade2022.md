---
title: Machine Learning and Trade Direction Classification: Insights Fromthe Corporate Bond Market
authors: Tavy Ronen, Mark A. Fedenia, Seunghan Nam
year: 2021
---

“As trade initiation information is generally not provided in most intraday transaction databases, trade direction is often inferred from local price and/or quote behavior,” (Ronen et al., 2022, p. 2)

“The efficacy of these rules has been hotly debated, and their relative accuracy in different markets has been studied extensively.” (Ronen et al., 2022, p. 2)

“Our paper contributes to the market microstructure literature by examining the applicability of machine learning methods to better understand and improve trade direction classification. In particular, we introduce a machine learning model that outperforms traditional classifiers. Moreover, we illustrate the importance of optimizing the feature set in correctly classifying trade direction and provide new insights on the efficacy of trading rules in different market conditions.” (Ronen et al., 2022, p. 3)

“We find that machine learning models offer trade classification accuracy that at least matches and often exceeds the accuracy of the most commonly used models in the bond and stock markets.” (Ronen et al., 2022, p. 4)

“The Random Forest (RF) algorithm is found to be superior to a number of other machine learning choices as well as to traditional classifiers. The combination of additional features and a more flexible form of the classification function improves classification accuracy. For example, improvements over the tick rule (TR) are roughly 8.3%.” (Ronen et al., 2022, p. 4)

“The accuracy of trade direction predictors depends on the trading and information environment in the market at the time of the trade. Consistent with Ellis et al. (2000), Finucane (2000), Chakrabarty et al. (2007), and Chakrabarty et al. (2012) who find that trade direction classifier accuracy is generally lower for larger trades in the equity markets, we find that the accuracy for smaller trades seem to be higher than for larger ones.” (Ronen et al., 2022, p. 4)

“Indeed, the positive correlation between the RF algorithm and trade frequency across all ranges renders it more suitable for today’s faster markets, particularly in light of recent papers on new trade classification algorithms (Easley et al. (2016), Panayides et al. (2019), Chakrabarty et al. (2015), and Carrion and Kolay (2021)), which call in to question the ability of traditional trade direction classifiers to achieve high accuracy rates as the number of trades per unit time increases.” (Ronen et al., 2022, p. 6)

“By the same token, we admit that a myriad of models and combinations of models (ensemble models) are not explored. Our purpose here is not to identify the best prediction model possible. Instead, our goal is to see if an inductive approach can deliver insights into a classic problem in market microstructure” (Ronen et al., 2022, p. 10)

“We examine all corporate bonds disseminated on the Enhanced Trade Reporting and Compliance Engine (TRACE) between July 1, 2002 and December 31, 2019.” (Ronen et al., 2022, p. 11)

“We consider several algorithms including Decision Trees, Discriminant Analysis, Logistic Regression, Support Vector Machines, k-Nearest Neighbor and a Neural Network (NN)” (Ronen et al., 2022, p. 13)

“We explicitly report results for RF, NN, and Logistic Regression. However, we focus our attention on RF as it is the dominant performer” (Ronen et al., 2022, p. 14)

“RF was implemented with 100 trees and increasing that number did little to improve out-of-sample accuracy. We increased the number of leaves in a tree and varied the pruning parameters. These changes did not affect the accuracy significantly. Additionally, we employed 10-Fold cross-validation to check if our 70/30 random partition is appropriate. In 10-Fold cross-validation, we divide the sample into ten equal sizes and ten iterations of Random Forest using 90/10 partition are performed.” (Ronen et al., 2022, p. 14)

“The algorithms we employ do not efficiently incorporate lagged information. Therefore we include relevant lagged data in each observation.” (Ronen et al., 2022, p. 15)

“We apply the classification algorithms to seven feature sets labeled {1} - {7} described below and shown in Table 4.” (Ronen et al., 2022, p. 15)

“We find that a Neural Network performs on par with most other decision tree models (not reported) except for the Random Forest. Logistic Regression performs about as well as TR (the most widely used classifier in markets without pre-trade transparency) but not as well as either a Neural Network or Random Forest. As mentioned earlier, the Random Forest approach (Breiman (2001) has an advantage over other algorithms we tried partly because it is an ensemble method.” (Ronen et al., 2022, p. 16)

“Choices regarding the rolling window periods and percentages of training and testing batches are somewhat ad-hoc by nature. Our results are robust to all other choices made regarding the testing-to-training ratio (85%/15%, and 60%/40%), as well as to fixing the ratios to the first and last data points (as opposed to using random batches that maintain these ratios). In addition, we consider one and two year rolling windows and obtain similar results.” (Ronen et al., 2022, p. 16)

“We now examine how well the models shown in Table 5 hold up out-of-sample” (Ronen et al., 2022, p. 17)

“Compared to out-of-bag accuracy levels shown in Table 5, the out of sample prediction rates are mildly lower, although the ordinal properties across model accuracy rates remains similar. RF{4} still outperforms NN{4}, LG{4}, and TR. Since RF{4} and RF{5} are effectively equally accurate, and since RF{5} is more universally applicable in both the bond and stock markets, we focus on RF{5} and de-emphasize RF{4} going forward.” (Ronen et al., 2022, p. 17)

“Table 7 reports the incremental out-of-sample prediction accuracy provided by each of the feature sets described in Table 4.” (Ronen et al., 2022, p. 18)

“As the time elapsed since the last trade decreases, accuracy rates for model RF{5} and TR improve. Prediction rates for RF{5} (TR) range from 71.57% (68.05%) in decile 1 to 81.43% (72.33%) in decile 10.” (Ronen et al., 2022, p. 19)

“Bulk RF{5}.25 Table 11 reports that across all bars (volume, time, and trade), Bulk RF{5} outperforms Bulk TR, which in turn outperforms the BVC.26” (Ronen et al., 2022, p. 25)

“In previous sections we found that in the bond market RF{5} and its bulk counterpart outperform the TR and BVC” (Ronen et al., 2022, p. 28)

“The corresponding accuracy rates for RF{5}, LR, and TR are 74.6%, 74.0%, and 74.3%, respectively.31” (Ronen et al., 2022, p. 30)

“Our finding that trade frequency is positively related to RF{7} accuracy across all ranges is notable in light of recent literature on new trade classification algorithms (Easley et al. (2016), Panayides et al. (2019), Chakrabarty et al. (2015), and Carrion and Kolay (2021)), which generally suggest that as markets become increasingly faster (more trades per unit time), existing algorithms perform worse, exacerbating the need for new, more effective ones, that are more compatible with these faster markets. Thus, one clear advantage of the machine learning algorithm we propose is that it performs better as the number of trades increases” (Ronen et al., 2022, p. 31)

“As an alternative to the ITCH trained RF{7}, we also test the ability of the TRACEtrained RF{5} to classify equity market trades and find that its accuracy in the ITCH data is an unimpressive 61.2%, trailing that of TR (71.7%). While it is disappointing that we could not train the model in one market and predict well in another, we harvested useful insights from the TRACE dataset regarding the relative efficacy of different machine learning and traditional trade direction classifiers. The TRACE trained models led us to RF{5} and ultimately RF{7} in the equity market. Despite the lackluster transfer efficacy of cross-market trained models, we can report that machine learning does improve accuracy when trained on in-market data.” (Ronen et al., 2022, p. 32)

“One caveat is that machine learning and other data driven methods achieve limited success with incomplete data. Since equity markets are fragmented, even if researchers obtain full sets of data, the inability to ensure correct sequencing of trades due to fragmentation and high frequency trading in the current equity markets presents problems for constructing machine learning models.” (Ronen et al., 2022, p. 32)

“Looking at the specific features that improve forecasting accuracy we found the biggest improvements came from including the Tick Rule forecast and a variety of rolling price statistics. These features were important in both the stock and bond markets.” (Ronen et al., 2022, p. 33)

“Accuracy for smaller trades is higher than for larger trades. Our model is also sensitive to information as it performs best when price uncertainty is lower.” (Ronen et al., 2022, p. 33)

“We find that we cannot train a model only on the data used to compute existing classification rules without human intervention. At least for the algorithms we employ, it appears that machine learning must be paired with intuition provided by the financial economist to improve the accuracy of traditional estimation methods.” (Ronen et al., 2022, p. 34)

“Future work might include many other features that we did not investigate. Although we employed a number of machine learning algorithms for our models, methods that handle the time-series component of panel data in a more sophisticated way may yield even better results” (Ronen et al., 2022, p. 34)