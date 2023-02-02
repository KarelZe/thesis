
**Machine learning:**

- Some feature sets contain as many as 42 features, some of which a forward looking (cp. [[@ronenMachineLearningTrade2022]] (p. 49)).

- [[@rosenthalModelingTradeDirection2012]] writes that he uses a logistic-link generalized linear model (GLM). Obviously a fancy way of saying (in R wording) that he uses logistic regression. (see here. https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/R/R-Manual/R-Manual20.html)

- Parts of this may be due to the are bound by the best performance of their commitee members. Voting classifiers are bound to the of their ensemble members. Voting classifier / ensemble


- Seldomly used but ML-like. Would probably be sufficient to cover it under related works.
- [[@ronenMachineLearningTrade2022]] / [[@fedeniaMachineLearningCorporate2021]] They employ a machine learning-based approach for trade side classification. Selection of method follows no clear research agenda, so does sample selection or tuning. Also leaves out latest advancements in prediction of tabular data such as GBM or dedicated NN architectures. Data set only spans two days? General saying ML based predictor (random forest) outperforms tick rule and BVC. Still much human inutition is required for feature engineering. Treated as **supervised tasks**. More recent approaches and also ML approaches outperform classical approaches due to a higher trading frequency. Transfer learning not successful. **Note:** Tick rule has been among the poorest predictors in Grauer. **Note:** Check what the actual difference between the two papers are....
- We find serious flaws in their research.
- Which works performed trade side classification for stocks, for options or other products.
- Also the number of features is not the same...
- [[@rosenthalModelingTradeDirection2012]] incorporates different methods into a model for the likelihood a trade was buyer-initiated. It's a simple logistic regresssion. Performed on stocks. 
- [[@blazejewskiLocalNonparametricModel2005]] compare $k$-nn and logistic regression for trade-side classification. Performed for Australian stocks. Unclear how results compare to classical rules. 

**Option data set:**
- Do not compare accuracies across different datasets. This won't work. Might mention [[@grauerOptionTradeClassification2022]] as it is calculated on (partly) the same data set.
- Results were very different for the option markets between the studies. Compare the frequency some literature (in the stock market) suggest, that  for higher frequencies classical approaches like the tick test deteriorate.
> Easley, O’Hara, and Srinivas (1998) use the Lee and Ready approach to test their game theoretic model of informed trading in stock and option markets. It is, therefore, important to determine whether the application of stock trade classification rules to derivatives is valid. [[@savickasInferringDirectionOption2003]] Don't propose their own rules. 

**Trade classification rules**
- see e. g., [[@frommelAccuracyTradeClassification2021]] for a recent overview on datasets / Methods. Interestingly the call the CLNV method, MEMO for modified EMO
- For classical rule-based approaches see some citations in [[@olbrysEvaluatingTradeSide2018]]. E. g., [[@chakrabartyTradeClassificationAlgorithms2012]] or [[@chakrabartyEvaluatingTradeClassification2015]]



**Machine learning:**
<mark style="background: #BBFABBA6;">“The k-nearest neighbor with three predictor variables achieves an average out-of-sample classification accuracy of 71.40%, compared to 63.32% for the linear logistic regression with seven predictor variables.” </mark>([Blazejewski and Coggins, 2005, p. 481](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=1&annotation=SKICD63H))

“<mark style="background: #FFB8EBA6;">The result suggests that a non-linear approach may produce a more parsimonious trade sign inference model with a higher out-of-sample classification accuracy.</mark>” ([Blazejewski and Coggins, 2005, p. 481](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=1&annotation=I9P2NWE9))

<mark style="background: #BBFABBA6;">“(1) Among the k-NN classifiers, the higher the value of k the greater the mean accuracy. The difference between accuracies for k ¼ 9 and 5, however, can be minimal and sometimes negative, but on average k ¼ 9 is the best (12). (2) The mean accuracy of the k-NN classifier, where k ¼ 9; is a monotonically increasing function of the training interval length. The rate of the increase, however, rapidly declines. Small, negligible fluctuations are sometimes present (10). (3) The mean accuracy of the k-NN classifier, where k ¼ 9; is greater than the mean accuracy of the logistic regression classifier for all training timescales (8).</mark>” ([Blazejewski and Coggins, 2005, p. 491](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=11&annotation=PEISTE82))

“<mark style="background: #ADCCFFA6;">The mean accuracy of the k-NN classifier, where k ¼ 9; is greater than the mean accuracies of the trade continuation and the majority vote classifiers, for all training timescales (12). The total number of models constructed for each stock was 145: 2 3 16 k-NN, 2 16 logistic regression, 1 trade continuation, and 16 majority vote models. </mark>” ([Blazejewski and Coggins, 2005, p. 493](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=13&annotation=TTJGZ7EW))

<mark style="background: #ADCCFFA6;">“These results suggest that a non-linear approach may produce a more parsimonious trade sign inference model with a higher out-of-sample classification accuracy. Furthermore, for most of our stocks the classification accuracy of the k-nearest-neighbor ðk ¼ 9Þ with contemporaneous predictor variables is a monotonically increasing function of the training interval length, with 30 days being the best interval.”</mark> ([Blazejewski and Coggins, 2005, p. 494](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=14&annotation=MCA94DNA))


**Unused:**
- Similarly, [[@aitkenIntradayAnalysisProbability1995]] perform trade side classification with logistic regression. Similar, but not equal as they predict whether a trade is at the ask or at the bid, not how initiated the trade
