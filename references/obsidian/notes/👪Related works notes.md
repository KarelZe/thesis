
**Machine learning:**
- Seldomly used but ML-like. Would probably be sufficient to cover it under related works.
- [[@ronenMachineLearningTrade2022]] / [[@fedeniaMachineLearningCorporate2021]] They employ a machine learning-based approach for trade side classification. Selection of method follows no clear research agenda, so does sample selection or tuning. Also leaves out latest advancements in prediction of tabular data such as GBM or dedicated NN architectures. Data set only spans two days? General saying ML based predictor (random forest) outperforms tick rule and BVC. Still much human inutition is required for feature engineering. Treated as **supervised tasks**. More recent approaches and also ML approaches outperform classical approaches due to a higher trading frequency. Transfer learning not successful. **Note:** Tick rule has been among the poorest predictors in Grauer. **Note:** Check what the actual difference between the two papers are....
- We find serious flaws in their research.
- Which works performed trade side classification for stocks, for options or other products.
- Also the number of features is not the same...
- [[@rosenthalModelingTradeDirection2012]] incorporates different methods into a model for the likelihood a trade was buyer-initiated. It's a simple logistic regresssion. Performed on stocks. 
- [[@blazejewskiLocalNonparametricModel2005]] compare $k$-nn and logistic regression for trade-side classification. Performed for Australian stocks. Unclear how results compare to classical rules. 
- Similarily, [[@aitkenIntradayAnalysisProbability1995]] perform trade side classification with logistic regression.

**Option data set:**
- Do not compare accuracies across different datasets. This won't work. Might mention [[@grauerOptionTradeClassification2022]] as it is calculated on (partly) the same data set.
- Results were very different for the option markets between the studies. Compare the frequency some literature (in the stock market) suggest, that  for higher frequencies classical approaches like the tick test deteriorate.
> Easley, O’Hara, and Srinivas (1998) use the Lee and Ready approach to test their game theoretic model of informed trading in stock and option markets. It is, therefore, important to determine whether the application of stock trade classification rules to derivatives is valid. [[@savickasInferringDirectionOption2003]] Don't propose their own rules. 

**Trade classification rules**
- see e. g., [[@frommelAccuracyTradeClassification2021]] for a recent overview on datasets / Methods. Interestingly the call the CLNV method, MEMO for modified EMO
- For classical rule-based approaches see some citations in [[@olbrysEvaluatingTradeSide2018]]. E. g., [[@chakrabartyTradeClassificationAlgorithms2012]] or [[@chakrabartyEvaluatingTradeClassification2015]]


