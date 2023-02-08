<mark style="background: #D2B3FFA6;">(What is the intuition?)</mark>

Like the previous two algorithms, the CLVN method ([[@chakrabartyTradeClassificationAlgorithms2012]] 3809) is a hybrid of the quote and tick rule and extends the EMO rule by a fragmented treatment of trades inside the quotes, which are notoriously hard to classify. ([[@chakrabartyTradeClassificationAlgorithms2012]] 3809) segment the spread into ten equal-width bins and classify trades around the midpoint (4th - 7th decile) by the tick rule and trades close to the quotes (1st-2nd, 9th-10th decile) by the quote rule. Like in the [[🔢EMO rule]] trades outside the quotes are categorized by the tick rule.
$$
tbd
$$
The algorithm, as summarized in Equation (...), is derived from a performance comparison of the tick rule / EMO rule against the quote rule / LR algorithm on stock data, whereby the accuracy was assessed separately for each decile.  The ([[@chakrabartyTradeClassificationAlgorithms2012]] 3811) The classical CLVN method uses the backward-looking tick rule. In the spirit of ([[@leeInferringTradeDirection1991]] 735), the tick test could be replaced for the reverse tick test.

([[@chakrabartyTradeClassificationAlgorithms2012]]) test their algorithm NASDAQ stocks traded at INET and ArcaEX modestly outperform the [[🔢Tick test]], [[🔢EMO rule]], and  [[🔢LR algorithm]] in terms of out-of-sample accuracy. The method has not yet been tested with option trades. Part of this might be due to the stronger reliance on the tick test, which has lead to a deteriorating performance in past studies (cp. [[@savickasInferringDirectionOption2003]]), ([[@grauerOptionTradeClassification2022]]).  

[[@chakrabartyTradeClassificationAlgorithms2007]] continue the trend for more complex classification rules, leading to a higher fragmented decision surface, and eventually resulting in a improved classification accuracy.  
Since the decision, which of the base rules is applied, is inferred from *static* cut-off points at the decile boundaries of the spread -- including the midspread and the quotes -- current classification rules may not realize their full potential. A obvious question is, if classifiers, *learned* on price and quote data, can adapt to the data and thereby improve over traditional trade classification rules. 

The trend towards sophisticated, hybrid rules, sometimes combining as many four base rules into a single classifier (see [[@grauerOptionTradeClassification2022]] 18), has conceptual parallels to (stacked) *ensembles* found in machine learning and expresses the need for better classifiers. In the subsequent section we provide an overview over state-of-the-art classification machine learning approaches for classification. We start by framing trade classification as a supervised learning problem. 

**Notes:**
[[🔢CLNV method notes]]