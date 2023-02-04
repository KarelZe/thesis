[[@grauerOptionTradeClassification2022]] promote using trade size information to improve the classification performance of midspread trades. In their *depth rule*, they infer the trade initiator from the bid's depth and ask for quotes. Based on the observation that an exceeding bid or ask quote relates to higher liquidity on one side, trades are classified as buyer-iniated for a larger ask size and seller-iniated for a higher bid size.

As shown in Algorithm 2, the depth rule classifies midspread trades only, if the ask size is different from the bid size, as the ratio between the ask and bid size is the sole criterion for assigning the initiator. To sign the remaining trades, other rules must follow thereafter.

In a similar vain the subsequent *trade size rule* utilizes the ask and bid quote size to improve classification performance.

**Notes:**
- proposed on the CBOE and ISE dataset
- Alternative to handle midspread trades, that can not be classified using the quote rule.
- In the spirit of stacked ensembles the depth rule needs to be combined with other methods
- “We hypothesize that a larger bid or ask quoted size, i.e., a higher depth at the best bid or ask, indicates a higher liquidity similar to a tighter bid or ask quote” ([[@grauerOptionTradeClassification2022]]), p. 14)
- trade size matches exactly either the bid or ask quote size, it is likely that the quote came from a customer, the market maker found it attractive and, therefore, decided to fill it completely. (see [[@grauerOptionTradeClassification2022]])
- “As a consequence, we classify midspread trades as buyer-initiated, if the ask size exceeds the bid size, and as seller-initiated, if the bid size is higher than the ask size. If the ask size matches the bid size, midspread trades still cannot be classified by this approach” ([[@grauerOptionTradeClassification2022]]), p. 14)
- Improves LR algorithm by 0.8 %. Overall accuracy 75 %. ([[@grauerOptionTradeClassification2022]]), p. 4)
- “Applying our depth rule after using the trade size rule and the quote rule and classifying the very small number of midspread trades that cannot be signed by our depth rule using the reverse tick test yields an additional improvement of 1.2% on average. It improves the success rate of the quote rule applied to CBOE quotes first and then to the NBBO to 73.37 %. These results confirm that our depth rule outperforms the standard and reverse tick test in classifying midspread trades.” ([[@grauerOptionTradeClassification2022]]), p. 18)
- “Most importantly, in contrast to standard and reverse tick tests, our newly proposed depth rule leads to a significant improvement compared to using the quote rule alone, pointing to its superior performance to sign midspread trades that quote rules cannot classify.” ([[@grauerOptionTradeClassification2022]]), p. 21)