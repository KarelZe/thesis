Tags: #trade-classification 

**Notes:**
- proposed on the CBOE and ISE dataset
- Alternative to handle midspread trades, that can not be classified using the quote rule.
- In the spirit of stacked ensembles the depth rule needs to be combined with other methods
**motivation:**
- “We hypothesize that a larger bid or ask quoted size, i.e., a higher depth at the best bid or ask, indicates a higher liquidity similar to a tighter bid or ask quote” ([[@grauerOptionTradeClassification2022]]), p. 14)
- “As a consequence, we classify midspread trades as buyer-initiated, if the ask size exceeds the bid size, and as seller-initiated, if the bid size is higher than the ask size. If the ask size matches the bid size, midspread trades still cannot be classified by this approach” ([[@grauerOptionTradeClassification2022]]), p. 14)

**limitation:** 👩‍🚒 only a proxy for tick rule. Must be combined with other rules.

**results:** 💸
- Improves LR algorithm by 0.8 %. Overall accuracy 75 %. ([[@grauerOptionTradeClassification2022]]), p. 4)
- “Applying our depth rule after using the trade size rule and the quote rule and classifying the very small number of midspread trades that cannot be signed by our depth rule using the reverse tick test yields an additional improvement of 1.2% on average. It improves the success rate of the quote rule applied to CBOE quotes first and then to the NBBO to 73.37 %. These results confirm that our depth rule outperforms the standard and reverse tick test in classifying midspread trades.” ([[@grauerOptionTradeClassification2022]]), p. 18)
- “Most importantly, in contrast to standard and reverse tick tests, our newly proposed depth rule leads to a significant improvement compared to using the quote rule alone, pointing to its superior performance to sign midspread trades that quote rules cannot classify.” ([[@grauerOptionTradeClassification2022]]), p. 21)