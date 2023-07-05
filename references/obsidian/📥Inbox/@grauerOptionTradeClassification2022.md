
title: Option Trade Classification
authors: Caroline Grauer, Philipp Schuster, Marliese Uhrig-Homburg
year: 2021
tags:  #rule-based #trade-classification 
status : #📦 
related:
- [[@savickasInferringDirectionOption2003]]


## Key takeaways
- Getting option-trade classification right is central for several applications like the calculation of order imbalances. One Example is: [[@garleanuDemandBasedOptionPricing2009]]
- Authors use quote, tick, Lee and Ready (1991), and Ellis, Michaely, and O’Hara (2000) rule to infer the trade direction of option trades. Also suggest two own rules” (Grauer et al., 2022, p. 0)
- New rules:
	1. **Trade size rule:** trade size matches exactly either the bid or ask quote size, it is likely that the quote came from a customer, the market maker found it attractive and, therefore, decided to fill it completely.  Accuracy of 80 %. 
	2. **Depth rule:** classify midspread trades based on the comparison of bid and ask quoted depths. Improves LR algorithm by 0.8 %. Overall accuracy 75 %.
	3. **New: Depth rule:💥**  As a consequence, we classify midspread trades as buyer-initiated, if the ask size exceeds the bid size, and as seller-initiated, if the bid size is higher than the ask size.
	4. **New: Trade size rule:💥** classify trades for which the trade size is equal to the quoted bid size as customer buys and those with a trade size equal to the ask size as customer sells.
- “the prevailing LR algorithm is only able to correctly sign between 60% to 64% of option trades, which is a similar magnitude as using the quote rule alone.”  (Grauer et al., 2022, p. 0)
- midspread trades are particularly difficult to classify, which leads to the poor performance of the LR and EMO rule compared to the quote rule. (Grauer et al., 2022, p. 4)
- “The general finding from this literature is that the overall success of classification rules for stock markets is relatively high, but varies widely across security markets and time periods.” (Grauer et al., 2022, p. 5)
- Reasons why application of stock rules is doubtfull are: “options are more illiquid, trading is spread accross different exchange with nation-wide bid offers” (Grauer et al., 2022, p. 1)
- Only one similar work (Savickas and Wilson (2003)).” (Grauer et al., 2022, p. 1)
- “Generally, quote rules outperform tick rules by far.” (Grauer et al., 2022, p. 3)
- **Novelty:** 💥 “The highest success rate of 63.92% can be achieved by applying the quote rule first to NBBO and then to ISE quotes, and classifying all remaining trades using the reverse tick rule” (Grauer et al., 2022, p. 3)
- "Authors hypothesise, that weak performance that sophisticated customers placing limit order instead of market orders cause the poor performance"

## Importance of option trade classifcation
- “Particularly, the trade direction is required to determine the information content of trades, the order imbalance and inventory accumulation of liquidity providers, the price impact of transactions, and to calculate many liquidity measures.” (Grauer et al., 2022, p. 1)

## Data and Inference of true trade side
- “LiveVol provides intraday transaction-level option data for all option trades on all U.S. exchanges.” (Grauer et al., 2022, p. 7)
- “We philtre out option trades with a trading price less than or equal to zero. We also remove trades with negative or zero volume and those whose trading volume exceeds 10 million contracts. Furthermore, we delete entries with multiple underlying symbols for the same root and other duplicates along with any cancelled trades.” (Grauer et al., 2022, p. 7)
- "We end up with eight categories covering buy and sell volumes for each of the four trader types by option series and trading day." (Grauer et al., 2022, p. 7)
- ISE sample contains twelve-year period from May 2, 2005 to May 31, 2017. (Grauer et al., 2022, p. 7)
- The matched ISE sample contains 49,203,747 option trades. (Grauer et al., 2022, p. 7)
- **Labelling:** “We take advantage of the fact that if there were only customer buy (sell) orders on a specific day for a given option series at one particular exchange, Open/Close data allows to classify all transactions in the LiveVol dataset on that day at the respective exchange as buy (sell) orders.” (Grauer et al., 2022, p. 8)
-  “We use the unique key specified by trade date, expiration date, strike price, option type, and root symbol of the underlying to match the samples.” (Grauer et al., 2022, p. 8)

## Classical rules

- **Quote rules:** “If the trade occurs above the midpoint of the bid-ask spread, it is classified as buyer-initiated. Conversely, if the trade price is below the midspread, the trade is classified as seller-initiated. Trades that occur exactly at the midpoint **cannot** be classified.” (Grauer et al., 2022, p. 10)

- **Tick tests** use changes in trade prices and look at previous trade prices to infer trade direction. If the trade occurs at a higher price than the previous one, it is classified as buyer-initiated. Conversely, if the trade price is below the previous one, it is classified as seller-initiated. If there is no price change between successive trades, the trade direction is inferred using the last price that differs from the current price.” (Grauer et al., 2022, p. 10)
- “There are hybrid combinations like the Lee and Ready (LR) or the Ellis, Michaely and O'Hara (EMO). LR uses quote rule first, remainder with tick test. EMO uses quote rule only for trades, where the trade price matches the bid or ask price and tick rule for remainder” (Grauer et al., 2022, p. 11)

## Results  of classical rules
- Applying LR and EMO algorithms to NBBO (Nation-wide price) quotes and price information across all exchanges as well as using subsequent trade prices to infer trade direction yields higher success rates.
- **Tick test negatively impacts results.** “Moreover, we find that the LR algorithm outperforms the EMO rule as, in addition to midspread trades, the latter uses the tick test to a greater extent. However, the commonly used LR rule using the tick test to classify midspread trades is only able to classify 63.53% of trades correctly, which is **worse** than using the quote rule alone” (Grauer et al., 2022, p. 12)
- **Trade size negatively impacts results.** With trade sizes equal to either the bid quote size or the ask quote size at the ISE at the time of the trade.” (Grauer et al., 2022, p. 12)

## New rules

- **Intuition for trade size rule:** When trade size is equal to either the size of the ask or the bid quote is due to limit orders placed by sophisticated customers. (Grauer et al., 2022, p. 13)
- **Trade size rule:** classify trades for which the trade size is equal to the quoted bid size as customer buys and those with a trade size equal to the ask size as customer sells.
- “After applying this “trade size rule”, the existing trade classification algorithms are applied to all other trades for which the trade size is not equal to one of the quote sizes (or for which it is equal to both the bid and the ask size). Panel A of Table 4 shows that this modification leads to a substantial improvement between 10.7% and 11.3% in the performance of the quote rule and combined methods and an improvement of 5.6% to 7.3% for the tick tests.” (Grauer et al., 2022, p. 13)

- **Intuition for depth rule:** “We hypothesise that a larger bid or ask quoted size, i.e., a higher depth at the best bid or ask, indicates a higher liquidity similar to a tighter bid or ask quote” (Grauer et al., 2022, p. 14)
- **Depth rule:** As a consequence, we classify midspread trades as buyer-initiated, if the ask size exceeds the bid size, and as seller-initiated, if the bid size is higher than the ask size.
- Applying our proposed “depth rule” after using the trade size rule and quote rules improves the performance by around 0.8%.
- “We show the overall success rates of the classification algorithms using our trade size rule and also calculate the change in the success rates compared to the same algorithms not using the trade size rule in parentheses. The results show that our new rule works best for small to medium-sized trades and even leads to a slight deterioration of the performance for the largest trade sizes.” (Grauer et al., 2022, p. 15)
- “Based on our findings so far, we recommend that researchers use our new trade size rule together with quote rules successively applied to NBBO and quotes on the trading venue. Quotes at the midpoint on both the NBBO and the exchange should be classified first with the depth rule and any remaining trades with the reverse tick test. Most importantly, the LR algorithm alone, which is heavily used in the literature (see, e.g., Pan and Poteshman (2006); Hu (2014); Easley, O’Hara, and Srinivas (1998)), does a poor job to identify buy and sell orders in option trade data.8 Overall, the accuracy of all common classification algorithms to infer option trade direction can be significantly improved by our two new rules” (Grauer et al., 2023, p. 15)

## Out-of-sample-tests
- “Namely, tick tests perform best when using most current price information across all exchanges and reverse tick tests based on subsequent prices dominate their counterparts based on preceding ones.” (Grauer et al., 2022, p. 16)
- "However,** tick tests** perform significantly **worse** than quote rules and are only able to **correctly classify** slightly more than 50% of option trades, which is not much better than a random allocation of buys and sells."
- “For this reason, the LR algorithm outperforms the EMO rule as the former uses the **tick tests** to a **smaller extent.**” (Grauer et al., 2022, p. 16)
- Weak performance is once again mainly driven by trades with trade sizes equal to either the bid quote size or the ask quote size at the CBOE at the time of the trade. For them, average success rates of **quote rules** are only about **19%**. Because our trade size rule addresses exactly these trades, this is a promising first indication that our new rule also works for the CBOE sample. (Grauer et al., 2022, p. 17)

- “The highest success rate of 72.40% is achieved by the **trade size rule** in combination with the **quote rule** first applied to CBOE quotes and then to the NBBO. In combination with the tick test or reverse tick test for the LR algorithm, we are able to correctly classify 72.12% and 72.39% of the option trades, respectively.” (Grauer et al., 2022, p. 18)

- “Applying our **depth rule** after using the **trade size rule** and the **quote rule** and classifying the very small number of midspread trades that cannot be signed by our depth rule using the reverse tick test yields an additional improvement of 1.2% on average. It improves the success rate of the quote rule applied to CBOE quotes first and then to the NBBO to 73.37%. These results confirm that our depth rule outperforms the standard and reverse tick test in classifying midspread trades.” (Grauer et al., 2022, p. 18)

## Robustness
- “Finally, to compare the performance of the algorithms over time, we look at the **individual years** of our sample period. To conserve space, we compute average success rates for the different specifications of the quote, tick, LR, reverse LR, EMO, and depth rules.” (Grauer et al., 2022, p. 19)
- **Low success rate on index options:** “Comparing the classification precision of options written on common stocks, index options, and options written on other underlyings (mainly ETFs), we find **lower success rates** for **index options**, which is consistent with Savickas and Wilson (2003). Interestingly, the improvements due to our trade size rule are particularly high for index options at the CBOE.” (Grauer et al., 2022, p. 20)
- "Summarising the results from our robustness cheques, we find that in all subsamples and for **all existing trade classification algorithms**, improvements due to the application of our new trade size rule are positive and range between 1% and 23%."(Grauer et al., 2022, p. 20)
- “Most importantly, in contrast to standard and reverse tick tests, our newly proposed **depth rule leads to a significant improvement** compared to using the quote rule alone, pointing to its superior performance to sign midspread trades that quote rules cannot classify.” (Grauer et al., 2022, p. 21)

## Conclusion
- “Using our new methodology allows to correctly classify between 73% and 75% of option trades in our sample, which is more than 10% higher compared to the rules that are currently used in the literature.” (Grauer et al., 2022, p. 22)
