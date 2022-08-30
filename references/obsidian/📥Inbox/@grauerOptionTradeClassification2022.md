---
title: Option Trade Classification
authors: Caroline Grauer, Philipp Schuster, Marliese Uhrig-Homburg
year: 2021
tags: #option-trade-classification #rule-based
---

“We evaluate the performance of common stock trade classification algorithms including the quote, tick, Lee and Ready (1991), and Ellis, Michaely, and O’Hara (2000) rule to infer the trade direction of option trades.” (Grauer et al., 2022, p. 0)

“the prevailing Lee and Ready algorithm is only able to correctly sign between 60% to 64% of option trades, which is a similar magnitude as using the quote rule alone.” (Grauer et al., 2022, p. 0)

“Particularly, the trade direction is required to determine the information content of trades, the order imbalance and inventory accumulation of liquidity providers, the price impact of transactions, and to calculate many liquidity measures.” (Grauer et al., 2022, p. 1)

“First, options are much more illiquid than stocks with many series not recording a trade for days or weeks. For that reason, tick rules that depend on the information from preceding or succeeding trades might be problematic.” (Grauer et al., 2022, p. 1)

“Against this backdrop, it is surprising that there is just one study comparing trade classification rules in option markets, which is conducted on a small and more than twenty-five year old dataset (Savickas and Wilson (2003)).” (Grauer et al., 2022, p. 1)

“The document is available via the following link: https://osf.io/kj86r/ ?view_only=388a89b23254425a8271402e2b11fc4e.” (Grauer et al., 2022, p. 2)

“Generally, quote rules outperform tick rules by far.” (Grauer et al., 2022, p. 3)

“The highest success rate of 63.92% can be achieved by applying the quote rule first to NBBO and then to ISE quotes, and classifying all remaining trades using the reverse tick rule” (Grauer et al., 2022, p. 3)

“Overall, the accuracy of existing classification methods is considerably lower for option trades than for stocks, which is mostly between 70% and 90%” (Grauer et al., 2022, p. 3)

“main idea of our new “trade size rule” is that when the trade size matches exactly either the bid or ask quote size, it is likely that the quote came from a customer, the market maker found it attractive and, therefore, decided to fill it completely.” (Grauer et al., 2022, p. 4)

“The hypothesis of market makers filling the limit orders of customers seems most plausible for relatively small orders and trades that are not outside of the bid ask spread (for them, it is likely that customers submitted a market order exceeding the prevailing bid or ask quote size)” (Grauer et al., 2022, p. 4)

“Our second improvement addresses the fact that midspread trades are particularly difficult to classify, which leads to the poor performance of the LR and EMO rule compared to the quote rule.” (Grauer et al., 2022, p. 4)

“The general finding from this literature is that the overall success of classification rules for stock markets is relatively high, but varies widely across security markets and time periods.” (Grauer et al., 2022, p. 5)

“To the best of our knowledge, Savickas and Wilson (2003) provide the only study that examines the trade classification accuracy for option trades.” (Grauer et al., 2022, p. 6)

“Based on this mechanism and the poor performance of the tick test, we propose two simple rules that can be used in combination with existing classification algorithms” (Grauer et al., 2022, p. 6)

“LiveVol provides intraday transaction-level option data for all option trades on all U.S. exchanges.” (Grauer et al., 2022, p. 7)

“We filter out option trades with a trading price less than or equal to zero. We also remove trades with negative or zero volume and those whose trading volume exceeds 10 million contracts. Furthermore, we delete entries with multiple underlying symbols for the same root and other duplicates along with any cancelled trades.” (Grauer et al., 2022, p. 7)

“Because evaluating the performance of trade classification algorithms requires information on the true side of the trade, we combine information from intraday transaction data and daily Open/Close data to arrive at such a benchmark. Our two Open/Close datasets are available on a daily level and cover trading volume at the ISE and the CBOE, respectively.” (Grauer et al., 2022, p. 8)

“We take advantage of the fact that if there were only customer buy (sell) orders on a specific day for a given option series at one particular exchange, Open/Close data allows to classify all transactions in the LiveVol dataset on that day at the respective exchange as buy (sell) orders.” (Grauer et al., 2022, p. 8)

“We use the unique key specified by trade date, expiration date, strike price, option type, and root symbol of the underlying to match the samples.” (Grauer et al., 2022, p. 8)

“with the OSF (see footnote 2)” (Grauer et al., 2022, p. 9)

“As the probability of observing only buy trades or only sell trades decreases with an increasing number of trades, the number of trades per option day is lower and the time between two trades is higher in our matched samples compared to their full sample equivalents.” (Grauer et al., 2022, p. 9)

“Because most classification rules have a lower performance for illiquid securities, our results can be interpreted as a lower boundary on their overall performance.” (Grauer et al., 2022, p. 9)

“If the trade occurs above the midpoint of the bid-ask spread, it is classified as buyer-initiated. Conversely, if the trade price is below the midspread, the trade is classified as seller-initiated. Trades that occur exactly at the midpoint cannot be classified.” (Grauer et al., 2022, p. 10)

“Second, tick tests use changes in trade prices and look at previous trade prices to infer trade direction. If the trade occurs at a higher price than the previous one, it is classified as buyer-initiated. Conversely, if the trade price is below the previous one, it is classified as seller-initiated. If there is no price change between successive trades, the trade direction is inferred using the last price that differs from the current price.” (Grauer et al., 2022, p. 10)

“Conversely, if the next distinguishable price is above the current price, the current trade is classified as seller-initiated. The tick test and reverse tick test can be applied using trade prices on all option exchanges or one specific exchange only” (Grauer et al., 2022, p. 10)

“To make the performance of algorithms that are unable to completely classify all trades comparable, we assume unclassified trades to be correctly classified with a random probability of 50%.” (Grauer et al., 2022, p. 11)

“This affects quote rules only, as they are unable to classify midspread trades.” (Grauer et al., 2022, p. 11)

“Moreover, we find that the LR algorithm outperforms the EMO rule as, in addition to midspread trades, the latter uses the tick test to a greater extent. However, the commonly used LR rule using the tick test to classify midspread trades is only able to classify 63.53% of trades correctly, which is worse than using the quote rule alone” (Grauer et al., 2022, p. 12)

“The last two columns of Table 3 show that the weak performance is mainly driven by trades with trade sizes equal to either the bid quote size or the ask quote size at the ISE at the time of the trade.” (Grauer et al., 2022, p. 12)

“We start with the hypothesis that the weak performance of existing trade classification methods for trades with a trade size equal to either the size of the ask or the bid quote is due to limit orders placed by sophisticated customers.” (Grauer et al., 2022, p. 13)

“After applying this “trade size rule”, the existing trade classification algorithms are applied to all other trades for which the trade size is not equal to one of the quote sizes (or for which it is equal to both the bid and the ask size). Panel A of Table 4 shows that this modification leads to a substantial improvement between 10.7% and 11.3% in the performance of the quote rule and combined methods and an improvement of 5.6% to 7.3% for the tick tests.” (Grauer et al., 2022, p. 13)

“We hypothesize that a larger bid or ask quoted size, i.e., a higher depth at the best bid or ask, indicates a higher liquidity similar to a tighter bid or ask quote” (Grauer et al., 2022, p. 14)

“We coin this combination “depth rule + reverse LR”” (Grauer et al., 2022, p. 14)

“We show the overall success rates of the classification algorithms using our trade size rule and also calculate the change in the success rates compared to the same algorithms not using the trade size rule in parentheses. The results show that our new rule works best for small to medium-sized trades and even leads to a slight deterioration of the performance for the largest trade sizes.” (Grauer et al., 2022, p. 15)

“Namely, tick tests perform best when using most current price information across all exchanges and reverse tick tests based on subsequent prices dominate their counterparts based on preceding ones.” (Grauer et al., 2022, p. 16)

“For this reason, the LR algorithm outperforms the EMO rule as the former uses the tick tests to a smaller extent.” (Grauer et al., 2022, p. 16)

“The highest success rate of 72.40% is achieved by the trade size rule in combination with the quote rule first applied to CBOE quotes and then to the NBBO. In combination with the tick test or reverse tick test for the LR algorithm, we are able to correctly classify 72.12% and 72.39% of the option trades, respectively.” (Grauer et al., 2022, p. 18)

“Applying our depth rule after using the trade size rule and the quote rule and classifying the very small number of midspread trades that cannot be signed by our depth rule using the reverse tick test yields an additional improvement of 1.2% on average. It improves the success rate of the quote rule applied to CBOE quotes first and then to the NBBO to 73.37%. These results confirm that our depth rule outperforms the standard and reverse tick test in classifying midspread trades.” (Grauer et al., 2022, p. 18)

“Finally, to compare the performance of the algorithms over time, we look at the individual years of our sample period. To conserve space, we compute average success rates for the different specifications of the quote, tick, LR, reverse LR, EMO, and depth rules.” (Grauer et al., 2022, p. 19)

“Comparing the classification precision of options written on common stocks, index options, and options written on other underlyings (mainly ETFs), we find lower success rates for index options, which is consistent with Savickas and Wilson (2003). Interestingly, the improvements due to our trade size rule are particularly high for index options at the CBOE.” (Grauer et al., 2022, p. 20)

“Most importantly, in contrast to standard and reverse tick tests, our newly proposed depth rule leads to a significant improvement compared to using the quote rule alone, pointing to its superior performance to sign midspread trades that quote rules cannot classify.” (Grauer et al., 2022, p. 21)

“Using our new methodology allows to correctly classify between 73% and 75% of option trades in our sample, which is more than 10% higher compared to the rules that are currently used in the literature.” (Grauer et al., 2022, p. 22)
