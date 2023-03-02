## Four sources
1. Livevol intraday data with prices and quotes w/o trader type
2. EOD trade volumes w trader type through 
	- ISE Open/Close Trade Profile 
	- CBOE Open-Close Volume Summary
3. Ivy DB OptionMetrics


## Open / Close 🏦

Specification: https://datashop.cboe.com/documents/OpenCloseSpecification.pdf
- contains daily trade volumes for the option series by trader type
1. Differentiated by whether position is opened `O` / closed `C`
2. Differentiated by buy  `B` and sell  `S` 
3. Differentiated by trader types: customer `C`, professional customer `U`, firm proprietary `F/L`, broker/dealer `B/J`. Market maker `M/N` (?)
- aggregate buy and sell volumes by account type:
	- $\{C\} \times \{B\} \times \{O,C\}$ (Customer buy (against market maker?))
	- $\{C\} \times \{S\} \times \{O,C\}$ (Customer sell (against market maker?))
	-  $\{U\} \times \{B\} \times \{O,C\}$ (professional customer buy)
	- $\{U\} \times \{S\} \times \{O,C\}$ (professional customer sell)
	- $\left\{BJ \right\} \times \{B\} \times \{O,C\}$ (dealer buy)
	- $\left\{BJ \right\} \times \{S\} \times \{O,C\}$ (dealer sell)
	- $\left\{FL \right\} \times \{B\} \times \{O,C\}$ (firm proprietary buy)
	- $\left\{FL \right\} \times \{S\} \times \{O,C\}$ (firm proprietary sell)
- Sum all 8 groups to obtain total trade volume at exchange
- Classify transactions in the live vol data set, if the daily volume in livevol matches with customer buys or customer sells. What about professional customer orders?
- Mach based on  unique key of trade date, expiration date, strike price, option type, root symbol of underlying. (root, trade date, expiration date, strike price, option tpye, day vol). If both customer buy and sell orders at a day no decision can be inferred.
- Classify based on customer buy / sell indicator from customer buy or sell volumes. Similar to [[@theissenTestAccuracyLee2000]] and [[@ellisAccuracyTradeClassification2000]]. They assume customer is the party with demand for options. 
- Merge with option metrics data based on unique key

## Option Quotes🪙
Specification: https://datashop.cboe.com/documents/Option_Quotes_Layout.pdf




- We construct datasets that suffice (...) and serve as an input to our machine learning models. 
- Refer to chapters (...) for which algorithm requires quote data (e. g., quote rule) and which requires price data (e. g., tick test).
- the ISE data set is the primary target for the study. Use CBOE data set as backup, if needed.
- data comes at a intra-day frequency (which resolution?)
- Data spans from May 2, 2005 to May 31, 2017 + (new samples until 2020). The dates are chosen non-arbitrarily: May 2, 2005 is first day of ISE Open/ Close. May, 31 2017 is last day of availability. We adhere to the data ranges to maintain consistency with [[@grauerOptionTradeClassification2022]]
- **Data sources from [[@grauerOptionTradeClassification2022]]:**
	- intraday option price data from `LiveVol` at a transaction level resolution
	- intraday option quote data from `LiveVol`
	- end-of-day buy and sell trading volumes from `ISE Open/Close Trade Profile` (daily resolution)
	- option and underlying characteristics from `Ivy DB OptionMetrics`
- `LiveVol` dataset contains Specifially, the trade price and trade size / volume, nbbo quotes, quotes and quote sizes of the exchanges where the option is quoted and information on the exchange where the trade is executed (see [[@grauerOptionTradeClassification2022]])
- Provide summary statistics, but wait until pre-processing and generation of true labels are introduced. May add pre-processing and generation of true labels in this chapter.

TODO: Is the delay between trades and quotes relevant here? Probably not, due to how matching is performed in [[@grauerOptionTradeClassification2022]] (See discussion in [[@rosenthalModelingTradeDirection2012]]) 


![[summary-statistic.png]]

(from [[@muravyevOptionsTradingCosts2020]])