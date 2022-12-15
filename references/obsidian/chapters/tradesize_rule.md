Motivated by the deminishing performance of the classical algorithms (such as the previously introduced tick test and quote rule) for option trades, where the trade size matches the bid or ask size, [[@grauerOptionTradeClassification2022]] propose to 

![[tradesize-rule.png]]

Due to the restrictions on the trade size, this rule needs to be combined with other rules.

- classify trades for which the trade size is equal to the quoted bid size as customer buys and those with a trade size equal to the ask size as customer sells. (see [[@grauerOptionTradeClassification2022]])
- **Intuition:** trade size matches exactly either the bid or ask quote size, it is likely that the quote came from a customer, the market maker found it attractive and, therefore, decided to fill it completely. (see [[@grauerOptionTradeClassification2022]])  
- Accuracy of 79.92 % on the 22.3 % of the trades that could classified, not all!. (see [[@grauerOptionTradeClassification2022]])
- Couple with other algorithms if trade sizes and quote sizes do not match / or if the trade size matches both the bid and ask size. For other 
- Requires other rules, similar to the quote rule, as only a small proportion can be matched.
- tested on option data / similar data set