
[[@grauerOptionTradeClassification2022]] promote using trade size information to improve the classification performance of midspread trades. In their >>depth rule<<, they infer the trade initiator from the bid's depth and ask for quotes. Based on the observation that an exceeding bid or ask quote relates to higher liquidity on one side, trades are classified as buyer-iniated for a larger ask size and seller-iniated for a higher bid size.

As shown in Algorithm 2, the depth rule classifies midspread trades only, if the ask size is different from the bid size, as the ratio between the ask and bid size is the sole criterion for assigning the initiator. To sign the remaining trades, other rules must follow thereafter.

![[depth-rule-algo.png]]

In a similar vain the subsequent *trade size rule* utilizes the ask and bid quote size to improve classification performance.

- classify midspread trades as buyer-initiated, if the ask size exceeds the bid size, and as seller-initiated, if the bid size is higher than the ask size (see [[@grauerOptionTradeClassification2022]])
- **Intuition:** trade size matches exactly either the bid or ask quote size, it is likely that the quote came from a customer, the market maker found it attractive and, therefore, decided to fill it completely. (see [[@grauerOptionTradeClassification2022]])
- Alternative to handle midspread trades, that can not be classified using the quote rule.
- Improves LR algorithm by 0.8 %. Overall accuracy 75 %.
- Performance exceeds that of the LR algorithm, thus the authors assume that the depth rule outperforms the tick test and the reverse tick test, that are used in the LR algorithm for for classifying midspread trades.
