
The LR algorithm ([[@leeInferringTradeDirection1991]]745) combines the (reverse) tick test and quote rule into a single algorithm. The algorithm signs the trades according to the quote rule. Trades at the midpoint of the spread, unclassifiable by the quote rule, are classified by the tick rule. The LR algorithm, $\operatorname{lr} \colon \mathbb{R}^3 \to \left\{0,1\right\}$, are defined as:
$$
\begin{equation}
  \operatorname{lr}(p_{i,t-1}, p_{i,t}, m_{i,t})=
  \begin{cases}
    1,                                       & \text{if}\ p_{i, t} > m_{i, t} \\
    0,                                       & \text{if}\ p_{i, t} < m_{i, t} \\
    \operatorname{tick}(p_{i,t-1}, p_{i,t}), & \text{else}.
  \end{cases}
\end{equation}
$$

The algorithm is derived from an analysis of stock trades inside the quotes ([[@leeInferringTradeDirection1991]] 742). <mark style="background: #FFF3A3A6;">-> Better to grab the two points from the paper.</mark>

<mark style="background: #D2B3FFA6;">quoted spreads are available  “Studies examining more recent data have found more mixed results, with LR accuracy lower than TR rates in some cases, and ranging from 72.1%-93.57%. Ellis et al. (2000) attribute lower LR rates in recent data to high order submission and cancellation rates along with significant market fragmentation which render quote/trade matching less precise.” (Ronen et al., 2022, p. 7)
</mark>

<mark style="background: #FFF3A3A6;">Based on their observations, the authors recommended using the quote-based approach over the tick test due to its greater precision. They also demonstrated that the tick test correctly classified approximately 85% of trades at the spread midpoint in a simple model. This suggests that combining the two methods would be optimal, given the high predicted accuracy of the tick method for midpoint trades and the likely superiority of the quote method.</mark>

<mark style="background: #ABF7F7A6;">“First, they noted that &the primary limitation of the tick test is its relative imprecision when compared to a quotebased approach'. This implies that the quote method should be employed whenever possible. Furthermore, in the context of a simple model, they demonstrated that the tick test correctly classiffied roughly 85% of trades occurring at the spread midpoint. The high predicted rate of accuracy of the tick method for midpoint trades, along with the likely superiority of the quote method, suggested that the proposed combination of the two was optimal.” (Odders-White, 2000, p. 264)</mark>

<mark style="background: #D2B3FFA6;">“Although LR's method is perhaps the most commonly used method for classifying trades, its general acceptance has been based in large part upon a simple analytical model that suggests a high degree of accuracy. LR predict that the tick test can be expected to correctly classify at least 85% of the trades that occur at the midpoint of the spread and at least 90% of bid or ask trades, but their predictions are based in part on a model that assumes constant quoted prices and independent Poisson processes for the arrival of market buy and sell orders and buy and sell orders being worked by floor brokers. To the extent that their assumptions are not met, LR's predictions may not accurately reflect the precision of the tick test when it is applied to actual financial markets.” (Finucane, 2000, p. 557)
</mark>

The strength of the algorithm lies in combining the strong classification accuracies of the quote rule with the universal applicability of the tick test.
As it requires both trade and quote data, it is less data-efficient than its parts. Even if data is readily available, in past studies the algorithm does not significantly outperform the quote rule ([[@grauerOptionTradeClassification2022]]30--32) and ([[@savickasInferringDirectionOption2003]]886).  Nevertheless, the algorithm is a common choice in option research (cp. [[@easleyOptionVolumeStock1998]]453). 

The LR algorithm is the basis for more advanced algorithms, such as the [[🔢EMO Rule]] rule, which we cover next. 

**Notes:**
[[🔢LR algorithm notes]]
