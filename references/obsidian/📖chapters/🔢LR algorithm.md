
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

The algorithm is derived from an analysis of stock trades inside the quotes ([[@leeInferringTradeDirection1991]] 742). 

The strength of the algorithm lies in combining the strong classification accuracies of the quote rule with the universal applicability of the tick test.
As it requires both trade and quote data, it is less data-efficient than its parts. Even if data is readily available, in past studies the algorithm does not significantly outperform the quote rule ([[@grauerOptionTradeClassification2022]]30--32) and ([[@savickasInferringDirectionOption2003]]886).  Nevertheless, the algorithm is a common choice in option research (cp. [[@easleyOptionVolumeStock1998]]453). 

The LR algorithm is the basis for more advanced algorithms, such as the [[🔢EMO Rule]] rule, which we cover next. 

**Notes:**
[[🔢LR algorithm notes]]
