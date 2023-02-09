As the chapter [[🔢Tick Test]] unveils, the tick rule yields significantly lower success rates than the quote rule. For midspread trades, that can otherwise not be classified by the advantageous [[🔢Quote Rule]], ([[@grauerOptionTradeClassification2022]] p.14) propose the *depth rule*.

The depth rule infers the trade initiator from the quoted size at the best bid and ask. Based on the observation that an exceeding bid or ask size relates to higher liquidity on one trade side, trades are classified as a buy for a larger ask size and sell for a higher bid size ([[@grauerOptionTradeClassification2022]] 14).

Let $\tilde{A}_{i,t}$ denote the quoted size of the ask and $\tilde{B}_{i,t}$ the size of the bid. The depth rule is formally given by: 

$$
\tag{5}
  
\begin{equation}
    \text{Trade}_{i,t}=
    \begin{cases}
      0, & \text{if}\ P_{i, t} = m_{i, t}\ \text{and}\ \tilde{A}_{i,t} < \tilde{B}_{i,t}\\
      1, & \text{if}\ P_{i, t} = m_{i, t}\ \text{and}\ \tilde{A}_{i,t} > \tilde{B}_{i,t}.  \\
    \end{cases}
  \end{equation}
$$
As shown in Equation $(5)$, the depth rule classifies midspread trades only, if the ask size is different from the bid size, as the ratio between the ask and bid size is the sole criterion for inferring the trade's aggressor. Due to these restrictive conditions, the depth rule can only sign a small fraction of all trades and is best stacked with others rules.

Like the [[🔢Quote Rule]], the depth rule has additional data requirements being dependent on quote data. Despite being applied to midspread trades only, ([[@grauerOptionTradeClassification2022]] p.4) report an improvement in the overall accuracy $1.2~\%$ for the CBOE data set and by $0.8~\%$ on the ISE sample merely through the depth rule. The rule has not yet been tested in other markets.

**Notes:**
[[🔢Depth rule notes]]