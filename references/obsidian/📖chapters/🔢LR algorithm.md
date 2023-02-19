
The LR algorithm ([[@leeInferringTradeDirection1991]]745) combines the (reverse) tick test and quote rule into a single rule, which is derived from two observations. First, ([[@leeInferringTradeDirection1991]] 735--743) observe a higher precision of the quote-rule over the tick rule, which makes it their preferred choice. Second, by the means of a simple model, the authors demonstrate that the tick test can correctly classify at least $85~\%$ of all midspread trades, if the model's assumptions of constant quotes between trades and the arrival of market and standing orders following a Poisson process are met. 

In combination, the algorithm primarily signs trades according to the quote rule. Trades at the midpoint of the spread, unclassifiable by the quote rule, are classified by the tick rule. Overall:
$$
\begin{equation}
  \operatorname{lr} \colon \mathbb{N}^2 \to \left\{0,1\right\}\quad\operatorname{lr}(i,t)=
  \begin{cases}
    1,                                       & \text{if}\ p_{i, t} > m_{i, t} \\
    0,                                       & \text{if}\ p_{i, t} < m_{i, t} \\
    \operatorname{tick}(i, t), & \text{else}.
  \end{cases}
\end{equation}
$$
As the algorithm requires both trade and quote data, it is less data-efficient than its parts. Even if data is readily available, in past option studies the algorithm does not significantly outperform the quote rule and outside the model's tight assumptions the expected accuracy of the tick test is unmet ([[@grauerOptionTradeClassification2022]]30--32) and ([[@savickasInferringDirectionOption2003]]886).  Nevertheless, the algorithm is a common choice in option research (cp. [[@easleyOptionVolumeStock1998]]453). 
The LR algorithm is the basis for more advanced algorithms, such as the [[🔢EMO Rule]] rule, which we cover next. 

**Notes:**
[[🔢LR algorithm notes]]
