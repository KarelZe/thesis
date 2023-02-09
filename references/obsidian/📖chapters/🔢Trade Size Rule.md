
As the chapter [[🔢Quote Rule]] derives, quote-based approaches are generally preferred due to the improved performance relative to the [[🔢Tick Test]]. (More general background on the problem?) ([[@grauerOptionTradeClassification2022]] 13) stress that the quote rule, however, systematically misclassifies limit orders and propose an alternative procedure to identify and override predictions for them.

$$
\tag{10}
  \begin{equation}
    \text{Trade}_{i,t}=
    \begin{cases}
      1, & \text{if}\ \tilde{P}_{i, t} = \tilde{B}_{i, t}\ \text{and}\ \tilde{P}_{i, t} \neq \tilde{A}_{i, t} \\
      0, & \text{if}\ \tilde{P}_{i, t} = \tilde{A}_{i, t}\ \text{and}\ \tilde{P}_{i, t} \neq \tilde{B}_{i, t}  \\
    \end{cases}
  \end{equation}
$$
The *trade size rule* in Equation $(10)$ classifies based on a match between the size of the trade $\tilde{P}_{i, t}$ and the quoted bid and ask sizes. The rationale is, that the market maker tries to fill the limit order of a customer, which results in the trade being executed at the prevailing bid or ask, with a trade size equalling the quoted size ([[@grauerOptionTradeClassification2022]] 13). By definition, when both the size of the ask and bid correspond with the trade size, no class can be assigned. 

([[@grauerOptionTradeClassification2022]] 13) report an accuracy of $79.92~\%$ for the subset of option trades at the ISE ($22.3~\%$ of all trades) that can be signed using the methodology, which contributes to a performance improvement of $11~\%$ for the entire sample. Expectedly, the improvement is highest for trades at the quotes and reverses for trades outside the quote ([[@grauerOptionTradeClassification2022]] 15). Based on these results, the trade size rule may only be applied selectively to trades inside or at the quote. Since only a fraction of all trades can be classified with the trade size rule, the rule must be combined with other basic or hybrid rules for complete coverage. The subsequent section introduces three such hybrid algorithms, that combine the tick and quote rule into more sophisticated algorithms to address the limitations of the former.

**Notes:**
[[🔢Trade Size Rule Notes]]