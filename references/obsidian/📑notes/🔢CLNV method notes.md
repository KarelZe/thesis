Tags: #trade-classification #CLNV 

Long form:
$$
  \begin{equation}

    \text{Trade}_{i,t}=
    \begin{cases}
      \operatorname{tick}(), & \text{if}\ p_{i, t} \in \left(a_{i, t}, \infty\right) \\
      1, & \text{if}\ p_{i, t} \in \left[\frac{3}{10} b_{i,t} + \frac{7}{10} a_{i,t}, a_{i, t}\right] \\
      \operatorname{tick}(), & \text{if}\ p_{i, t} \in \left(\frac{7}{10} b_{i,t} + \frac{3}{10} a_{i,t}, \frac{3}{10} b_{i,t} + \frac{7}{10} a_{i,t}\right) \\
      0, & \text{if} p_{i, t} \in \left[ b_{i,t}, \frac{7}{10} b_{i,t} + \frac{3}{10} a_{i,t}\right] \\
	  \operatorname{tick}(), & \text{if} \ p_{i, t} \in \left(-\infty, b_{i, t}\right) \\
    \end{cases}
  \end{equation}
$$


Long form:
$$
  \begin{equation}

    \text{Trade}_{i,t}=
    \begin{cases}
      \operatorname{tick}(), & \text{if}\ P_{i, t} \in \left(A_{i, t}, \infty\right) \\
      1, & \text{if}\ P_{i, t} \in \left[\frac{3}{10} B_{i,t} + \frac{7}{10} A_{i,t}, A_{i, t}\right] \\
      \operatorname{tick}(), & \text{if}\ P_{i, t} \in \left(\frac{7}{10} B_{i,t} + \frac{3}{10} A_{i,t}, \frac{3}{10} B_{i,t} + \frac{7}{10} A_{i,t}\right) \\
      0, & \text{if} P_{i, t} \in \left[ B_{i,t}, \frac{7}{10} B_{i,t} + \frac{3}{10} A_{i,t}\right] \\
	  \operatorname{tick}(), & \text{if} \ P_{i, t} \in \left(-\infty, B_{i, t}\right) \\
    \end{cases}
  \end{equation}
$$


- “We compare the accuracy rates of various algorithms in classifying ECN trades. We divide trades according to their price distribution relative to quotes. We expect the classification rules to perform better when trades occur at the ask or bid. When trades receive price improvement, buys and sells will execute at prices inside the quotes. In these instances, not only does the quote rule meet with some challenge, the tick rule will also be more difficult as buys (sells) will more likely occur on downticks (upticks).” (Chakrabarty et al., 2007, p. 3811)

- “We find that ECN trades are difficult to classify as the overall success rates of the LR, EMO, and tick rules are 74.42%, 75.80%, and 75.40%, respectively. Our algorithm modestly outperforms these three, with an overall success rate of 76.52%, but we show that our algorithm substantially outperforms the others for trades inside the quotes. For trades inside 1 Our primary sample is from the INET ECN. We use ArcaEx data to confirm the robustness of our results across another ECN. Table 3 reports results using the ArcaEx data. All other tables use the INET sample. B. Chakrabarty et al. / Journal of Banking & Finance 31 (2007) 3806–3821 380” (Chakrabarty et al., 2007, p. 3807) “the quotes our success rate is 76.32% compared to 71.85% for the LR rule and 71.35% for both the tick and EMO rules. We also find that a significant proportion of trades execute outside the quotes. This phenomenon, called a ‘‘trade-through’’, is prevalent for NASDAQ markets due to the absence of the trade-through rule in our study period.2” (Chakrabarty et al., 2007, p. 3808)

- started by [[@leeInferringTradeDirection1991]] (733-746),
- CLNV-Method is a hybrid of tick and quote rules when transactions prices are closer to the ask and bid, and the the tick rule when transaction prices are closer to the midpoint [[@chakrabartyTradeClassificationAlgorithms2007]]. Authors continue the trend to deeper / more sophisticated rules.
- extension to the [[@ellisAccuracyTradeClassification2000]] algorithm. Algorithm was invented after the EMO rule. Thus the improvement, comes from a higher segmented decision surface. (also see graphics [[visualisation-of-quote-and-tick.png]])
- Sometimes referred to as the MEMO (modified EMO) algorithm. (See e. g., [[@frommelAccuracyTradeClassification2021]]) 
- authors use a true out-of-sample test set to test their hypothesis. April for training and May to June for testing. Also, they test their hypothesis on a second data set. [[@chakrabartyTradeClassificationAlgorithms2007]] (p. 3809)
- “Another variation divides the spread into three parts, and uses the quote rule for trades close to the bid and ask, while applying the tick rule to the 20% band around the midpoint, and for trades outside the spread (Chakrabarty et al., 2007, “CLNV”). We use these three algorithms, in addition to the tick rule, to compare with BVC.” (Pöppe et al., 2016, p. 167)
- **idea:** We divide trades according to their price distribution relative to quotes. We expect the classification rules to perform better when trades occur at the ask or bid. When trades receive price improvement, buys and sells will execute at prices inside the quotes. In these instances, not only does the quote rule meet with some challenge, the tick rule will also be more difficult as buys (sells) will more likely occur on downticks (upticks).” ([Chakrabarty et al., 2007, p. 3811](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=6&annotation=6NIJNJ58)) 
- “When comparing the performance of the tick and quote rules for each decile, a clear pattern arises showing which rule performs better in each decile. Uniformly, the quote rule is superior to the tick rule when transaction prices are closer to the quotes (in deciles A3–A5 and B3–B5). However, when transaction prices are closer to the midpoints (in deciles A1, A2, B1, and B2), the performance of the tick rule is better than that of the quote rule.” ([Chakrabarty et al., 2007, p. 3811](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=6&annotation=NEYHHSVW)) “When trades receive price improvement, classifications are problematic as buys (sells) execute at prices away from the ask (bid). Dividing inside trades into deciles shows that the quote rule is better for trades closer to the ask and the bid and the tick rule does better when transaction prices are closer to the midpoint.” ([Chakrabarty et al., 2007, p. 3812](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=7&annotation=ASB83EBG))
- **algorithm:** “Our algorithm is a hybrid of the tick and quote rules; it uses the quote rule when transaction prices are closer to the ask and bid and uses the tick rule when transaction prices are closer to the midpoint. Specifically, we divide the spread into deciles (10% increments). We use the quote rule if transaction prices are in the top (A5, A4 and A3) and bottom (B5, B4 and B3) three deciles. If transaction prices are in the two deciles above the midpoint or two deciles below the midpoint, we use the tick rule. For trades at the quotes, we use the quote rule, since the results from Table 1 Panel A show that the quote rule is better at these points. Fig. 1 illustrates this alternative algorithm.” ([Chakrabarty et al., 2007, p. 3812](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=7&annotation=4QD7Q4NX))
- Use clear formula 🟰
![[clnv-method-visualisation.png]]
(image copied from [[@chakrabartyTradeClassificationAlgorithms2007]])

From the algorithm it remains unclear what is the upper boundary of the third quartile. I decided to to classify it with the tick rule similar to algorithm in [[@jurkatisInferringTradeDirections2022]] and the assumption that intverals go like $[0, \frac{1}{10}), [\frac{1}{10}, \frac{2}{10})...$ 

![[pseudocode-of-algorithms.png]]
(found in [[@jurkatisInferringTradeDirections2022]]). Overly complex description but helpful for implementation?


- **word explanation:** “The CLNV algorithm assigns the trade initiator to the buying (selling) side if the transaction price is equal to the ask (bid) or up to 30% of the spread below (above) the ask (bid). For all trades above (below) the ask (bid) or within a 40% range of the spread around the mid-point, the tick-test is used. The classification algorithms in pseudo code are summarised in Table A2 in the Appendi” ([[@jurkatisInferringTradeDirections2022]], 2022, p. 7)
- **dataset:** “The evidence for the success of our algorithm is based on INET data. While INET handles the largest share of NASDAQ stocks, we also verify the success rate of our algorithm using data from the ArcaEx, the second largest venue for NASDAQ stocks. We obtain one week’s complete order book and trade data (September 12–16, 2005) for a sample of the top 100 (by trading volume) NASDAQ stocks traded on the ArcaEx, and compare the accuracy rates of the various trade classification algorithms with the actual numbers of buy/sell-initiated trades.” ([Chakrabarty et al., 2007, p. 3815](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=10&annotation=XIFPZQET))
- results: “The overall success rates of the LR, EMO, and tick rules are 74.42%, 75.80%, and 75.40%, respectively.  ([Chakrabarty et al., 2007, p. 3821](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=16&annotation=I4A9CCUN))
- “The EMO and MEMO rule were specifically created to cope with this problem and their superior performance should thus imply that this bias is also present in this study. Odders-White (2000) and Theissen (2000) also reported worse performances for trades occurring on the midpoint of the b/a spread, which is a specific case of trades occurring inside the quotes. We do, however, not look at trades at the midpoint, because there are too few of them in our sample.” ([[@frommelAccuracyTradeClassification2021]], p. 7)
- “Furthermore, the most important biases encountered in the literature have been confirmed in this study: Seller-initiated trades perform remarkably better than buyer-initiated trades. The EMO rule, and especially the MEMO rule, offer substantial improvements over LR as they have far more power for classifying trades that occurred inside the quotes. The biggest disadvantage of the TR is its poor performance for zero ticks.” (Frömmel et al., 2021, p. 9) -> How are things in [[@savickasInferringDirectionOption2003]] and [[@grauerOptionTradeClassification2022]]




