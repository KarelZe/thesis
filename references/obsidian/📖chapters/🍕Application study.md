## Setup
Albeit the classification accuracy is a reasonable measure for comparing classifiers, one cannot immediately infer how changes in accuracy e. g., an improvement by $1{}\%$, affect the application domains. In an attempt to make our results tangible, we apply all algorithms to estimate trading cost, a problem we previously identified to be reliant on correct trade classification (cp. [[👶Introduction]]) and a common testing ground for trade classification rules (cp. [[@ellisAccuracyTradeClassification2000]]541) and ([[@finucaneDirectTestMethods2000]]569)  and ([[@petersonEvaluationBiasesExecution2003]]271--278) and ([[@savickasInferringDirectionOption2003]]896--897).

One of the most widely adopted measures for trading costs is the effective spread ([[@Piwowar_2006]]112). It is defined as the difference between the trade price and the fundamental value of the asset ([[@bessembinderIssuesAssessingTrade2003]]238--239).  Following ([[@bessembinderIssuesAssessingTrade2003]]238--239), we define the *nominal, effective spread* as 
$$
S_{i,t} = 2 (P_{i,t} - V_{i,t}) D_{i,t}.
$$
Like before, $i$ indexes the security and $t$ denotes the trade. Here, $D_{i,t}$ is the trade direction, which is either $1$ for customer buy orders and $-1$ for customer sell orders. If the trade initiator is known, we set $D_{i,t} = y_{i,t}$ and $D_{i,t}=\hat{y}_{it}$, if inferred from a rule or classifier. As the fundamental value $V_{i,t}$ is unobserved at the time of the trade, we follow a common track in research and use the midpoint of the prevailing quotes as an observable proxy. footnote-(For an alternative treatment for options (cp.[[@muravyevOptionsTradingCosts2020]]4975--4976). Our focus is on the midspread, as it is the most common proxy for the value.) This is also a natural choice, assuming that, on average, the spread is symmetrical and centred around the true fundamental value ([[@leeMarketIntegrationPrice1993]]1018). ~~~([[@hagstromerBiasEffectiveBidask2021]]317) reasons that the appeal of using the midpoint lies in the high data availability, simplicity, and applicability in an online setting.~~ We multiply the so-obtained half-spread by $2$ to obtain the effective spread, which represents the cost for a round trip trade involving a buy and sell ex commissions.

Readily apparent from (cref-eq), poor estimates for the predicted trade direction, lead to an under or over-estimated effective spread, and hence to a skewed trade cost estimate. By comparing the true effective spread from the estimated, we can derive the economical significance. For convenience, we also calculate the *relative effective spread* as 
$$
{PS}_{i,t} = S_{i,t} / V_{i,t}.
$$
The subsequent section estimates both the nominal and relative effective spread for our test sets.

## Results
The actual and the estimated effective spreads, as well as the quoted spread, are shown in the (cref tab) aggregated by mean.  ([[@savickasInferringDirectionOption2003]] 896--897) previously estimated the effective spreads on a subset of rules for option trades at the gls-CBOE, which can be compared against.

A $t$-test is used to test if the estimated, effective spread is significantly different from the mean true effective spread / significantly greater than zero at $p=0.01$ (cp.[[@finucaneDirectTestMethods2000]]570). Similarily in [[@chakrabartyTradeClassificationAlgorithms2007]]. Alternatively compare correlations $\rho$ and medians using the Wilcoxon test with null hypothesis of the equal medians with $p=0.01$ (cp.[[@theissenTestAccuracyLee2000]]12).

(🔥What can we see? How do the results compare?)

**Notes:**
[[🍕Application study notes]]