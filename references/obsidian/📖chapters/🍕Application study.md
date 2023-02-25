
-> Where are trade classificaton rules applied? What is the advantage of x% higher accuracy? Since these methods are most commonly applied in estimating effective bid-ask spreads and signed volume, the possibility that imperfectly inferring trade direction might bias empirical inferences is tested by comparing true effective spreads to effective spreads estimated using LR's method and the tick test, and by comparing true signed volume to signed volume estimated using the two methods of inferring trade direction.

- “We define the true transaction cost for a market order as the difference between the average transaction price and the true value of the asset.” (Goettler et al., 2005, p. 2172)
When trades are executed inside or outside the posted spread, the quoted spread does no longer represent the true spread for a trader. 

A common proxy for the transaction cost is the *effective spread*, which is the difference between trade price and fundamental value (citation). The *nominal, effective spread* is given by:
$$
S_{i,t} = 2 (P_{i,t} - V_{i,t}) D_{i,t}
$$
Here, $D_{i,t}$ is the trade direction, which is either $1$ for customer buy orders and $-1$ for customer sell orders. Unless the trade initiator is known, we set $D_{i,t} = y_{i,t}$, and $D_{i,t}=\hat{y}_{it}$, if inferred from rule or classifier. $V_{i,t}$ is an observable proxy of the $i$ -th security at $t$.  As the fundamental value is unobserved, it is must approximated. the bid-ask midpoint typically serves as a proxy (cp.[[@leeMarketIntegrationPrice1993]]1018). ([[@hagstromerBiasEffectiveBidask2021]]317) reasons that the appeal of using the midpoint lies it in it's high data availability, simplicity, and its applicability in an online setting.  We multiply by $2$ to convert the so-obtained half-spread into the effective spread.

As easily visible in (cref-eq) the , the correctness of the of the trade with a trade classification rule or machine learning-based classifier, as it leads to an under- or over-estimated effective spread, and ultimately or trade costs overall.

For convenience we also calculate the *relative effective spread* as:
$$
{PS}_{i,t} = S_{i,t} / V_{i,t}.
$$
The actual and the estimated effective spreads, as well as the quoted spread, are shown in the (cref tab). We aggregate results by (-> dimension). We are fortunate, that ([[@savickasInferringDirectionOption2003]] 896--897) estimated the effective spreads on a subset of rules for  option trades at the gls-CBOE, which can be compared.

A $t$-test is used to test if the estimated, effective spread is significantly different from the mean true effective spread / significantly greater than zero at $p=0.01$ . (inspired by [[@finucaneDirectTestMethods2000]]570) (Alternatively compare medians using Wilcoxon test with null hypothesis of equal median with p=0.01 + correlations $\rho$. Previously done in ([[@theissenTestAccuracyLee2000]] p. 12))

**Notes:**
[[🍕Application study notes]]