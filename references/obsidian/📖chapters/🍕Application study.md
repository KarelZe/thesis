Albeit the classification accuracy is a reasonable measure for comparing classifiers, one cannot immediately infer how changes in accuracy e. g., an improvement by 1%, affect the application domains. In attempt to make our results tangible, we apply all algorithms to estimate trading cost, a problem we previously identified identified to be heavily reliant on correct trade classification (see [[👶Introduction]]) and a common testing ground for new trade classification rules (cp. ...).

A common for trading cost is the *effective spread*, which is the difference between trade price and fundamental value (citation). The *nominal, effective spread* is given by:
$$
S_{i,t} = 2 (P_{i,t} - V_{i,t}) D_{i,t}
$$
Like before, $i$ indexes the security and $t$ denotes the time.Here, $D_{i,t}$ is the trade direction, which is either $1$ for customer buy orders and $-1$ for customer sell orders. Unless the trade initiator is known, we set $D_{i,t}=\hat{y}_{it}$, and $D_{i,t} = y_{i,t}$ if inferred from rule or classifier. As the fundamental value, is typically unobserved, it must be approximated by $V_{i,t}$ . Typically, the midpoint of the quoted spread serves as a proxy (cp.[[@leeMarketIntegrationPrice1993]]1018). ([[@hagstromerBiasEffectiveBidask2021]]317) reasons that the appeal of using the midpoint lies it in it's high data availability, simplicity, and its applicability in an online setting. We multiply by $2$ to convert the so-obtained half-spread into the effective spread. <mark style="background: #ABF7F7A6;">(Lee Why does the midpoint make sense from a theoretical perspective?)</mark>

<mark style="background: #FF5582A6;">Readily apparent from (cref-eq), the correctness of the with a trade classification rule or machine learning-based classifier, as it leads to an under- or over-estimated effective spread, and ultimately or trade costs overall.</mark>

For convenience we also calculate the *relative effective spread* as:
$$
{PS}_{i,t} = S_{i,t} / V_{i,t}.
$$
The actual and the estimated effective spreads, as well as the quoted spread, are shown in the (cref tab) aggegated by. We are fortunate, that ([[@savickasInferringDirectionOption2003]] 896--897) estimated the effective spreads on a subset of rules for  option trades at the gls-CBOE, which can be compared.

A $t$-test is used to test if the estimated, effective spread is significantly different from the mean true effective spread / significantly greater than zero at $p=0.01$ . (inspired by [[@finucaneDirectTestMethods2000]]570) (Alternatively compare medians using Wilcoxon test with null hypothesis of equal median with p=0.01 + correlations $\rho$. Previously done in ([[@theissenTestAccuracyLee2000]] p. 12))

(🔥What can we see? How do the results compare?)

**Notes:**
[[🍕Application study notes]]