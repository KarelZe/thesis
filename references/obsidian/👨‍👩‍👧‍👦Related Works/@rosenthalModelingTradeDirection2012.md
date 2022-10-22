
title: Modeling Trade Direction
authors: D. W. R. Rosenthal
year: 2012
tags : #rosenthal #trade-classification #lr #tick-rule #quote-rule
status : #📥

# Notes 

# Annotations  
(08/10/2022, 13:07:12)

“Trade classification is used in many different areas and is thus important to many researchers. For example, some event studies seek to determine the balance of buying and selling in response to a release of information or a governmental action.” (Rosenthal, 2012, p. 1)

“A 1%–2% accuracy improvement would increase estimates of price impact by 2%–4% and result in more careful and inexpensive trading of illiquid orders. Given that a large investment bank may trade $2 billion per day, a cost savings of 0.01% would be worth $100 million annually to one such bank.” (Rosenthal, 2012, p. 2)

“A few methods exist for classifying trades, most comparing trade prices to prevailing price quotes.” (Rosenthal, 2012, p. 2)

“A further complication is the delays between publishing times (i.e. timestamps) of trades and quotes; however, classification methods without rigorous delay assumptions are often compared.” (Rosenthal, 2012, p. 2)

“To improve classification accuracy, I incorporate different methods into a model for the likelihood a trade was buyer-initiated. I also allow for joint estimation of a (latent) delay model. Modeling trade classifications is a new approach and one of the unique contributions of this work.” (Rosenthal, 2012, p. 2)

“Trade classification infers which trade participant initiated a trade by being the aggressor, consistent with Odders-White (2000) defining the laterarriving order as the trade initiator. Three approaches to trade classification dominate the literature: tick tests, midpoint tests, and bid/ask tests. Finucane (2000) recommends a tick test: comparing a trade price to the previous (differing) trade price for that stock. A lower previous trade price is taken as evidence the current trade was buy-initiated. Lee and Ready (1991) suggested a midpoint test: comparing a trade price to the lagged midpoint (i.e. average of best bid and ask quote). Trades at prices above (below) the midpoint are classified as buy- (sell-) initiated. Trades at the midpoint are resolved with a tick test. Ellis et al. (2000) suggested a bid/ask test for Nasdaq stocks; Peterson and Sirri (2003) then suggested it for NYSE stocks. Trades at the lagged ask (bid) are classified as buy- (sell-) initiated; other trades are resolved with a tick test. Trades are published with delay relative to quotes. Therefore, midpoint and bid/ask methods require delay assumptions. For midpoint methods, Lee and Ready (1991) use a delay of five seconds; Vergote (2005) suggests” (Rosenthal, 2012, p. 3)

“two seconds; and, Henker and Wang (2006) suggest a one-second delay. For bid/ask methods, all previous analyses used unlagged quotes.” (Rosenthal, 2012, p. 4)

“Many studies note that trades are published with non-ignorable delays. Lee and Ready (1991) first suggested a five-second delay (now commonly used) for 1988 data, two seconds for 1987 data, and “a different delay . . . for other time periods”. Ellis et al. (2000) note (Section IV.C) that quotes are updated almost immediately while trades are published with delay2. Therefore, determining the quote prevailing at trade time requires finding quotes preceding the trade by some (unknown) delay” (Rosenthal, 2012, p. 4)

## Partial Likelihood
The classification model is formally valid when formulated as a partial likelihood as in Cox (1975) and Wong (1986). Since we are classifying a sequence of trades and conditioning on $\mathcal{F}_t, t$ is not random. The randomness in the (conditional) classification model is due only to (i) the unknown amount of time to look backwards for a quote; and, (ii) the unknown trade classification. Were this not so, we would need to condition on the likelihood of each trade happening at its observed time.
If $t_i$ is the $i$-th trade time and $\mathcal{G}_{i-1}$ is a sigma-field encapsulating trade classifications $1, \ldots, i-1$, the full likelihood ratio can be decomposed as:
(5) $\mathcal{L}($ all data $)=\prod_{i=1}^n \mathcal{L}\left(B_{t_i} \mid \mathcal{F}_{t_i}, \mathcal{G}_{i-1}\right) \times \prod_{i=1}^n \mathcal{L}\left(\mathcal{F}_{t_i}, \mathcal{G}_{i-1} \mid \mathcal{F}_{t_{i-1}}, \mathcal{G}_{i-1}\right)$
For inference we only use the first factor, making this a partial likelihood. We assume $B_{t_i}$ is conditionally independent of $\mathcal{G}_{i-1}$ given $\mathcal{F}_{t_i}$, yielding $\mathcal{L}\left(B_{t_i} \mid \mathcal{F}_{t_i}, \mathcal{G}_{i-1}\right)=\mathcal{L}\left(B_{t_i} \mid \mathcal{F}_{t_i}\right)$

## Model Statement. 
Thus we can now state the (conditional) model:
$P\left(B_{j t}=\operatorname{Buy} \mid \mathcal{F}_t, c_k, d_{k \ell} ; \theta_o, \kappa_o\right)=\pi_{j t} ;$
$\pi_{j t}=\operatorname{logit}\left(\eta_{j t}\right) ; \quad$ and ,
$\eta_{j t}=\underbrace{\beta_0}_{\begin{array}{c}\text { bias } \\ =0 ?\end{array}}+\underbrace{\beta_{o 1} g\left(p_{j t}, \hat{m}_{j t}\right)}_{\text {midpoint metric }}+\underbrace{\beta_{o 2} g\left(p_{j t}, p_{j t-}^{\prime}\right)}_{\text {tick metric }}+\underbrace{\beta_{o 3} J\left(p_{j t}, \hat{b}_{j t}, \hat{a}_{j t} ; \tau\right)}_{\text {bid/ask metric }}+$
$\underbrace{\beta_{o 4} g\left(p_{j t-}, \hat{m}_{j t-}\right)}_{\text {lag-1 midpoint metric }}+\underbrace{\beta_{o 5} g\left(p_{j t-}, p_{j t-}^{\prime}\right)}_{\text {lag-1 tick metric }}+\underbrace{\beta_{o 6} J\left(p_{j t-}, \hat{b}_{j t-}, \hat{a}_{j t-;} ; \tau\right)}_{\text {lag-1 bid/ask metric }}+$
$\underbrace{c_k}_{\begin{array}{c}\text { overall } \\ \text { effect }\end{array}}+\underbrace{d_{k \ell}}_{\begin{array}{c}\text { within- } \\ \text { sector } \\ \text { effect }\end{array}}$
where $j$ indexes stocks, $k$ indexes time bins; $\ell$ indexes sectors; and, o indexes primary exchanges (e.g. NYSE, Nasdaq). A stock $j$ thus implies a sector $\ell$ and primary exchange $o$. The parameters of $f_Y$ are estimated jointly with model coefficients. The random effects are assumed to be $c_k \stackrel{\text { iid }}{\sim} N\left(0, \sigma_c^2\right)$ for all bins $k$ and $d_{k \ell} \stackrel{\text { iid }}{\sim} N\left(0, \sigma_d^2\right)$ for all bins $k$ and sectors $\ell$. Further, $c_k$ and $d_{k \ell}$ are assumed to be independent of the sigma-field $\mathcal{F}_t$.