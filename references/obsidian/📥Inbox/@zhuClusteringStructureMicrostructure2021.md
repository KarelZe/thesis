*title:* Clustering Structure of Microstructure Measures
*authors:* Liao Zhu, Ningning Sun, Martin T. Wells
*year:* 2021
*tags:* #related-works
*status:* #📦 
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖
Contains different definitions of spreads:

For the liquidity, we include different measures of the spreads. For each measure, we consider both prevailing and time-weighted values. For each 10-second time interval, the measures for spreads are:
1. Dollar bid-ask (quoted) spread $=$ ask price $-$ bid price
2. Proportional bid-ask (quoted) spread $=\frac{\text { ask price-bid price }}{\text { mid quote }} \times 100 \%$
3. Dollar effective spread: $\left|\text{trade price} - \text{mid quote}\right|$.
4. Proportional effective spread: $2\left|\frac{\text { trade price-mid quote }}{\text { mid quote }}\right| \times 100 \%$.

The depths of the market are based on the quotes for each stock, as a measure of liquidity and an indicator of price movement direction. The measures related to the depth of the market can be:
1. The last prevailing ask / bid / ask - bid / |ask - bid $\mid$.
2. The time weighted values, specifically, $\int \operatorname{ask}_t d t, \int \operatorname{bid}_t d t, \int\left(\operatorname{ask}_t-\operatorname{bid}_t\right) d t, \int\left|\operatorname{ask}_t-\operatorname{bid}_t\right| d t$.
For each ask (or bid) in the equation above, we can use dollar volume / number of shares / number of shares normalised by Average Daily Trading Volume (ADTV) of the prevailing month. And we can consider the measure for quotes in each exchange or the best quote nation-wide.

We also include the imbalance of quotes using a similar expression with that of imbalance of trades. For each 10-second time interval, the equations are:
$$
\frac{\text { ask - bid }}{\text { ask }+\text { bid }} \quad \text { (directional) } \quad \text { or } \quad\left|\frac{\text { ask - bid }}{\text { ask }+\text { bid }}\right| \quad \text { (nondirectional) }
$$