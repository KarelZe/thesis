*title:* An Intraday Analysis of the Probability of Trading on the Asx at the Asking Price
*authors:* Michael Aitken, Amaryllis Kua, Philip Brown, Terry Watter, H. Y. Izan
*year:* 1995
*tags:* #weekday-effects #logistic-regression #feature-enginering #trade-classification 
*status:* #📦 
*related:*
- [[@bessembinderIssuesAssessingTrade2003]] (found paper here)

## Notes 📍
- Paper doesn't cover trade classification directly. Authors use logistic regression to study how trading patterns affect trading at the asking price.
- They find that the probability of a trade at the asking price is related to the time of day, the dollar volume of the trade, the buy order imbalance, the bid-ask spread, the firm size and the average trading frequence, its price level and whether it is approved for short selling.
- They study the Australian stock market. Their database consists of SEATS transactions obtained from the ASX. The data includes the price, quantity and time signature information of all regular trades (some filtering done) between 4 September, 1990 to 3 September, 1993.
- They create some features like the number bid-ask spread (relative spread compared to mid), the order imbalance (ratio between ask and bid orders before to the trade). Similarily, the volume can be used. Also they use the trading volume, short selling indicator and firm size as a feature.
- Interestingly use $\chi^2$ test for categorical variables to test for the association whether a trade was at the asking price or not.

## Annotations 📖

“We explain the probability of a trade at the asking price across time.” ([Aitken et al., 1995, p. 115](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=1&annotation=ASL2HWPI))

“Our study is based heavily on intraday data.” ([Aitken et al., 1995, p. 117](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=3&annotation=DXXS6HKL))

“Systematic trading patterns contribute to returns anomalies. Portfolio returns reflect the intraspread movements if systematic trading patterns lead to more frequent trading at ask or bid prices. The wider the bid-ask spread, the larger the intraspread movement.” ([Aitken et al., 1995, p. 122](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=8&annotation=B6KUE2K2))

“But according to Schwartz (1988), the bid-ask spread has four components: activity, risk, information and competition.” ([Aitken et al., 1995, p. 124](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=10&annotation=YJBQUE2R))

“Previous empirical studies (Brock and Kleidon 1992; McInish and Wood 1992) have shown a direct relationship between the number of trades and the bid-ask spread.” ([Aitken et al., 1995, p. 125](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=11&annotation=AR7WH2A7))

“O ur database consists of the complete set of SEATS transactions obtained from the ASX. The data include price, quantity and time signature information for all live bids and asks, all trades, and all bids and asks entered or amended in any way.” ([Aitken et al., 1995, p. 126](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=12&annotation=JY5NG68C))

“The data used in this analysis are all regular trades on SEATS during the three year period from 4 September, 1990 to 3 September, 1993.” ([Aitken et al., 1995, p. 127](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=13&annotation=5B8ZTNVI))

“Bid-ask spread: a. Number of price steps between the best ask and best bid (SPRDSTEP).20 b. The relative spread, defined to be the ratio of the bid-ask spread to the simple average of the best bid and best ask (SPRDPERC).” ([Aitken et al., 1995, p. 129](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=15&annotation=3GYR6MLH))

“Order Imbalance between Bids and Asks: a. The ratio of the number of bid orders to the number of ask orders just before a trade occurs (IMBAL 1). b. The ratio of the volume of bid orders to the volume of ask orders just before a trade occurs (IMBAL 2).” ([Aitken et al., 1995, p. 129](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=15&annotation=356ZMLF7))

“We use univariate tests to examine the relationship between each independent variable and a trade at the asking price. One-tailed t-tests are employed to indicate if there is any significant directional difference between the means of independent continuous variables across trades at the asking price and trades at the bidding price” ([Aitken et al., 1995, p. 130](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=16&annotation=BWXEEXWN))

“The following logistic regression model is estimated in our multivariate analysis: Ωi = δ0 + t =1 2 Σ0 δt TIMEi,t + d =1 4 Σ δ20+d DAYi,d + δ25 PRICEi + δ26 DOLVOLi + δ27 IMBAL 1(IMBAL 2)i + δ28 SPRDPERC (SPRDSTEP)i + δ29 APPROSECi + δ30 FREQLOGi + δ31 SIZELOGi + ei” ([Aitken et al., 1995, p. 130](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=16&annotation=VJF4SF6Q))

“We investigate if systematic trading patterns affect the frequency with which a trade occurs at the asking price. For each minute, the proportion of trades at the asking price is computed” ([Aitken et al., 1995, p. 132](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=18&annotation=DB5MJKIS))

“A chi-square test was used for the categorical variables. Table 3 presents two-dimensional chi-square tests of association between whether a trade was or was not at the asking price and the time of day.24 The test in Table 3 is whether, within the various categories, the proportion of trades at the ask differ significantly from 0.5.” ([Aitken et al., 1995, p. 134](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=20&annotation=EMKUM4IL))

“We employ a dichotomous logistic regression to conduct a simultaneous test of our hypotheses. It is particularly suited to this study, as none of the continuous” ([Aitken et al., 1995, p. 136](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=22&annotation=LSPWCRWK))

“explanatory variables is Normally distributed (see Table 1).” ([Aitken et al., 1995, p. 138](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=24&annotation=RA2VQJVE))

“Our main aim was to explore timerelated differences in the probability of a trade at the asking price, for instance across time and days of the week, and their relationship to intraday average rates of return. In addition, other determinants of the probability are examined.” ([Aitken et al., 1995, p. 151](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=37&annotation=CTE5WZVW))

“The probability of a trade at the asking price was significantly related to seven factors other than the time of day: it was inversely related to dollar volume of the trade, buy order imbalance, bid-ask spread and firm size; and directly related to the average trading frequency in the stock, its price level, and whether the stock is approved for short selling.” ([Aitken et al., 1995, p. 151](zotero://select/library/items/PGAMDXH9)) ([pdf](zotero://open-pdf/library/items/RVL9H4BI?page=37&annotation=BZH6CXJB))