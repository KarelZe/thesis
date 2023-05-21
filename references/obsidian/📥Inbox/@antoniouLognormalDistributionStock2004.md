*title:* On the log-normal distribution of stock market data
*authors:* I Antoniou, Vi.V Ivanov, Va.V Ivanov, P.V Zrelov
*year:* 2003
*tags:* #stocks #log-transform #log-normal #feature-engineering
*status:* #📦 
*related:*
*code:* None
*review:* None

## Notes 📍
- They propose a new indicator or stachstic variable $\Xi$, which is defined as the closing price normalised by the volume. The variable $\Theta$ has a stable mean and dispersion, compared to its nominator and denominator.
- The resulting distribution roughly matches a lognormal distribution. 
- The goodness of fit can be measured with the $\chi^2$ test with certain degrees of freedom.
- They approximate the lognormal distribution with:
$$
f(x)=\frac{A}{\sqrt{2 \pi} \sigma x} \exp ^{-\left(1 / 2 \sigma^2\right)(\ln x-\mu)^2}.
$$
![[lognormal-distribution-logs.png]]

## Annotations 📖
“In Section 3, we show that for some stock market data the statistical distributionof the closing prices normalised by corresponding traded volumes (“price/volume” ratio) 1ts well the log-normal law.” ([Antoniou et al., 2004, p. 618](zotero://select/library/items/X5DBX927)) ([pdf](zotero://open-pdf/library/items/2UH6YXLZ?page=3&annotation=V34PBRUM))

“Aiming to minimise the in=uence of the trend, we introduce a new stochastic variable , which is the closing price normalised by the corresponding traded volume (price/volume). The variable (see left bottom plots inFigs. 3–9) has a relatively stable mean value and dispersion, compared to the original time series of closing prices and traded volumes.” ([Antoniou et al., 2004, p. 620](zotero://select/library/items/X5DBX927)) ([pdf](zotero://open-pdf/library/items/2UH6YXLZ?page=5&annotation=CKPYGLQ4))

“In order to test the hypothesis on the correspondence of these statistical distributions to the log-normal law, they are approximated by (see, for example, Refs. [3,10]) f(x)= A √2x exp−(1=22)(ln x− )2 ;” ([Antoniou et al., 2004, p. 621](zotero://select/library/items/X5DBX927)) ([pdf](zotero://open-pdf/library/items/2UH6YXLZ?page=6&annotation=36XKPPGV))

“The quality of approximationis represented by the value  
2=ndf , where “ndf” is the number of degrees of freedom.” ([Antoniou et al., 2004, p. 622](zotero://select/library/items/X5DBX927)) ([pdf](zotero://open-pdf/library/items/2UH6YXLZ?page=7&annotation=ILEV494H))
