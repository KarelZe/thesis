*title:* Liquidity Measurement Problems in Fast, Competitive Markets: Expensive and Cheap Solutions: Liquidity Measurement Problems in Fast, Competitive Markets
*authors:* Craig W. Holden, Stacey Jacobsen
*year:* 2014
*tags:* #quote-rule #trade-classification #timing-problems
*status:* #📦 
*related:*
- [[@jurkatisInferringTradeDirections2022]] (made me aware of this paper)
*code:*
*review:*

## Notes 📍

## Annotations 📖
Note: 

“In fast markets, the millisecond time stamp might well be important in matching trades and quotes and the official NBBO quotes may contain fewer errors than the raw quotes.” ([Holden and Jacobsen, 2014, p. 1748](zotero://select/library/items/Q8JL6HEW)) ([pdf](zotero://open-pdf/library/items/ZYFX3T4I?page=2&annotation=7QWXQI4H))

“There are three popular trade-typing conventions for determining whether a given trade is a liquidity-demander “buy” or liquidity-demander “sell,” which, in turn, determines whether Dk is +1or−1. Using the Lee and Ready (1991, LR) convention, a trade is a buy when Pk > Mk,asellwhenPk < Mk,andthe tick test is used when Pk = Mk. The tick test specifies that a trade is a buy (sell) if the most recent prior trade at a different price was at a lower (higher) price than Pk. Using the Ellis, Michaely, and O’Hara (2000, EMO) convention, atradeisabuywhenPk = Ak,asellwhenPk = Bk,andtheticktestisused otherwise. Using the Chakrabarty et al. (2006, CLNV) convention, a trade is abuywhenPk ∈ [0.3Bk + 0.7 Ak, Ak], a sell when Pk ∈ [Bk, 0.7Bk + 0.3 Ak], and the tick test is used otherwise.15 We consider three versions of dollar realized spread and three versions of dollar price impact based on these three tradetyping conventions.” ([Holden and Jacobsen, 2014, p. 1757](zotero://select/library/items/Q8JL6HEW)) ([pdf](zotero://open-pdf/library/items/ZYFX3T4I?page=11&annotation=KPJYYBPR))

“In particular, we consider three quote timing techniques: (1) Prior Second, (2) Same Second, and (3) Interpolated Time. Prior Second matches a trade in second s to the calculated NBBO quotes that are in-force in the prior second s−1. Same Second matches a trade in second s to the calculated NBBO quotes that are in-force during the same second s” ([Holden and Jacobsen, 2014, p. 1765](zotero://select/library/items/Q8JL6HEW)) ([pdf](zotero://open-pdf/library/items/ZYFX3T4I?page=19&annotation=84CVY8BM)) In the JF-JFE-RFS survey described earlier, we find significant variation in the quote timing techniques used. Seven articles use Prior Second, three articles use Same Second, five articles use the quote five seconds earlier, and the rest provide no information on quote timing.” ([Holden and Jacobsen, 2014, p. 1765](zotero://select/library/items/Q8JL6HEW)) ([pdf](zotero://open-pdf/library/items/ZYFX3T4I?page=19&annotation=FXDFXVMQ))

“We introduce a new, potentially more accurate, quote timing method that we call Interpolated Time. Suppose that the MTAQ data set lists I trades and J quotes as occurring in second s. We do not know in which millisecond those trades or quotes occurred, but we do know the order of the trades and the order of the quotes in MTAQ.” ([Holden and Jacobsen, 2014, p. 1766](zotero://select/library/items/Q8JL6HEW)) ([pdf](zotero://open-pdf/library/items/ZYFX3T4I?page=20&annotation=4G64GQHZ))