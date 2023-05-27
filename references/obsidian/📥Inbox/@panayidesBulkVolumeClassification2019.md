*title:* Bulk volume classification and information detection
*authors:* Marios A. Panayides, Thomas D. Shohfi, Jared D. Smith
*year:* 2019
*tags:*  #🇪🇺 #stocks #trade-classification #bvc #bulk-classification #bias #lr #tick-rule #quote-rule 
*status:* #📥
*related:*
- [[@leeInferringTradeDirection1991]]
- [[@easleyDiscerningInformationTrade2016]] (bulked versions)
- [[@holthausenEffectLargeBlock1987]]
- [[@easleyDiscerningInformationTrade2016]] (bulked versions)

## Notes 📍
- paper might be interesting for my papers due to the markets studied and as it also considers bulked versions of LR and tick rule.
- Authors compare the BVC algorithm of [[@easleyDiscerningInformationTrade2016]] on two samples of European stocks. 
- Their data set consists of subsamples from the NYSE Euronext and the London Stock exchange with low, medium and high volatility. Thus the samples are not consecutive.
- The authors praise the BVC algorithm for being data effficient. The accuracy however, lacks behind the bulked version of the tick rule, lr algorithm etc. in the Euronext sample. For the Euronext sample in 2017 BVC is however most accurate.
- The BVC algorithm doesn't classify trades on a trade-to-trade basis, but rather over blocks of trades. BVC uses the total volume and price changes within a block to classify the order flow into buyer or seller-iniated.
- BVC works by putting trades into blocks or bars by time or volume. Based on the movement of the prices around the bars, a degree of trades within the block is classified as buys and all other trades as sells.
- Authors also study trading intensions in order flow with the BVC algorithm. 
- They also write about *systematic bias* and investigate bias. (similar to [[@odders-whiteOccurrenceConsequencesInaccurate2000]])

## Annotations 📖


“uropean stock data” ([Panayides et al., 2019, p. 113](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=1&annotation=X5IB86CF))

“BVC is data efficient, but may identify trade aggressors less accurately than “bulk” versions of traditional trade-level algorithms.” ([Panayides et al., 2019, p. 113](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=1&annotation=RKIRSAZK))

“BVC-estimated trade flow is the only algorithm related to proxies of informed trading, however.” ([Panayides et al., 2019, p. 113](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=1&annotation=DP7R2ACJ))

“Finally, we find that after calibrating BVC to trading characteristics in out-of-sample data, it is better able to detect information and to identify trade aggressors.” ([Panayides et al., 2019, p. 113](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=1&annotation=UJ9VR7C8))

“hough information detection is of primary importance, aggressor-signing can be useful as well, for characterising investor clientele behaviour and assessing trading costs, and an ideal algorithm should do both.” ([Panayides et al., 2019, p. 113](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=1&annotation=YGKER9HN))

“This paper helps address these issues by examining the newly developed bulk volume classification algorithm (Easley et al., 2016); hereafter BVC) in modern, low-latency equity markets. BVC uses total volume and price changes within a block of trades to classify order flow into buying and selling volume. We assess BVC performance in terms of the accuracy in finding trade aggressors and ability to capture informative trade flow” ([Panayides et al., 2019, p. 113](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=1&annotation=J5R3ZT5N))

“To help us calibrate the algorithm, we compare BVC’s performance to bulk versions of traditional trade-level algorithms, bulk tick test (Smidt, 1985; Holthausen et al., 1987), and the Lee and Ready (1991) algorithm (hereafter LR)” ([Panayides et al., 2019, p. 113](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=1&annotation=9CJYHHDM))

“We find that BVC can identify trade aggressors as https://doi.org/10.1016/j.jbankfin.2019.04.001 0378-4266/© 2019 Elsevier B.V. All rights reserved” ([Panayides et al., 2019, p. 113](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=1&annotation=FX343W6C))

“114 M.A. Panayides, T.D. Shohfi and J.D. Smith / Journal of Banking and Finance 103 (2019) 113 – 12 9 well as traditional algorithms, and its signed order flow is the only measure that is reliably related to different illiquidity measures shown to capture the trading intensions of informed traders (e.g., Easley et al., 2016).” ([Panayides et al., 2019, p. 114](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=2&annotation=ZFSL47NI))

“We use equities data from NYSE Euronext for 2007 and 2008 and from the London Stock Exchange for 2017 to perform our analyses.” ([Panayides et al., 2019, p. 114](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=2&annotation=PPBSFYP8))

“We begin our analysis by investigating how well BVC classifies trade aggressors in our samples. BVC involves putting trades into blocks, or bars, by either volume or time. A percentage of the block is then classified as buys (the remainder as sells) based upon the movement of prices around the bars. By construction, the BVC algorithm is highly data efficient as it uses aggregate bar-size trading volume and prices, which translates to less than 1% of the trade data points.” ([Panayides et al., 2019, p. 114](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=2&annotation=ZE478IQF))

“The Euronext sample results on finding trade aggressors comport with those in Chakrabarty et al. (2015) and Easley et al. (2016); BVC is not as accurate as bulk versions of traditional trade-level algorithms. This reverses, however, in our more recent 2017 sample. Indeed, even though all three algorithms perform worse in the LSE sample, BVC is the most accurate.” ([Panayides et al., 2019, p. 114](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=2&annotation=IJ336P3H))

“In our next set of analyses, we examine whether BVC can effectively uncover underlying trading intentions in order flow. In today’s markets, researchers and practitioners are increasingly interested in identifying buying or selling pressure that can be destabilising and/or toxic. Information-related order flow will unavoidably disadvantage other traders (retail traders and some institutional traders; O’Hara, 2015).” ([Panayides et al., 2019, p. 114](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=2&annotation=H686BPHL))

“We measure spreads in two ways, using the Corwin and Schultz (2012) high-low spread and calculating intraday effective spreads.” ([Panayides et al., 2019, p. 114](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=2&annotation=XKCG2S5U))

“These results indicate that trade aggressor identification does not convey underlying information in today’s fast markets, where sophisticated traders use smart algorithmic trading to hide their trading intentions and minimise market impact.” ([Panayides et al., 2019, p. 114](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=2&annotation=78FVV98Z))

“For example, informed traders are increasingly relying on passive orders, i.e., limit orders, to disguise themselves in the market (Bouchaud et al., 2009; Menkhoff et al., 2010; Zhang, 2013). Therefore, any traditional trade-level algorithm designed to find a trade’s aggressor will not tell us much about information, no matter how accurate.” ([Panayides et al., 2019, p. 114](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=2&annotation=CZEKPWH5))

“This paper contributes to the nascent literature on trade classification algorithms (including BVC) and low-latency trading, as well as adding to the long list of papers investigating the performance of the LR and tick test algorithms” ([Panayides et al., 2019, p. 114](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=2&annotation=6Q9Q6942))

“In our data from less fragmented, European equity markets, we find that an out-of-sample calibrated BVC successfully captures the aggressor side of trades and information, and it does so without requiring costly, real-time analysis of low latency individual trade and quote data. This suggests that researchers can use BVC to both classify aggressors and measure information, while capturing the data efficiency gains inherent in BVC.” ([Panayides et al., 2019, p. 115](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=3&annotation=C9WHZ8CA))

“The data are time-stamped at the second-level. The second set, from the London Stock Exchange (LSE), is built from the “Tick Data” and “Rebuild Order Book” data.” ([Panayides et al., 2019, p. 116](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=4&annotation=R4CLY9G8))

“First, we choose sample periods. From Euronext, we use April 2007, February 2008, and April 2008, because these months represent different periods of volatility, stable-low, stable-high, and dropping periods of volatility, respectively. From LSE, we choose a much more recent sample, February and April 2017. These data are characterised by lower, more stable volatility and much greater trade volume.” ([Panayides et al., 2019, p. 116](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=4&annotation=T9SNIK4E))

“We also impose standard trade and quote philtres on the data, such as positive price, volume, and quote size, and the bid must be weakly lower than the ask. These philtres result in approximately 210 and 335 gigabytes of trade, quote, and order data for Euronext and LSE respectively” ([Panayides et al., 2019, p. 116](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=4&annotation=NS7SGPMV))

“o identify whether each trade in our sample is a buy or a sell, we follow the definition of trade aggressor/initiator used in Odders-White (2000), which Ellis et al. (2000) note is preferred when a researcher has access to the order book. She defines the trade initiator based on chronological order arrival, that is, the order that arrives second is the order that actually “initiates” the trade. For example, if a market buy order comes in at 11:15AM and hits a limit sell order that had been standing in the book since 11:00AM, that trade would be classified as a buy for our purposes. To determine the trade aggressor in our sample, we first classify fully-executed orders into active and passive categories. An active order is executed at the same date and time as it is submitted to the marketplace, and is, essentially, a market order. A passive order is a non-market order whose execution time is always later than its submission time. In this case, the initiator of a trade will be the opposite buy or sell direction of a matching passive order. Active orders account for approximately 98% of trade aggressors identified across our sample.” ([Panayides et al., 2019, p. 116](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=4&annotation=XJR45S7N))

“The bulk volume classification procedure was developed in ELO for use in the Easley et al. (2012) volume-synchronised probability of informed trading (VPIN) calculation. It is designed to classify bars of trades (i.e., trades put in blocks either by time or” ([Panayides et al., 2019, p. 116](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=4&annotation=C3V5MPNB))

“volume)5 as a percentage of buys and sells, rather than classifying each individual trade.” ([Panayides et al., 2019, p. 117](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=5&annotation=CCK673QG))

“By putting the trades into volume blocks, the algorithm mitigates any impact from order splitting and economises on the number of data points used for classification.” ([Panayides et al., 2019, p. 117](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=5&annotation=3NRCQ65P))

“Next, for each stock-month combination, we calculate the volume-weighted standard deviation of price changes between consecutive bars as shown in formula (1). σPi = √∑ τn =1 Vi,τ (Pi,τ − Pi )2 ∑ τn =1 Vi,τ (1) where Vi, τ is the actual volume of shares traded of stock-month i during the time or volume bar τ which is decomposed into the buy ( ˆ V Buy i,τ ) and sell ( ˆ V Sell i,τ ) volume estimate components. Pi,τ = Pi,τ − Pi,τ −1 is the price change between two consecutive bars. With these available data points, we can then use formula (2) of ELO to calculate BVC’s buy volume for each bar: ˆ V Buy i,τ = Vi,τ · t ( Pi,τ − Pi,τ −1 σPi , df ) ˆ V Sell i,τ = Vi, τ − ˆ V Buy i,τ = Vi, τ · [ 1−t ( Pi,τ − Pi,τ −1 σPi , df )] (2)” ([Panayides et al., 2019, p. 117](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=5&annotation=WGZWFWL4))

“Because BVC puts trades into bars and offsets misclassifications, we compute bulk versions of LR and the tick test to better compare across algorithms (Chakrabarty et al., (2015); ELO, 2016).” ([Panayides et al., 2019, p. 118](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=6&annotation=XVHHJRSN))

“We first note that the accuracy rates are lower than those reported in ELO, who use futures data; their accuracy rates top 94% versus 89% in our analysis.” ([Panayides et al., 2019, p. 118](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=6&annotation=3MRI5F2Q))

“Despite this challenge, BVC performs well in our two samples. Regarding the Euronext sample (columns 1–3 of Table 2) BVC accuracy ranges from 62.62% to 87.90% in Panel A with volume bars, and from 58.66% to 89.68% in Panel B with time bars.” ([Panayides et al., 2019, p. 118](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=6&annotation=WSLL5N9Y))

“n the LSE sample (columns 4–6) we see some important differences from the Euronext results. First, accuracy rates are lower for BVC, as well as bulk tick test and bulk LR.” ([Panayides et al., 2019, p. 118](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=6&annotation=YJJ8K6S2))

“To further investigate the effect of bar size choice on BVC accuracy, we consider scenarios in which bar size can either be “too small” or “too large,” given the distribution of trade sizes. In particular, with respect to time bars, if the bar size is too small, the 15 Bakshi et al. (2003) identify that the returns distribution kurtosis across stocks increases with market capitalisation so, we follow their return distribution analysis by reducing the degrees of freedom in the large and mid-cap group Student’s tdistributions to 0.05 and 0.1, respectively. bar will not contain enough trades to benefit from netting misclassified trades. This reduced netting will impact both bulk tick test/LR and BVC algorithms.” ([Panayides et al., 2019, p. 122](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=10&annotation=V6CZDQZV))

“A potential source of bias in the volume bar BVC applied to equities arises not from the choice of bar size but how that bar size is applied to the data. BVC proposed in ELO does not specify whether volume bars should contain volume equal to bar size or if that size is a minimum amount of volume for each bar.” ([Panayides et al., 2019, p. 124](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=12&annotation=YRCB7NVE))

“Researchers commonly use the Lee and Ready trade classification algorithm if quotes are available or the tick test if not. The recently introduced bulk volume classification (BVC) has an alternative design that makes it much more data efficient.” ([Panayides et al., 2019, p. 128](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=16&annotation=STL3SES9))

“Importantly, we find—using both spread regressions and a returns event study—that BVC has a consistent advantage in capturing information rather than just trade aggressors, which suggests that BVC offers real advantages over methods built on signing individual trades.” ([Panayides et al., 2019, p. 128](zotero://select/library/items/ZTRPVBTG)) ([pdf](zotero://open-pdf/library/items/T3UT4TF6?page=16&annotation=6UR9IQX9))