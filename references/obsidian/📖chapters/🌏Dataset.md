

- We construct datasets that suffice (...) and serve as an input to our machine learning models. 
- Refer to chapters (...) for which algorithm requires quote data (e. g., quote rule) and which requires price data (e. g., tick test).
- Provide summary statistics, but wait until pre-processing and generation of true labels are introduced. May add pre-processing and generation of true labels in this chapter.

The following chapter describes how we construct datasets, that suffice the data requirements of classical trade classification rules and for our machine learning models. We also discuss we infer the trade initiator. 

**Data**
Testing the empirical accuracy of our approaches requires option trades where the true initiator is known. To arrive at labelled sample, we combine data from four individual data sources. Our primary source is LiveVol, which records option trades executed at US option exchanges at a transaction level. We limit our focus to option trades executed at the CBOE and ISE. LiveVol contains both trade and the corresponding quote data. Like most proprietary data sources, it does not distinguish the initiator nor does it include the involved trader types. For the CBOE and ISE exchange, the ISE Open/Close Trade Profile and CBOE Open-Close Volume Summary contains the buy and sell volumes for the option series by trader type aggregated on a daily basis. A combination of the LiveVol data set with the ISE Open/Close Trade Profile or the CBOE Open-Close Volume Summary respectively, allows us to infer the trade initiator for a subset of all trades, as we explain next. For a detailed evaluation and use in some of our machine learning models, we acquire additional underlying and option characteristics from IvyDB's OptionMetrics.

**Sample Construction**
Our sample construction follows ([[@grauerOptionTradeClassification2022]]7--9), fostering comparability between both works. 

The 337,234,107

For the ISE, our matched sample spans from 2 May 2005 to 31 May 2017 and consists of 49,203,747 trades. The period covers the full history of ISE open/close data up to the last date the dataset was available to us.  Our matched CBOE sample consists of 37,155,412 trades between 1 January 2011 and 31 October 2017. The sample period is governed by a paradigm shift in the construction of the CBOE open/close dataset and our most recent trades in LiveVol. 

Following our initial reasoning to employ semi-supervised methods, we reserve unlabelled customer trades between 24 October 2012 and 24 October 2013 at the ISE for pre-training. We provide further details in cref-[[👨‍🍳Tain-Test-split]].


The ISE Open/Close spans a period from May 2, 2005 to May 31, 2017. 

Our matched samp

Following a common track in literatu4ree

The final




Our two Open/Close datasets from the ISE and the CBOE each contain daily buy and sell trading volumes for the option series traded at the two exchanges. The volume is disaggregated by whether the trades open new or close existing option positions and is categorized by customer, professional customer, firm proprietary, and firm broker/dealer account type.


and information on the exchange on which the trade is executed. 



For use in our machine learning models and and a enriched evaluation, we acquire underlying and option characteristics from OptionMetrics.

We datasets.


and whether an existing position was closed or opened aggregated o

Live vol It contains the trade price, the trade volume, as well 


  provides intra-day trade data at the transaction level. LiveVol, however, does not contain the trade initiator.

**Notes:**
[[🌏Dataset notes]]