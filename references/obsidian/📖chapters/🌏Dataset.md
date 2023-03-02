- Our empirical analyses use option data
- Distinguish between subchapter **data** and **sample construction**
- We construct datasets that suffice (...) and serve as an input to our machine learning models. 
- Refer to chapters (...) for which algorithm requires quote data (e. g., quote rule) and which requires price data (e. g., tick test).
- Provide summary statistics, but wait until pre-processing and generation of true labels are introduced. May add pre-processing and generation of true labels in this chapter.

**Data**
Testing the empirical accuracy of our approaches requires option trades where the true initiator is known. To arrive at labelled sample, we combine data from four individual data sources. Our primary source is LiveVol, which records option trades executed at US option exchanges at a transaction level. We limit our focus to option trades executed at the CBOE and ISE. LiveVol contains both trade and the corresponding quote data. Like most proprietary data sources, it does not distinguish the initiator nor does it include the involved trader types. For the CBOE and ISE exchange, the ISE Open/Close Trade Profile and CBOE Open-Close Volume Summary contains the buy and sell volumes for the option series by trader type aggregated on a daily basis. A combination of the LiveVol data set with the ISE Open/Close Trade Profile or the CBOE Open-Close Volume Summary respectively, allows us to infer the trade initiator for a subset of all trades, as we explain next. For an evaluation and use in some of our machine learning models, we acquire additional underlying and option characteristics from IvyDB's OptionMetrics.

**Sample Construction**
Our sample construction follows ([[@grauerOptionTradeClassification2022]]7--9), fostering comparability between both works. W

Following a common track in literatu4ree

The final


Our analyses in the first step are based on the matched ISE sample, for which we have data for a twelve-year period from May 2, 2005 to May 31, 2017. The starting date corresponds to the coverage of ISE Open/Close data going back to May 2005 and the end date is governed by the availability of the dataset to us. The matched ISE sample contains 49,203,747 option trades. In the second step, we test our newly developed classification algorithms out-of8 Electronic copy available at: https://ssrn.com/abstract=409847

sample on the CBOE data after registering the results from the first step with the OSF (see footnote 2). The CBOE dataset begins on January 1, 2011 as there was a structural change in how CBOE Open/Close data is constructed at the beginning of 2011. The CBOE sample period ends on October 31, 2017 as our LiveVol data goes to this date. The matched CBOE sample contains 37,155,412 option trades


Our two Open/Close datasets from the ISE and the CBOE each contain daily buy and sell trading volumes for the option series traded at the two exchanges. The volume is disaggregated by whether the trades open new or close existing option positions and is categorized by customer, professional customer, firm proprietary, and firm broker/dealer account type.


and information on the exchange on which the trade is executed. 



For use in our machine learning models and and a enriched evaluation, we acquire underlying and option characteristics from OptionMetrics.

We datasets.


and whether an existing position was closed or opened aggregated o

Live vol It contains the trade price, the trade volume, as well 


  provides intra-day trade data at the transaction level. LiveVol, however, does not contain the trade initiator.

**Notes:**
[[🌏Dataset notes]]