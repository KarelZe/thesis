

- We construct datasets that suffice (...) and serve as an input to our machine learning models. 
- Refer to chapters (...) for which algorithm requires quote data (e. g., quote rule) and which requires price data (e. g., tick test).
- Provide summary statistics, but wait until pre-processing and generation of true labels are introduced. May add pre-processing and generation of true labels in this chapter.

The following chapter describes how we construct datasets, that suffice the data requirements of classical trade classification rules and for our machine learning models. We also discuss we infer the trade initiator. 

**Data**
Testing the empirical accuracy of our approaches requires option trades where the true initiator is known. To arrive at labelled sample, we combine data from four individual data sources. Our primary source is LiveVol, which records option trades executed at US option exchanges at a transaction level. We limit our focus to option trades executed at the CBOE and ISE. LiveVol contains both trade and the corresponding quote data. Like most proprietary data sources, it does not distinguish the initiator nor does it include the involved trader types. For the CBOE and ISE exchange, the ISE Open/Close Trade Profile and CBOE Open-Close Volume Summary contains the buy and sell volumes for the option series by trader type aggregated on a daily basis. A combination of the LiveVol data set with the ISE Open/Close Trade Profile or the CBOE Open-Close Volume Summary respectively, allows us to infer the trade initiator for a subset of all trades, as we explain next. For a detailed evaluation and use in some of our machine learning models, we acquire additional underlying and option characteristics from IvyDB's OptionMetrics.

**Sample Construction**
Our sample construction follows ([[@grauerOptionTradeClassification2022]]7--9), fostering comparability between both works. We start with 
LiveVol (What does it contain?)
What is the definition of the trade initiator?

ur definition of buyer- and seller-initiated trades is based on the position taken by the Makler. It has the potential advantage of being directly related to many traditional microstructure models which assume the presence of a market maker. It has, on the other hand, the drawback that only transactions involving the Makler as a buyer or a seller can be classified.

Following a standard procedure in literature, we filter out option trades with a trade price equal or less than zero and eliminate trades with a negative or zero trade volume as well as large trades with a trade volume exceeding 10,000,000 contracts. We further remove cancelled or duplicated trades and eliminate entries with multiple underlying symbols for the same root.

The Open/Close datasets for the ISE and CBOE contain the daily buy and sell volumes for the option series by trader type, the trade volume and whether a position was closed or opened. Four trader types are available: customer, professional customer, broker/dealer, and firm proprietary. (...) Trade volumes for customers and professional customers are further detailed into small-sized trades ($\leq 100$ contracts), medium-sized trades ($101-199$ contracts), and large trades. As well as, if an existing position was closed or new position is opened. We first sum buy and sell orders of all trader types to obtain the daily trading volume at the ISE or CBOE per option series and day. Similarly, we calculate the aggregate volumes of customer buy and sell orders identified by the account type »customer«.

To infer the true label, we exploit the fact, that if there were only customer buy or sell orders, hence the customer buy or sell volume equals the daily trading volume, we can confidently sign all transactions for the option series at the specific date and exchange as either buyer- or seller-initiated. The applicability of our labelling approach is limited by the  existence of non-customer or simultaneous customer buy and sell trades. The so-obtained trade initiator is merged with the LiveVol trades of the exchange based on a unique key consisting of the trade date, expiration date, root symbol of the underlying, option type, and strike price.

For the ISE, our matched sample spans from 2 May 2005 to 31 May 2017 and consists of 49,203,747 trades. The period covers the full history of ISE open/close data up to the last date the dataset was available to us.  Our matched CBOE sample consists of 37,155,412 trades between 1 January 2011 and 31 October 2017. The sample period is governed by a paradigm shift in the construction of the CBOE open/close dataset and our most recent trades in LiveVol. Following our initial reasoning to employ semi-supervised methods, we reserve unlabelled customer trades between 24 October 2012 and 24 October 2013 at the ISE for pre-training. We provide further details in cref-[[👨‍🍳Tain-Test-split]].

While our procedure makes the inference of the true trade initiator partly feasible, providing us with a labelled data set, concerns regarding a selection bias due to the excessive filtering have to be raised. We acknowledge these concerns as part of our exploratory data analysis in cref-[[🚏Exploratory Data Analysis]], in which we compare unmerged and merged sub-samples.

**Notes:**
[[🌏Dataset notes]]