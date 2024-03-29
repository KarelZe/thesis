In absence of a universal definition of the trade initiator, the following definitions are frequently found in research. ([[@odders-whiteOccurrenceConsequencesInaccurate2000]] 267) adapts a chronological view based on the order arrival. She defines the trade initiator as the party (buyer or seller) who places their order last, chronologically. This definition requires knowledge about the order submission times. In contrast, ([[@leeInferringInvestorBehavior2000]]94--97) put forth a definition based on the party in demand for immediate execution. <mark style="background: #FFF3A3A6;">The initiator is assumed to be the investor whose market order is matched against standing limit orders.</mark> This definition remains ambiguous for trades that result from crossed limit orders, matched market orders, or batched orders ([[@leeInferringInvestorBehavior2000]]94--95). Independent from the order type and submission time, ([[@ellisAccuracyTradeClassification2000]]533) define the trade initiator based on the position of the broker and customer relative to the market maker. When a customer or dealer buys from the market maker, the trade it is considered buyer-initiated, as it wouldn't occur without the customer. Similarily for sells.

In absence of a universal definition of the trade initiator, the following views are prevailing in research.

\emph{Chronological definition:} \textcite[][267]{odders-whiteOccurrenceConsequencesInaccurate2000} adapts a chronological view based on the order arrival. She defines the initiator of the trade as the party (buyer or seller) who places their order last, chronologically. This definition requires knowledge about the order submission times.

\emph{Immediacy definition:} In contrast, \textcite[][94--97]{leeInferringInvestorBehavior2000} equate the trade initiator with the the party in demand for immediate execution. Thus, traders placing market orders, immediately executable at whatever price, or executable limit orders, are considered the trade initiator. By contrast, for non-executable limit orders, that do not demand immediate execution, the trading party is the non-initiator. This definition remains ambiguous for trades resulting from crossed limit orders, matched market orders, or batched orders \autocite[][94--95]{leeInferringInvestorBehavior2000}.

\emph{Positional definition:} Independent from the order type and submission time, \textcite[][533]{ellisAccuracyTradeClassification2000} deduce their definition of the trade initiator based on the position of the involved parties opposite the market maker or broker. The rationale is, that the market maker or broker only provides liquidity to the investor and the trade would not exist without the initial investor's demand. 

Regardless of the definition used, the trade initiator is binary and can either be the seller or the buyer. Henceforth, we denote it by $\gls{y} \in \mathcal{Y}$ with $\mathcal{Y}=\{-1,1\}$, with $y=-1$ indicating a seller-initiated and $y=1$ a buyer-initiated trade. The predicted trade initiator is denoted as $\hat{y}$.



The applicability 

In all cases, we denote it by $y \in \{0,1\}$, whereby $y=0$ indicates a trade that was initiated by a seller and $y=1$ by a buyer. As the trade initiator is commonly not provided with the option data sets, it must be inferred using trade classification algorithms ([[@easleyOptionVolumeStock1998]] 453). We use $\hat{y}$ to distinguish the predicted from the observed trade initiator.

The following section introduces basic rules for option trade classification. We start with the classical quote and tick rule and continue with the more recent depth and trade size rule. Our focus is on classification rules, that sign trades on a trade-by-trade basis. Consequently, we exclude algorithms for aggregated trades, like the BVC algorithm of ([[@easleyFlowToxicityLiquidity2012]]1466--1468).

**Notes:**
[[🔢Trade Initiator notes]]

