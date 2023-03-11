Every option trade has a buyer and seller side. For many problems in option research, it's vital to determine the party that initiated the transaction. The trade initiator is binary and can either be the seller or the buyer. 


Consequently, we denote it by $y \in \{0,1\}$, whereby $y=0$ indicates a trade that was initiated by a seller and $y=1$ by a buyer. As the trade initiator is commonly not provided with the option data sets, it must be inferred using trade classification algorithms ([[@easleyOptionVolumeStock1998]] 453).


The following section introduces basic rules for option trade classification. We start with the classical quote and tick rule and continue with the more recent depth and trade size rule. Our focus is on classification rules, that sign trades on a trade-by-trade basis. Consequently, we exclude algorithms for aggregated trades, like the BVC algorithm of ([[@easleyFlowToxicityLiquidity2012]]1466--1468).

**Notes:**

[[🔢Basic rules notes]]