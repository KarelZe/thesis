The basic trade classification rules from [[🔢Basic rules]] can be combined into a hybrid algorithm to enforce a universal applicability to all trades and or to improve the classification performance.

Popular variants include the LR algorithm, the EMO rule, and CLVN method. All three algorithms utilize the quote and tick rule to a varying degree, as depicted in Figure a) - c)., Both rules are selected based on the proximity of the trade price to the quotes. We study all algorithms in detail in ... . 

([[@grauerOptionTradeClassification2022]] 18) combine rules like the trade size rule, depth rule, as well as other basic or hybrid rules through stacking. This approach is notably different from the aforementioned algorithms, as the applied rule is no longer dependent on the proximity to the quotes, but rather on the classifiability of the trade with the primary rules and their ordering. Like before all rules are applied mutually-exclusively. Theoretically, all algorithms can be stacked. As such, we don't discuss this concept as a separate rule.

**Notes:**
[[🔢Hybrid rules notes]]