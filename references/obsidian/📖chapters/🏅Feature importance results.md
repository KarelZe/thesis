**Attention Maps:**

![[attention_maps_ise.png]]

**SAGE**
```python
    fg_classical = {
        'chg_all_lead (grouped)': ['price_all_lead', 'chg_all_lead'],
        'chg_all_lag (grouped)': ['price_all_lag', 'chg_ex_lag'],
        'chg_ex_lead (grouped)': ['price_ex_lead', 'chg_ex_lead', 'chg_all_lag'],
        'chg_ex_lag (grouped)': ['price_ex_lag'],
        'quote_best (grouped)': ['BEST_ASK', 'BEST_BID', 'prox_best'],
        'quote_ex (grouped)': ['bid_ex', 'ask_ex','prox_ex' ],
        'TRADE_PRICE': ['TRADE_PRICE'],
        }
    
    fg_size = {'size_ex (grouped)': [ 'bid_ask_size_ratio_ex', 'rel_bid_size_ex',  'rel_ask_size_ex', 'bid_size_ex', 'ask_size_ex','depth_ex'], 'TRADE_SIZE': ['TRADE_SIZE']}
    
    fg_ml = {
        "STRK_PRC": ["STRK_PRC"],
        "ttm": ["ttm"],
        "option_type": ["option_type"],
        "root":["root"],
        "myn":["myn"],
        "day_vol":["day_vol"], 
        "issue_type":["issue_type"],
    }
```

- emphasize that these are global feature attributions
- explain definition of feature groups -> Limitation of implementation. Classical classifier sees only a fraction of all features, but features are inherently redundant. -> groups  are aimed to be mutually exclusive
- configuration
	- explain how sampling is done? Why is sampling even necessary. -> Calculating SAGE values is computationally intensive. Recommended sample size is 1024. We tested different sample sizes. Results stabilize at 2048. We set the sample size to 2048.
	- why zero-one loss and why not cross-entropy loss? -> penalize trade classification rules for over-confident predictions
- visualize in subplots how feature importances align (3x3 (benchmark, gbm, fttransformer), distinguished by feature set)
- What features are important?
	- Are there particularly dominant feature groups?
	- How does it align with literature?
	- Are features that are important in smaller feature sets also important in larger feature sets?
	- How does adding more features influence the impact of the other features?
	- Which ones are unimportant? Do models omit features that perform poorly in the empirical setting?
	- Are features important that have not been considered yet? Why's that?
- interpretation
	- Why are size-related features so important? Can we confirm the limit order theory? 

Results:
![[results-sage.png]]
- **Classical Rules** Results align with intuition. Largest improvements come from applying the quote rule (nbbo), which requires quote_best + Trade price, quote (ex) is only applied to a fraction of all trades. The rev tick test is of hardly any importance, as it does not affect classification rules much, nor is it applied often



**Token Embeddings:**
(see graphs)



![[informative-uniformative-features.png]]
([[@grinsztajnWhyTreebasedModels2022]])

**Attention**
- emphasize that these are local feature attributions
- Visualize attention for some trades
- interpret pattern. How does it align with the feature importances from SAGE?

**Rank correlation between approaches**
Compare different feature attributions:
![[feature_attributions_from_attention.png]]
(Found in [[@borisovDeepNeuralNetworks2022]])
