So far it remains open, what accelerates performance of our predictor. To address this question, we first visualise embeddings and attention of Transformer. 

**Categorical Embeddings**
For the Transformer we know from cref-chapter, that embeddings can capture similarities by arranging related objects closer in embedding space. Visualising the learnt embeddings gives additional insights into the model.

The embeddings are queried from the feature tokenizer in FT-Transformer. The similarity between embeddings is measured by cosine distance in embedding space. The high dimensional embeddings are then projected into 2D-space using $t$-SNE ([[@vandermaatenVisualizingDataUsing2008]]). As straight-forward to interpret, we restrict our analysis to the underlying ($\mathtt{ROOT}$), but note, that it is applicable to any numerical and categorical embeddings. 

Figure cref-jpm illustrates the embeddings exemplary for $\mathtt{SPY}$ and $\mathtt{JPM}$ which allows for some *qualitative* interpretation. -(As our analysis is restricted to two arbitrary underlyings, we encourage the reader to use our interactive visualisation (https://wandb.ai/fbv/thesis/runs/3cirr6nk) for further exploration. ) 

For SPDR S&P 500 ETF (SPY) the most similar embeddings are: iShares Russell 2000 (IWM),  iShares Russel 2000 (WYV), United States oil fund LP (IYS), SPDR S&P 500 ETF (OBM), SPDR S&P 500 ETF (OBV), DIREXION shares ETF (XDT).  This aligns with the intuition, that the embeddings, are glspl-ETF tracking identical or related indices. The model can distinguish glspl-ETF from other securities by the feature issue type. Other similar embeddings include the Citigroup Inc. (WRV), Kohl's Corp. (OSS), GOOGLE Inc. (YTM), Intel Corp. (INTC), which are long term index constituents.

For JPMorgan Chase & Co. (JPM), the most similar embedding is the of the Bank of America (BAC). Other similar embeddings, include financial service providers, such as, the Amerigroup (XGZ) or Janus Henderson group (ZPR). Given these results, it seems plausible, that the model learned to group US financials, albeit no sector information is provided to the model. For related embeddings i. e., Apollo Group (OKO), Autodesk Inc. (ADQ), Centex Corp. (YYV), United Parcel Service of America(YUP), Wild Oats Markets (ZAC), SPDR S&P 500 ETF (SUE), and SPDR Dow Jones Industrial Average (DIA) a similar argumentation does not apply.

While these results indicate that the model is capable to learn some meaningful connections between underlyings, we want to stress its limitations. Both underlyings are among the most frequently traded in our dataset. For infrequent underlyings, embedding are likely close to their random initialisation and hence not meaningful, as no parameter updates takes place. The described problem transfers to handling rare vocabulary items, intensively studied in the context of natural language processing. As the underlying has a subordinate role in classification, we accept this caveat.

![[embeddings-spy-jpm.png]]

**Attention Maps:**
waht model attends to
Attention has a clear interpretation as (...) We follow the approach of ([[@cheferTransformerInterpretabilityAttention2021]]) to derive attention maps.

![[attention-visualisation.png]]

- https://github.com/jessevig/bertviz
- https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/visualization/attention.py

![[attention-maps.png]]

Engineered features like the proximity to quotes, has low importance. The

Figure-attention-maps is the weighted average over all attention blocks and heads, which is a reasonable choice on the network level. Attached to the observation, that attention heads learn different patterns, we . i. e. if the pattern learned by attention heads maps to known trade classification rules or yet undiscovered algorithms.


While the maps has similarities 

Based on these visualisations we cannot reason about how features are used inside the Transformer e.g., if 

- For which trade do we do the limitations?
- What does the pattern look like graphically?

Are subject to interpretation.
Beyond these simple visualisations,

We document these results, but advise the reader to not generalise beyond data.

Again we exclude the $\mathtt{[CLS]}$ token, as it accumulates f

Beyond these simple visualisations, 

The results are 


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
