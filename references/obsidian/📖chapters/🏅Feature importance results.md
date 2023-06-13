So far it remains open, what accelerates performance of our predictor. To address this question, we first visualise embeddings and attention of Transformer. 

**Categorical Embeddings**
For the Transformer we know from cref-chapter, that embeddings can capture similarities by arranging related objects closer in embedding space. Visualising the learnt embeddings gives additional insights into the model.

The embeddings are queried from the feature tokenizer in FT-Transformer. The similarity between embeddings is measured by cosine distance in embedding space. The high dimensional embeddings are then projected into 2D-space using $t$-SNE ([[@vandermaatenVisualizingDataUsing2008]]). As straight-forward to interpret, we restrict our analysis to the underlying ($\mathtt{ROOT}$), but note, that it is applicable to any numerical and categorical embeddings. 

Figure cref-jpm illustrates the embeddings exemplary for $\mathtt{SPY}$ and $\mathtt{JPM}$ which allows for some *qualitative* interpretation. -(As our analysis is restricted to two arbitrary underlyings, we encourage the reader to use our interactive visualisation (https://wandb.ai/fbv/thesis/runs/3cirr6nk) for further exploration. ) 

For SPDR S&P 500 ETF (SPY) the most similar embeddings are: iShares Russell 2000 (IWM),  iShares Russel 2000 (WYV), United States oil fund LP (IYS), SPDR S&P 500 ETF (OBM), SPDR S&P 500 ETF (OBV), DIREXION shares ETF (XDT).  This aligns with the intuition, that the embeddings, are glspl-ETF tracking identical or related indices. The model can distinguish glspl-ETF from other securities by the feature issue type. Other similar embeddings include the Citigroup Inc. (WRV), Kohl's Corp. (OSS), GOOGLE Inc. (YTM), Intel Corp. (INTC), which are long term index constituents.

For JPMorgan Chase & Co. (JPM), the most similar embedding is the of the Bank of America (BAC). Other similar embeddings, include financial service providers, such as, the Amerigroup (XGZ) or Janus Henderson group (ZPR). Given these results, it seems plausible, that the model learned to group US financials, albeit no sector information is provided to the model. For related embeddings i. e., Apollo Group (OKO), Autodesk Inc. (ADQ), Centex Corp. (YYV), United Parcel Service of America(YUP), Wild Oats Markets (ZAC), SPDR S&P 500 ETF (SUE), and SPDR Dow Jones Industrial Average (DIA) a similar argumentation does not apply.

While these results indicate that the model is capable to learn some meaningful connections between underlyings, we want to stress its limitations. Both underlyings are among the most frequently traded in our dataset. For infrequent underlyings, embedding are likely close to their random initialisation and hence not meaningful, as no parameter updates takes place. The described problem transfers to handling rare vocabulary items, intensively studied in the context of natural language processing. As the underlying has a subordinate role in classification, we accept this caveat.

![[embeddings-spy-jpm.png]]

https://github.com/jessevig/bertviz/blob/master/bertviz/head_view.py



**Attention Maps:**

“The self-attention mechanism links each of the tokens to the [CLS] token. The strength of this attention link can be intuitively considered as an indicator of the contribution of each token to the classification. While this is intuitive, given the term “attention”, the attention values reflect only one aspect of the Transformer network or even of the selfattention head” (Chefer et al., 2021, p. 789)

Following our reasoning in cref-[[🧭Feature Importance Measure]], we estimate attention maps with the approach of  ([[@cheferTransformerInterpretabilityAttention2021]]) and interpret them qualitatively. Attention maps are local interpretability measures, providing interpretability on the trade level. For visualisation purposes we focus on a small subsets of trades, where performance is particularly strong and  select a random subsets of $16$ mid spread trades and $16$ trades at the quotes from the gls-ise test set. The attention map are visualised in cref-attention-maps. 
![[attention-maps.png]]
Each column represents a single trade. Following the standard rationale all x-axis indicates the trade, y-axis denotes the features

We not that, graphically, the trade price and quotes at the exchange level are most important, and are also used most frequently. 

Derived features, such as the proximity to quotes, or the change in trade price, are hardly important 

we focus on a small subset of trades at the  mid and quotes

Analogous to rule



**Attention**
- emphasize that these are local feature attributions
- Visualize attention for some trades
- interpret pattern. How does it align with the feature importances from SAGE?


waht model attends to
Attention has a clear interpretation as (...) We follow the approach of ([[@cheferTransformerInterpretabilityAttention2021]]) to derive attention maps.

As the 



Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.




![[layer_3_head_0.png]]
(layer 3, head 0)
![[layer_3_head_4.png]]
(layer 3, head 4)
![[layer_3_head_8.png]]
(layer 3, head 8)
Similar to ([[@vaswaniAttentionAllYou2017]]15) (Based on code of [[@clarkWhatDoesBERT2019]]) (change order(left to right))

Figure 5: BERT attention heads that correspond to linguistic phenomena. In the example attention maps, the darkness of a line indicates the strength of the attention weight. All attention to/from red words is colored red; these colors are there to highlight certain parts of the attention heads’ behaviors. For Head 9-6, we don’t show attention to [SEP] for clarity. Despite not being explicitly trained on these tasks, BERT’s attention heads perform remarkably well, illustrating how syntax-sensitive behavior can emerge from self-supervised training alone.

Figure 5 shows some examples of the attention behavior. While the similarity between machinelearned attention weights and human-defined syntactic relations are striking, we note these are relations for which attention heads do particularly well on. There are many relations for which BERT only slightly improves over the simple baseline, so we would not say individual attention heads capture dependency structure as a whole. We think it would be interesting future work to extend our analysis to see if the relations well-captured by attention are similar or different for other languages.


Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.


![[attention-visualisation.png]]

- https://github.com/jessevig/bertviz
- https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/visualization/attention.py

Engineered features like the proximity to quotes, has low importance. The

Figure-attention-maps is the weighted average over all attention blocks and heads, which is a reasonable choice on the network level. Attached to the observation, that attention heads learn different patterns, we . i. e. if the pattern learned by attention heads maps to known trade classification rules or yet undiscovered algorithms.


Based on these visualisations we cannot reason about how features are used inside the Transformer e.g., if 

- For which trade do we do the limitations?
- What does the pattern look like graphically?

Are subject to interpretation.
Beyond these simple visualisations,

We document these results, but advise the reader to not generalise beyond data.

Again we exclude the $\mathtt{[CLS]}$ token, as it accumulates f

Beyond these simple visualisations, 

The results are 

We have proposed a series of analysis methods for understanding the attention mechanisms of models and applied them to BERT. While most recent work on model analysis for NLP has focused on probing vector representations or model outputs, we have shown that a substantial amount of linguistic knowledge can be found not only in the hidden states, but also in the attention maps. We think probing attention maps complements these other model analysis techniques, and should be part of the toolkit used by researchers to understand what neural networks learn about language.


----

Visualisation of embeddings and attention is Transformer-specific. 


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

