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

Drawing on theory in cref-[[🧭Feature Importance Measure]], we employ the methodology of ([[@cheferTransformerInterpretabilityAttention2021]]2--4) to generate attention maps, which we then qualitatively interpret. These attention maps offer transparency at the *trade level*. To aid visualization, we focus to a subsets of trades, where performance of Transformers is particularly strong and select num-16 mid spread trades and num-16 trades at the quotes from the gls-ise test set. The resulting attention map for the Transformer trained on FS 3 are shown in cref-attention-maps. 

![[attention-maps.png]]

Attention Maps of Transformer Trained on ISE data set
Each column represents a single trade. Following the standard rationale all x-axis indicates the trade, y-axis denotes the features.
We exclude the CLS  token as it accumulates most feature importances. Darkness of pixel represents strength of attention score.

Visually, the trade price and quotes at the exchange or inter-exchange level are most important and most frequently used. This aligns with our intuition, as these features are core to the quote rule and numerous hybrid algorithms. Also, quote-based algorithms are among the best performing in our dataset. Aside from the trade price, features required to estimate the tick rule attain only low attention scores. Considering the devastating performance of tick-based algorithms in option trade classification, this is expected. Features from the depth and trade size rule, such as the trade size, are used selectively. For classification of trades at the quotes, option-specific feature like the issue type, moneyness, time to maturity, or daily trading volume of the option series receive relatively high attention scores. Overall, derived features, like the proximity to quotes, attain only low attention scores, which can be indication that the Transformer can synthesise the feature from the *raw* bid, ask and trade price itself.

The model assigns the highest attention scores to features found in the quote rule and hybrids there-off. Due to the existing link to rule-based trade classification, it is tempting to explore, if the fine-grained patterns learned by specific attention heads translate to existing trade classification rules i. e., if specific tokens attend to features that are jointly used in rule-based classification. This information is sacrificed when aggregating over multiple attention heads and layers, as done for cref-fig, but readily available from individual attention heads. To further analyse this aspect, we adapt the approach of ([[@clarkWhatDoesBERT2019]]4) to our setting. 

![[layer_3_head_0.png]]
(layer 3, head 0)
![[layer_3_head_4.png]]
(layer 3, head 4)
![[layer_3_head_8.png]]
(layer 3, head 8)

Figure cref-fig show three examples of attention heads involved in classifying a trade *at the quote*. The remaining attention heads are visualised in cref-appendix. Each subplots depicts the features to which the classification token ($\mathtt{[CLS]}$) attends to. The attention weight determines the intensity of the line between the two. Referring to the results from the appendix, we note that attention heads learn diverse patterns, as most heads attend to different tokens at once learning different relations. For earlier layers in the network, the classification tokens gathers from multiple tokens with non-obvious patterns, whereas for the final self-attention layers, attention heads specialise in relations that seems related to rule-based trade classification. In Fig-a) the classification token gathers simultaneously from multiple price / size-related features similar to the trade size rule. Fig-b depicts a neighbouring classification head that focuses solely on the change in trade price similar to the tick rule. Finally, fig-c) is an alike to the gls-LR algorithm with additional dependencies on the time to maturity. For other attention heads it remains open what purpose they serve in the network. While the similarity is striking, it requires a more rigorous analysis. It would be interesting for future work to extend this analysis, as it potentially enables to uncover new rule-based approaches as well as better understand Transformer-based trade classification as a whole.


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

