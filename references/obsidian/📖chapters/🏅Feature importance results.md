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

```python
# at quotes (ise)
Int64Index([39342191, 39342189, 39342188, 39342175, 39342174, 39342171,
            39342233, 39342241, 39342238, 39342239, 39342237, 39342193,
            39342194, 39342199, 39342202, 39342204],
           dtype='int64', name='index')
```

```python
# at mid (ise)
Int64Index([39342276, 39342363, 39342387, 39342437, 39342436, 39342428,
            39342464, 39342540, 39342608, 39342598, 39342620, 39342632,
            39342674, 39342781, 39342804, 39342824],
           dtype='int64', name='index')
```

```python
# at quotes (ise correct) index 1
buy_sell                                                    -1
TRADE_SIZE                                                   5
TRADE_PRICE                                                3.5
ask_ex                                                    3.85
ask_size_ex                                               11.0
bid_ex                                                     3.5
bid_size_ex                                               10.0
OPTION_TYPE                                                  P
issue_type                                        Stock option
TRADE_SIZE_binned                                        (3,5]
year_binned                                               2015
ttm_binned                                                <= 1
myn_binned                                           (0.9-1.1]
prox_q_binned                                        at quotes
mid                                                      3.675
all                                                        all
(fttransformer, fttransformer(classical))                   -1
(fttransformer, fttransformer(classical-size))              -1
(fttransformer, fttransformer(ml))                          -1
(classical, tick(ex))                                      1.0
Name: 39342191, dtype: object
```

```python
# at quotes (ise false) index 8
buy_sell                                                 -1
TRADE_SIZE                                               21
TRADE_PRICE                                             4.5
ask_ex                                                  4.5
ask_size_ex                                            21.0
bid_ex                                                  4.4
bid_size_ex                                            41.0
OPTION_TYPE                                               C
issue_type                                           Others
TRADE_SIZE_binned                                       >11
year_binned                                            2015
ttm_binned                                            (3-6]
myn_binned                                        (0.9-1.1]
prox_q_binned                                     at quotes
mid                                                    4.45
all                                                     all
(fttransformer, fttransformer(classical))                -1
(fttransformer, fttransformer(classical-size))            1
(fttransformer, fttransformer(ml))                        1
(classical, tick(ex))                                   1.0
Name: 39342388, dtype: object
```



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

Transformers outperform all previous approaches by a large margin on the gls-ISE dataset. To gain insights into the factors driving this performance, we conduct a qualitative analysis of the attention mechanism and learned embeddings. For an evaluation of feature importances, that suffice for a cross-model comparison, we utilize gls-SAGE, building upon our previous rationale presented in cref-feature-importances.

To gain deeper insights into the factors driving this performance, we conduct a qualitative analysis of the attention mechanism and learned embeddings, focusing specifically on Transformers. To evaluate the importance of features and enable cross-model comparisons, we utilize gls-SAGE, building upon our previous rationale presented in cref-feature-importances.

generate attention maps and probe individual attention heads.

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

Relation between attention techniques

- put special emphasise in analysis Transformer performance, as it is 
- arrange in graphics below. arrange like puzzle blocks, what is done by which approach
- pick up and describe two trades

- general overview in 

![[model-wide-attention.png]]
(from [[@coenenVisualizingMeasuringGeometry2019]])

The data for our first experiment is a corpus of parsed sentences from the Penn Treebank [13]. This dataset has the constituency grammar for the sentences, which was translated to a dependency grammar using the PyStanfordDependencies library [14]. The entirety of the Penn Treebank consists of 3.1 million dependency relations; we filtered this by using only examples of the 30 dependency relations with more than 5,000 examples in the data set. We then ran each sentence through BERTbase, and obtained the model-wide attention vector (see Figure 1) between every pair of tokens in the sentence, excluding the [SEP ] and [CLS] tokens. This and subsequent experiments were conducted using PyTorch on MacBook machines.

While [6] analyzed context embeddings, another natural place to look for encodings is in the attention matrices. After all, attention matrices are explicitly built on the relations between pairs of words.

This broader analysis shows that BERT’s attention heads pay little attention to the current token but rather specialize to attend heavily on the next or previous token, especially in the earlier layers.

A substantial amount of BERT’ attention focuses on a few special tokens such as the deliminator token [SEP] which means that such tokens play a vital role in BERT’s performance. The figure below shows the average attention behavior in each layer for some special tokens such as [CLS] and [SEP].

With a few more creative tests (see paper for full details), the authors found that BERT’s attention maps have a fairly thorough representation of English syntax.

Upon further investigation of the individual attention heads behavior for a given layer, the authors found that some heads behave similarly, possible due to some attention weights being zeroed-out via dropout. A surprising result, given that other researchers found that encouraging different behavior in attention heads improves a Transformer’s performance. There is more opportunity to conduct extended analysis to help further understand these behaviors in the attention layer.

The example given here is correctly classified. Crucially, only in the first couple of layers, there are some distinctions in the attention patterns for different positions, while in higher layers the attention weights are rather uniform. Figure 2 (left) gives raw attention scores of the CLS token over input tokens (x-axis) at different layers (y-axis), which similarly lack an interpretable pattern.These observations reflect the fact that as we go deeper into the model, the embeddings are more contextualized and may all carry similar information. This underscores the need to track down attention weights all the way back to the input layer and is in line with findings of Serrano and Smith (2019), who show that attention weights do not necessarily correspond to the relative importance of input tokens. ([[@abnarQuantifyingAttentionFlow2020]])

add a CLS token and use its embedding in the final layer as the input to the classifier

Figures 2 and 3 show the weights from raw attention, attention rollout and attention flow for the CLS embedding over input tokens (x-axis) in all 6 layers (y-axis) for three examples. The first example is the same as the one in Figure 1. The second example is “the article on NNP large systems ”. The model correctly classifies this example and changing the subject of the missing verb from “article” to “articles” flips the decision of the model. The third example is “here the NNS differ in that the female ”, which is a miss-classified example and again changing “NNS” (plural noun) to “NNP” (singular proper noun) flips the decision of the model. For all cases, the raw attention weights are almost uniform above layer three (discussed before). raw attention attention rollout attention flow (a) “The author talked to Sara about mask book.” raw attention attention rollout attention flow (b) “Mary convinced John of mask love.” Figure 4: Bert attention maps. We look at the attention weights from the mask embedding to the two potential references for it, e.g. “author” and “Sara” in (a) and “Mary” and “John” in (b). The bars, at the left, show the relative predicted probability for the two possible pronouns, “his” and “her”. In the case of the correctly classified example, we observe that both attention rollout and attention flow assign relatively high weights to both the subject of the verb, “article’ and the attractor, “systems”. For the miss-classified example, both attention rollout and attention flow assign relatively high scores to the “NNS” token which is not the subject of the verb. This can explain the wrong prediction of the model.

We claim that one-layer attention-only transformers can be understood as an ensemble of a bigram model and several "skip-trigram" models (affecting the probabilities of sequences "A… BC"). 12 Intuitively, this is because each attention head can selectively attend from the present token ("B") to a previous token ("A") and copy information to adjust the probability of possible next tokens ("C"). (https://transformer-circuits.pub/2021/framework/index.html)

- the innerworkings of transformers are not fully-understood yet. https://transformer-circuits.pub/2021/framework/index.html

Compare attention of pre-trained Transformer with vanilla Transformer?



The system visualizes these 1,000 context embeddings using UMAP [15], generally showing clear clusters relating to word senses. Different senses of a word are typically spatially separated, and within the clusters there is often further structure related to fine shades of meaning.
