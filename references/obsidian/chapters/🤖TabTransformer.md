#tabular-data #tabtransformer #transformer #embeddings #categorical #continuous 


![[tab_transformer.png]]
(own drawing. Inspired by [[@huangTabTransformerTabularData2020]]. Top layers a little bit different. They write MLP. I take the FFN with two hidden layers and an output layer. <mark style="background: #FFB8EBA6;">Better change label to MLP</mark>; Also they call the <mark style="background: #FFB8EBA6;">input embedding a column embedding)</mark> ^87bba0

Motivated by the success of (cp. [[@devlinBERTPretrainingDeep2019]]) of contextual embeddings in natural language processing, [[@huangTabTransformerTabularData2020]]  propose with *TabTransformer* an adaption of the classical Transformer for the tabular domain. *TabTransformer* is *encoder-only* and features a stack of Transformer layers (see chapter [[🤖Transformer]] or [[@vaswaniAttentionAllYou2017]]) to learn contextualized embeddings of categorical features from their parametric embeddings, as shown in Figure ([[#^87bba0]]]).  The transformer layers, are identical to those found in [[@vaswaniAttentionAllYou2017]] featuring multi-headed self-attention and a norm-last layer arrangement. Continuous inputs are normalized using layer norm ([[@baLayerNormalization2016]]) , concatenated with the contextual embeddings, and input into a multi-layer peceptron. More specifically, [[@huangTabTransformerTabularData2020]] (p. 4; 12) use a feed-forward network with two hidden layers, whilst other architectures and even non-deep models, such as [[🐈gradient-boosting]], are applicable.<mark style="background: #FFB8EBA6;"> (downstream network?)</mark> Thus, for strictly continuous inputs, the network collapses to a multi-layer perceptron with layer normalization.

Due to the tabular nature of the data, with features arranged in a row-column fashion, the token embedding (see chapter [[🛌Token Embedding]]) is replaced for a *column embedding*. 

- no positional encoding

- TabTransformers applies a sequence of multi-head attention based transformer layers to obtain contextual embeddings from parametric embeddings.


![[tabtransformer-explanation.png]]

![[column-embeddings.png]]

<mark style="background: #FFB8EBA6;">(What is done and how does it work?)</mark>
<mark style="background: #FFB8EBA6;">(How is the embedding different from the classical transformer?)</mark>

In large-scale experiments [[@huangTabTransformerTabularData2020]]  (p. 5 f.) can show, that the use of contextual embeddings elevates both the robustness to noise and missing data of the model. For various binary classification tasks, the TabTransformer outperforms other deep learning models e. g., vanilla multi-layer perceptrons in terms of *area under the curve* (AUC) and can compete with [[🐈gradient-boosting]].  

Yet, embedding and contextualizing categorical inputs remains imperfect, as no continuous data is considered for the contextualized embeddings and correlations between categorical and continuous features are lost due to the precessing in different subnetworks ([[@somepalliSAINTImprovedNeural2021]]; p. 2).
In a small experimental setup, [[@somepalliSAINTImprovedNeural2021]] (p. 8) address this concern for the TabTransformer by also embedding continuous inputs, which leads the substantial improvements in (AUC) . 

Their observation integrates with a wider strand of literature that suggests models can profit from embedding continuous features ([[@somepalliSAINTImprovedNeural2021]] (p. 8), [[@gorishniyRevisitingDeepLearning2021]] (p. ), [[@gorishniyEmbeddingsNumericalFeatures2022]] (p. )). To dwell on this idea, we introduce the [[🤖FTTransformer]], a transformer that contextualizes embeddings of all inputs  in the subsequent section.

---

## Notes
- What must be known from another chapter: Why does it make sense to use transformer-based models for tabular data? -> Learn contetualized embeddings.
- adapts the classical transformer of vaswani at el to the tabular domain
- TabTransformer is an encoder-only model
- learns contextualized embeddings of the categorical input
- contextual embeddings have been heavily studied in nlp. Things are different in the tabular domain. (p. 2)
- Contextual embeddings  are highly successful in nlp.
- Improved properties with regard to robustness through the use of contextual embeddings.
- no decoder, no positional encoding
- on a high level overwie model consists of an encoder for categorical inputss
- correlations are broken. Numerical input still procesed by mlp and may overfit, be not robust to noise / missing values.
- They conjecture that the improved robustness comes from the contextual embeddings [[@huangTabTransformerTabularData2020]].
- In their comparsion TabTransformer significantly outperforms MLP and recent deep networks for tabular data while matching the performance of tree-based ensemble models (GBDT).
- not just simple categorical embeddings but contextual embedding
- Use of Post-Norm (Hello [[🤖TabTransformer]]) has been deemed outdated in Transformers due to a more fragile training process (see [[@gorishniyRevisitingDeepLearning2021]]). May swap (?).


![[comparison-ft-tab-transformer.png]]
(found on reddit. Haven't found the original source yet. Guess it's taken from some yandex slide.)

## Notes from Talk by Zohar Karnin
(see here: https://www.youtube.com/watch?v=-ZdHhyQsvRc)

- networks can only interpret numbers.
	- Continuous: Easy for numerical input
	- Categorical: convert to embedding

- issues with categorical data
	- rare categories -> similar to infrequent words
	- similar categories; have a head-start by not initializing randomly -> semantic meaning of words
	- use transfer learning 
	- handle context

- pre-training is boroughed from nlp. Two approaches:
	- MLM: convert some categories to missing embedding, reconstruct the category
	- Replaced token detection: Replace category with random replacement, detect replaced
- not necessarily a mlp followed by transformer. Could be any network
- in a semi-supervised comparsion they also compared gbrts on pseudo labelled data



---
Related:
- [[@huangTabTransformerTabularData2020]] propose TabTransformer
- [[@vaswaniAttentionAllYou2017]] propose the Transformer architecture
- [[@cholakovGatedTabTransformerEnhancedDeep2022]] Rubish paper that extends the TabTransformer
- Video by the author: https://www.youtube.com/watch?v=-ZdHhyQsvRc