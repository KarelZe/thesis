### Attention visualization


One advantage of using transformer-based models is that attention comes with some interpretability, in contrast, MLPs are hard to interpret. In particular, when we use only one transformer stage, the attention maps reveal which features and which data points are being used by the model to make decisions. We use MNIST data to examine how self-attention and intersample attention behave in our models. While MNIST is not a typical tabular dataset, it has the advantage that its features can be easily visualized as an image

Transformer-based models are interpretable by de. Due to the limination, that TabTransformer ([[@huangTabTransformerTabularData2020]]) only feds categorical features through the transformer, only feature-attributions for non-contious features can be estimated.

TabNet, TabTransformer, and gradient-boosted trees are interpretable by design but rely on model-specific techniques such as feature activation masks found only in transformer-based models rendering them useless for cross-model comparisons. 

Transformers have the advantage of 

capabilities

Transformers have the additional advantage of being interpretable by visualizing attention in attention maps [[@somepalliSAINTImprovedNeural2021]] (p. 8). An example is shown in Figure (...).

![[attention-map-borisov.png]]
(from [[@borisovDeepNeuralNetworks2022]])

![[attention-map-somepalli.png]]
[[@somepalliSAINTImprovedNeural2021]]

In the *tabular domain* different approaches for obtaining attention from multiple attention heads and transformer blocks have been explored. [[@somepalliSAINTImprovedNeural2021]] and [[@borisovDeepNeuralNetworks2022]] gather attention maps from the first attention layer only. [[@borisovDeepNeuralNetworks2022]] obtain feature attributions by taking the diagonal of the attention matrix or through column-wise summation. In contrast, [[@gorishniyRevisitingDeepLearning2021]] leverage all attention matrices by averaging over multiple transformer blocks, attention heads, and samples to obtain global feature attributions. In absence of a ground truth for the true feature attribution, the *optimal* approach is to be determined.

While not explored systematically for the tabular domain, the specific roles of different attention heads have been studied in transformer-based machine translation (see e. g., [[@voitaAnalyzingMultiHeadSelfAttention2019]], [[@tangAnalysisAttentionMechanisms2018]]).  [[@voitaAnalyzingMultiHeadSelfAttention2019]] observe, that heads in the multi-headed self-attention mechanism serve distinct purposes like learning positional or syntactic relations between tokens.  Transferring their result back to the tabular domain, averaging over multiple heads may lead to undesired obfuscation effects. 

<mark style="background: #ABF7F7A6;">Study other approaches here https://m.youtube.com/watch?v=A1tqsEkSoLg</mark>

In consequence, we study the attributions from different attention heads in the encoder separately. Inline with previous research, feature attributions are averaged in the first attention layer or over all transformer blocks. Additionally, we consider  combining multiple attention matrices through the outer product. The level of agreement between attributions from attention maps and kernel SHAP is quantified by calculating Spearman's rank correlation between them.
