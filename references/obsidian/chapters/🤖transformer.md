
![[classical_transformer_architecture.png]]
(own drawing after [[@daiTransformerXLAttentiveLanguage2019]], <mark style="background: #FFB8EBA6;">use L instead of N, left encoder and right decoder. Add label.</mark>) ^2cf7ee

## Overview

The *Transformer* is a neural network architecture proposed by [[@vaswaniAttentionAllYou2017]] (p. 2 f.) for sequence-to-sequence modelling, such as machine translation. Since its introduction it has become ubiquitous in natural language processing ([[@lampleLargeMemoryLayers2019]], p. 3; ...). Its success for language representations has also led to adaptions for image representations ([[@parmarImageTransformer2018]], [[@carionEndtoEndObjectDetection2020]]) (found in [[@tayEfficientTransformersSurvey2022]]), as well as tabular data representations ([[@huangTabTransformerTabularData2020]], [[@somepalliSAINTImprovedNeural2021]], [[@gorishniyRevisitingDeepLearning2021]]).

<mark style="background: #D2B3FFA6;">(shortest description A transformer starts with a token embedding, followed by a series of “residual blocks”, and finally a token unembedding. Each residual block consists of an attention layer, followed by an MLP layer. Both the attention and MLP layers each “read” their input from the residual stream (by performing a linear projection), and then “write” their result to the residual stream by adding a linear projection back in. Each attention layer consists of multiple heads, which operate in parallel. from https://transformer-circuits.pub/2021/framework/index.html Think about it!)</mark>

The *classical* Transformer follows an encoder-decoder architecture. A sequence of inputs e. g., a sentence in the source language, is first mapped to a sequence of embeddings and enriched with positional information. The encoder receives the input and creates a rich representation from it by encoding the context in which the input appears. The output of the encoder is then fed to the decoder. The decoder takes the embedded target sequence along with parts of the encoded representation of the output to generate the output sequence e. g., a sentence in the target language in an auto-regressive fashion. <mark style="background: #FFB8EBA6;">(citation -> Vaswani )</mark> The architecture is depicted in Figure [[🤖Transformer#^2cf7ee]].

The encoder consists of $L$ stacked Transformer blocks. In the classical implementation $L$ is set to $6$ ([[@vaswaniAttentionAllYou2017]]; p. 6) . Each block consists of two sub-layers: a multi-head self-attention layer, followed by a position-wise, feed-forward network. In the encoder inputs can attend to any token within the sequence. Each of these sub-layer are connected by skip connections ([[@heDeepResidualLearning2015]]), whereby the input of the sub-layer is added to the sub-layer's output. Finally layer normalization ([[@baLayerNormalization2016]]) is applied. <mark style="background: #FFF3A3A6;">(Shortly, address why do we stack many layers at all? What is learned in lower layers? https://arxiv.org/abs/1311.2901)</mark>

Aside from the multi-headed self-attention and feed-forward sub-layer the decoder features a third sub-layer for multi-headed self-attention on the output of the encoder, known as *cross attention*. The multi-headed self-attention mechanism in the decoder differs from the one in the encoder. Specifically, future parts of the output sequence are causally masked to prevent the model from attending to subsequent positions during training. ([[@vaswaniAttentionAllYou2017]], p. 3) ([[@narangTransformerModificationsTransfer2021]], p. 15).

The output of the decoder is finally passed through a linear layer with a softmax activation function to unembed the output and retrieve the probabilities for the next token ([[@vaswaniAttentionAllYou2017]]) (p. 5). Since the output sequence is generated token by token, with the most probable token being fed back as input to the decoder to provide context for the following token until the remaining sequence is generated.

This chapter can only provides a high level of the Transformer. Subsequent chapters cover multi-headed self-attention, token and positional embeddings in detail. 

## Transformer modes
For it's original application, machine translation, both the encoder and decoder are used, where the input sequence in the source language is first mapped to a rich, intermediate representation and subsequently the output sequence is generated. Yet, the modular design, allows to adapt Transformers to a much wider range of use cases, some of which only require the encoder or decoder. [[@raffelExploringLimitsTransfer2020]] (p. 16 f.) differentiate these modes: 
1. **encoder-only:** use of only the encoder stage e. g. in sentiment classification. Since inputs are not masked, hence *fully-visible masking*, inputs can attend to any other input within the sequence. <mark style="background: #ABF7F7A6;">(What about padding masking in the encoder? https://stats.stackexchange.com/a/446571/351242)</mark>
2. **decoder-only:** Use of the decoder stage e. g., in auto-completion to auto-regressively generate a sequence. Use of *causal masking*, so that inputs only depend on all previously generated outputs.
3. **encoder-decoder:** Use of both the encoder and decoder stages e. g., in translation. Combines *Fully-visible masking* in the encoder, with *causal masking* in the decoder.
![[different-mask-patterns.png]]

As our focus is on probabilistic classification of tabular data, the goal is to learn an enriched representation of the features, which can be used for classifying the label. As such, only *encoder-only* Transformers suffice. Also, *fully-visible masking* is appropriate, as inputs are free to attend to all previous and subsequent columns or rows within the dataset.

In the subsequent sections we introduce the classical Transformer of [[@vaswaniAttentionAllYou2017]] more thoroughly. Our focus on the central building blocks, attention, multi-head self-attention, and cross-attention (see Chapter [[🅰️Attention]]) as well as feed-forward networks (chapter [[🎱Position-wise FFN]]). In the subsequent chapters we show, that the self-attention mechanism and embeddings are generic enough to be transferred to the tabular domain. With the [[🤖TabTransformer]] ([[@huangTabTransformerTabularData2020]], p. 1 f.) and [[🤖FTTransformer]] ([[@gorishniyRevisitingDeepLearning2021]] p. 1) we introduce two promising alternatives. For consistency we adhere to a notation suggested in [[@phuongFormalAlgorithmsTransformers2022]] (p. 1 f) throughout the work.

## $\texttt{add\_layer}()$

Before studying two concrete transformer models, we revisit *residual connections* and *layer norm* in the Transformer block and discuss alternatives for their arrangement. What seems like a pedantic detail, is vital for the training process and convergence of the network.

### residual connections
Recall from our introduction on Transformers (see Chapter [[🤖Transformer]]), that both the encoder and decoder stack multiple Transformer blocks, each of which consists off several sub-layers, resulting in a deep networks. As neural networks are commonly trained using back-propagation, which relies on the gradient for the error (with respect to the parameters) to be propagated through the network starting at the last layer, vanishing or exploding gradient pose a major difficulty in training deep neural networks (see e. g. [[@heDeepResidualLearning2015]]).

Thus, stacking multiple layers in the encoder or decoder prevents the gradient information to flow less efficiently through the network, and may affect the overall training performance ([[@wangLearningDeepTransformer2019]] ;  p. 1,811).  As a remedy, [[@vaswaniAttentionAllYou2017]] add residual connections around each sub-layer:

<mark style="background: #FFF3A3A6;">(Formula, similar to Wang et al)
for a solution. Let $\mathcal{F}$ be a sub-layer in encoder or decoder, and $\theta_l$ be the parameters of the sub-layer. A residual unit is defined to be (He et al., 2016b):
$$
\begin{aligned}
x_{l+1} & =f\left(y_l\right) \\
y_l & =x_l+\mathcal{F}\left(x_l ; \theta_l\right)
\end{aligned}
$$
where $x_l$ and $x_{l+1}$ are the input and output of the $l$-th sub-layer, and $y_l$ is the intermediate output followed by the post-processing function $f(\cdot)$. In this way, $x_l$ is explicitly exposed to $y_l$ (see Eq. (2)).</mark>

![[residual-connection.png]]
(from [[@heDeepResidualLearning2015]])

Intuitively, the residual connection provides an alternative path for information to flow through the network, since some information can bypass the sub-layer and is added to its output. Also, exploding or vanishing gradients are mitigated, as gradients can bypass the sub-layer, ultimately resulting in an easier optimization ([[@liuRethinkingSkipConnection2020]]).  Residual connections also help to preserve the positional embeddings ([[🧵Positional encoding]]) as, the layer's input are maintained in the identity mapping.<mark style="background: #FFB8EBA6;"> (may come back and read this https://transformer-circuits.pub/2021/framework/index.html)</mark>

## research on residual connections 🧬

<mark style="background: #CACFD9A6;">For a residual block $x+f(x)$, its shortcut output refers to $x$, its residual branch output refers to $f(x)$, and the dependency on its residual branch refers to $\frac{\operatorname{Var}[f(x)]}{\operatorname{Var}[x+f(x)]}$.(From [[@liuUnderstandingDifficultyTraining2020]])
</mark>


<mark style="background: #ADCCFFA6;">“As in Figure 7, at initialization, a Pre-LN layer has roughly the same dependency on its residual branch and any previous layer, whereas a Post-LN layer has a stronger dependency on its residual branch (more discussions are elaborated in Section 4.1). We find that strong dependencies of Post-LN amplify fluctuations brought by parameter changes and destabilize the training (as in Theorem 2 and Figure 4). Besides, the loose reliance on residual branches in Pre-LN generally limits the algorithm’s potential and often produces inferior models.” (Liu et al., 2020, p. 2)</mark>


<mark style="background: #FFF3A3A6;">“To summarize, our model is roughly equivalent to the original Transformer proposed by Vaswani et al. (2017) with the exception of removing the Layer Norm bias, placing the layer normalization outside the residual path, and using a different position embedding scheme.” (Raffel et al., 2020, p. 5)</mark>

<mark style="background: #FFB86CA6;">“Residual connections (He et al., 2016a) were first introduced to facilitate the training of deep convolutional networks, where the output of the `-th layer F` is summed with its input: x`+1 = x` + F`(x`). (1) The identity term x` is crucial to greatly extending the depth of such networks (He et al., 2016b). If one were to scale x` by a scalar λ`, then the contribution of x` to the final layer FL is (∏L−1 i=` λi)x`. For deep networks with dozens or even hundreds of layers L, the term ∏L−1 i=` λi becomes very large if λi > 1 or very small if  for enough i. When backpropagating from the last layer L back to `, these multiplicative terms can cause exploding or vanishing gradients, respectively. Therefore they fix λi = 1, keeping the total residual path an identity map.” (Nguyen and Salazar, 2019, p. 2) [[@nguyenTransformersTearsImproving2019]]</mark>

<mark style="background: #ADCCFFA6;">“Inspired by He et al. (2016b), we apply LAYERNORM immediately before each sublayer (PRENORM): x`+1 = x` + F`(LAYERNORM(x`)). (3) This is cited as a stabilizer for Transformer training (Chen et al., 2018; Wang et al., 2019) and is already implemented in popular toolkits (Vaswani et al., 2018; Ott et al., 2019; Hieber et al., 2018), though not necessarily used by their default recipes. Wang et al. (2019) make a similar argument to motivate the success of PRENORM in training very deep Transformers. Note that one must append an additional normalization after both encoder and decoder so their outputs are appropriately scaled.” (Nguyen and Salazar, 2019, p. 2)</mark> [[@nguyenTransformersTearsImproving2019]]

<mark style="background: #FFB86CA6;">Skip Connection bypasses the gradient exploding or vanishing problem and tries to solve the model optimization problem from the perspective of information transfer. It enables the delivery and integration of information by adding an identity mapping from the input of the neural network to the output, which may ease the optimization and allow the error signal to pass through the non-linearities.</mark> [[@liuRethinkingSkipConnection2020]]

<mark style="background: #D2B3FFA6;">“We conjecture this has caused past convergence failures (Popel and Bojar, 2018; Shazeer and Stern, 2018), with LAYERNORMs in the residual path acting similarly to λi 6= 1; furthermore, warmup was needed to let LAYERNORM safely adjust scale during early parts of training.” (Nguyen and Salazar, 2019, p. 2)</mark>

## Virtual Weights and the Residual Stream as a Communication Channel

<mark style="background: #BBFABBA6;">One of the main features of the high level architecture of a transformer is that each layer adds its results into what we call the “residual stream.” 2 The residual stream is simply the sum of the output of all the previous layers and the original embedding. We generally think of the residual stream as a communication channel, since it doesn't do any processing itself and all layers communicate through it.</mark>

![](data:image/PNG;base64,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%0AAv8HIEyZRrOwQKgAAAAASUVORK5CYII=)

<mark style="background: #FFB8EBA6;">The residual stream has a deeply linear structure. 3 Every layer performs an arbitrary linear transformation to "read in" information from the residual stream at the start, 4 and performs another arbitrary linear transformation before adding to "write" its output back into the residual stream. This linear, additive structure of the residual stream has a lot of important implications. One basic consequence is that the residual stream doesn't have a ["privileged basis"](https://transformer-circuits.pub/2021/framework/index.html#def-privileged-basis); we could rotate it by rotating all the matrices interacting with it, without changing model behavior. ([[@elhage2021mathematical]])
</mark>
“<mark style="background: #FFF3A3A6;">For Transformer, it is not easy to train stacked layers on neither the encoder-side nor the decoderside. Stacking all these sub-layers prevents the efficient information flow through the network, and probably leads to the failure of training. Residual connections and layer normalization are adopted for a solution. Let F be a sub-layer in encoder or decoder, and θl be the parameters of the sub-layer.</mark>  ([[@wangLearningDeepTransformer2019]])

A residual unit is defined to be (He et al., 2016b): xl+1 = f (yl) (1) yl = xl + F (xl; θl) (2) where xl and xl+1 are the input and output of the l-th sub-layer, and yl is the intermediate output followed by the post-processing function f (·). In this way, xl is explicitly exposed to yl (see Eq. (2)). Moreover, layer normalization is adopted to reduce the variance of sub-layer output because hidden state dynamics occasionally causes a much longer training time for convergence. There are two ways to incorporate layer normalization into the residual network. • Post-Norm. In early versions of Transformer (Vaswani et al., 2017), layer normalization is placed after the element-wise residual addition (see Figure 1(a)), like this: xl+1 = LN(xl + F (xl; θl)) (3) where LN(·) is the layer normalization function, whose parameter is dropped for simplicity. It can be seen as a post-processing step of the output (i.e., f (x) = LN(x)). • Pre-Norm. In recent implementations (Klein et al., 2017; Vaswani et al., 2018; Domhan, 2018), layer normalization is applied to the input of every sub-layer (see Figure 1(b)): xl+1 = xl + F (LN(xl); θl) (4” (Wang et al., 2019, p. 1811)

“2.2 On the Importance of Pre-Norm for Deep Residual Network The situation is quite different when we switch to deeper models. More specifically, we find that prenorm is more efficient for training than post-norm if the model goes deeper. This can be explained by seeing back-propagation which <mark style="background: #BBFABBA6;">is the core process to obtain gradients for parameter update. Here we take a stack of L sub-layers as an example. Let E be the loss used to measure how many errors occur in system prediction, and xL be the output of the topmost sub-layer. For post-norm Transformer, given a sub-layer l, the differential of E with respect to xl can be computed by the chain rule, and we have ∂E ∂xl = ∂E ∂xL × L−1 ∏ k=l ∂LN(yk) ∂yk × L−1 ∏ k=l ( 1 + ∂F (xk; θk) ∂xk ) (5) where ∏L−1 k=l ∂ LN(yk ) ∂yk means the backward pass of the layer normalization, and ∏L−1 k=l (1 + ∂F(xk;θk) ∂xk ) means the backward pass of the sub-layer with the residual connection. Likewise, we have the gradient for pre-norm 4: ∂E ∂xl = ∂E ∂xL × ( 1+ L−1 ∑ k=l ∂F (LN(xk); θk) ∂xl ) (6) Obviously, Eq. (6) establishes a direct way to pass error gradient ∂E ∂xL from top to bottom. Its merit lies in that the number of product items on the right side does not depend on the depth of the stack. In contrast, Eq. (5) is inefficient for passing gradients back because the residual connection is not 3We need to add an additional function of layer normalization to the top layer to prevent the excessively increased value caused by the sum of unnormalized output. 4For a detailed derivation, we refer the reader to Appendix A. a bypass of the layer normalization unit (see Figure 1(a)). Instead, gradients have to be passed through LN(·) of each sub-layer. It in turn introduces term ∏L−1 k=l ∂ LN(yk ) ∂yk into the right hand side of Eq. (5), and poses a higher risk of gradient vanishing or exploring if L goes larger. This was confirmed by our experiments in which we successfully trained a pre-norm Transformer system with a 20-layer encoder on the WMT English-German task, whereas the post-norm Transformer system failed to train for a deeper encoder (Section 5.1).” (Wang et al., 2019, p. 1812)</mark>


### layer norm
(to be done)

See also: https://arxiv.org/pdf/1901.09321.pdf
See also: https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/

Example for brittle training of Transformers with Adam [[@shazeerAdafactorAdaptiveLearning2018]]


By the placement of the layer norm, one differentiates the 


1. Why do we use layer norm and skip connections at all?
2. Are they vital for the overall architecture?
3. How do the two most commonly used variants look like?
4. What are the advantages and disadvantages of each approach?
5. What complicates training?
6. What is required for *post-norm*?
7. Give the formula
8. Visualize!


“Residual connection and layer normalization Besides the two sub-layers described above, the residual connection and layer normalization are also key components to the Transformer. For any vector v, the layer normalization is computed as LayerNorm(v) = γ v−μ σ + β, in which μ, σ are the mean and standard deviation of the elements in v, i.e., μ = 1 d ∑d k=1 vk and σ2 = 1 d ∑d k=1(vk − μ)2. Scale γ and bias vector β are parameters” (Xiong et al., 2020, p. 3)

“the impact of the layer normalization positions [32, 33]. There are currently two major layer normalization positions in Transformers: Pre-Layer Normalization (Pre-LN) and Post-Layer Normalization (Post-LN). Pre-LN applies the layer normalization to an input for each sub-layer, and Post-LN places the layer normalization after each residual connection. The original Transformer [28] employs PostLN. However, recent studies often suggest using Pre-LN [32, 2, 5] because the training in Post-LN with deep Transformers (e.g., ten or more layers) often becomes unstable, resulting in useless models. Figure 1 shows an actual example; loss curves of training 18L-18L Transformer encoder-decoders on a widely used WMT English-to-German machine translation dataset. Here, XL-Y L represents the number of layers in encoder and decoder, where X and Y correspond to encoder and decoder, respectively. These figures clearly show that 18L-18L Post-LN Transformer encoder-decoder fails to train the model. However, in contrast, Liu et al. [13] reported that Post-LN consistently achieved better performance than Pre-LN in the machine translation task when they used 6L-6L (relatively shallow) Transformers.” (Takase et al., 2022, p. 2)

![[layer-norm-first-last.png]]
Visualization of norm-first and norm last (similar in [[@xiongLayerNormalizationTransformer2020]])





Due to the order of the layers the setup is known as the *post layer norm* (Post LN) architecture.


<mark style="background: #BBFABBA6;">“We use a simplified version of layer normalization where the activations are only rescaled and no additive bias is applied. After layer normalization, a residual skip connection (He et al., 2016) adds each subcomponent’s input to its output.” (Raffel et al., 2020, p. 4)</mark>

<mark style="background: #FFF3A3A6;">“Our analysis starts from the observation: the original Transformer (referred to as Post-LN) is less robust than its Pre-LN variant2 (Baevski and Auli, 2019; Xiong et al., 2019; Nguyen and Salazar, 2019).” (Liu et al., 2020, p. 1)</mark>

- motivation to switch discuss the effects of layer pre-normalization vs. post-normalization (see [[@tunstallNaturalLanguageProcessing2022]])

Both the encoder and decoder stack $L$ Transformer blocks. The specific layer arrangement is referred to as *Post Layer Normalization* (Post-LN) derived from the placement of the normalization layer.

First, we apply layer normalization before the selfattention and feedforward blocks instead of after. This small change has been unanimously adopted by all current Transformer implementations because it leads to more effective training (Baevski and Auli, 2019; Xiong et al., 2020). [[@narangTransformerModificationsTransfer2021]]

Layer normalization improves the trainability of the Transformer by keeping.

- Update residual stream, refine inputs from previous layers? See [[@elhage2021mathematical]]

(brittle training, requirement for warm-up stages)

“Different orders of the sub-layers, residual connection and layer normalization in a Transformer layer lead to variants of Transformer architectures. One of the original and most popularly used architecture for the Transformer and BERT (Vaswani et al., 2017; Devlin et al., 2018) follows “selfattention (FFN) sub-layer → residual connection → layer normalization”, which we call the Transformer with PostLayer normalization (Post-LN Transformer), as illustrated in Figure 1.” (Xiong et al., 2020, p. 3)

*Pre-LN* is known to be particullary hard
- Our analysis starts from the observation: the original Transformer (referred to as Post-LN) is less robust than its Pre-LN variant2 (Baevski and Auli, 2019; Xiong et al., 2019; Nguyen and Salazar, 2019). (from [[@liuUnderstandingDifficultyTraining2020]])
Addnorm operation.

How it's done in [[@tayEfficientTransformersSurvey2022]]:
The inputs and output of the multi-headed self-attention module are connected by residual connectors and a layer normalization layer. The output of the multi-headed selfattention module is then passed to a two-layered feed-forward network which has its inputs/outputs similarly connected in a residual fashion with layer normalization. The sublayer residual connectors with layer norm is expressed as:
$$
X=\operatorname{LayerNorm}\left(F_S(X)\right)+X
$$
where $F_S$ is the sub-layer module which is either the multi-headed self-attention or the position-wise feed-forward layers.



“Both the multi-head self-attention and the feed-forward layer are followed by an add-norm operation. This transformation is simply a residual connection [17] followed by layer normalization [23]. The layer normalization computes the average and standard deviation of the output activations of a given sublayer and normalizes them accordingly. This guarantees that the input yt of the following sublayer is well conditioned, i.e., that yT t 1 = 0 and yT t yt = √d.” (Sukhbaatar et al., 2019, p. 3)

The later, is commonly known as pre-norm.

![[formulas-layer-norm.png]]

[[@xiongLayerNormalizationTransformer2020]]
[[@nguyenTransformersTearsImproving2019]]
[[@wangLearningDeepTransformer2019]]

https://stats.stackexchange.com/a/565203/351242 ResNet paper ([[@heDeepResidualLearning2015]]) on residual learning / residual connections. Discusses in general the problems that arise with learning deep neural networks.

2.3 Putting it all together (from [[@tayEfficientTransformersSurvey2022]])
Each Transformer block can be expressed as:
$$
\begin{aligned}
& \left.X_A=\text { LayerNorm(MultiheadAttention }(X, X)\right)+X \\
& X_B=\operatorname{LayerNorm}\left(\operatorname{PositionFFN}\left(X_A\right)\right)+X_A
\end{aligned}
$$
where $X$ is the input of the Transformer block and $X_B$ is the output of the Transformer block. Note that the MultiheadAttention() function accepts two argument tensors, one for query and the other for key-values. If the first argument and second argument is the same input tensor, this is the MultiheadSelfAttention mechanism.


The classical Transformer of [[@vaswaniAttentionAllYou2017]] features 

- layer norm is the same as batch norm except that it normalizes the feature dimension ([[@zhangDiveDeepLearning2021]] p. 423)

As mentioned earlier, the Transformer architecture makes use of layer normalization and skip connections. The former normalizes each input in the batch to have zero mean and unity variance. Skip connections pass a tensor to the next layer of the model without processing and add it to the processed tensor. When it comes to placing the layer normalization in the encoder or decoder layers of a transformer, there are two main choices adopted in the literature: Post layer normalization This is the arrangement used in the Transformer paper; it places layer normalization in between the skip connections. This arrangement is tricky to train from scratch as the gradients can diverge. For this reason, you will often see a concept known as learning rate warm-up, where the learning rate is gradually increased from a small value to some maximum value during training. Pre layer normalization This is the most common arrangement found in the literature; it places layer normalization within the span of the skip connections. This tends to be much more stable during training, and it does not usually require any learning rate warm-up. The difference between the two arrangements is illustrated in Figure 3-6. (unknown)

“To train a Transformer however, one usually needs a carefully designed learning rate warm-up stage, which is shown to be crucial to the final performance but will slow down the optimization and bring more hyperparameter tunings. In this paper, we first study theoretically why the learning rate warm-up stage is essential and show that the location of layer normalization matters. Specifically, we prove with mean field theory that at initialization, for the original-designed Post-LN Transformer, which places the layer normalization between the residual blocks, the expected gradients of the parameters near the output layer are large. Therefore, using a large learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful for avoiding this problem. On the other hand, our theory also shows that if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization. This motivates us to remove the warm-up stage for the training of Pre-LN Transformers. We show in our experiments that Pre-LN Transformers without the warm-up stage can reach comparable results with baselines while requiring significantly less training time and hyper-parameter tuning on a wide range of applications.” (Xiong et al., 2020, p. 1)

Our analysis starts from the observation: the original Transformer (referred to as Post-LN) is less robust than its Pre-LN variant2 (Baevski and Auli, 2019; Xiong et al., 2019; Nguyen and Salazar, 2019). We recognize that gradient vanishing issue is not the direct reason causing such difference, since fixing this issue alone cannot stabilize PostLN training. It implies that, besides unbalanced gradients, there exist other factors influencing model training greatly [[@liuUnderstandingDifficultyTraining2020]]

leading to a brittle optimization. 
A variant known as pre-norm, 

Why do we employ residual connections? add input back in. Requires the 

Besides the decoder also contains a third sub-layer.

<mark style="background: #FFF3A3A6;">The residual connection is crucial in the Transformer architecture for two reasons:

1.  Similar to ResNets, Transformers are designed to be very deep. Some models contain more than 24 blocks in the encoder. Hence, the residual connections are crucial for enabling a smooth gradient flow through the model.
    
2.  Without the residual connection, the information about the original sequence is lost. Remember that the Multi-Head Attention layer ignores the position of elements in a sequence, and can only learn it based on the input features. Removing the residual connections would mean that this information is lost after the first attention layer (after initialization), and with a randomly initialized query and key vector, the output vectors for position � has no relation to its original input. All outputs of the attention are likely to represent similar/same information, and there is no chance for the model to distinguish which information came from which input element. An alternative option to residual connection would be to fix at least one head to focus on its original input, but this is very inefficient and does not have the benefit of the improved gradient flow.</mark> (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)

## Resources
- Detailed explanation and implementation. Check my understanding against it: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- mathematical foundations [[@elhage2021mathematical]]
- general description of setup in [[@zhangDiveDeepLearning2021]]
- https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3
- http://nlp.seas.harvard.edu/2018/04/03/attention.html

Components in Embedding:
[[🛌Token Embedding]]
[[🧵Positional encoding]]

Components in Transformer block:
[[🅰️Attention]]
[[🎱Position-wise FFN]]

Specialized variants for tabular data:
[[🤖TabTransformer]]
[[🤖FTTransformer]]


## Visualizations



![[norm-first-norm-last-big-picture.png]]
(from https://github.com/dvgodoy/PyTorchStepByStep)

Layer norm / batch norm / instance norm:
![[layer-batch-instance-norm.png]]
![[viz-of image-embedding.png]]
(from https://github.com/dvgodoy/PyTorchStepByStep)

## Notes from Sukhabaatar
(see [[@sukhbaatarAugmentingSelfattentionPersistent2019]])
Feedforward sublayer. The second element of a transformer layer is a fully connected feedforward layer. This sublayer is applied to each position $t$ in the input sequence independently, and consists of two affine transformations with a pointwise non-linear function in between:
$$
\mathrm{FF}\left(\mathbf{x}_t\right)=\mathbf{U} \sigma\left(\mathbf{V} \mathbf{x}_t+\mathbf{b}\right)+\mathbf{c},
$$
where $\sigma(x)=\max (0, x)$ is the ReLU activation function; $\mathbf{V}$ and $\mathbf{U}$ are matrices of dimension $d \times d_f$ and $d_f \times d$ respectively; $\mathbf{b}$ and $\mathbf{c}$ are the bias terms. Typically, $d_f$ is set to be 4 times larger than $d$.
Add-norm. Both the multi-head self-attention and the feed-forward layer are followed by an add-norm operation. This transformation is simply a residual connection [17] followed by layer normalization [23]. The layer normalization computes the average and standard deviation of the output activations of a given sublayer and normalizes them accordingly. This guarantees that the input $\mathbf{y}_t$ of the following sublayer is well conditioned, i.e., that $\mathbf{y}_t^T 1=0$ and $\mathbf{y}_t^T \mathbf{y}_t=\sqrt{d}$. More precisely, the AddNorm operation is defined as:
$$
\operatorname{AddNorm}\left(\mathbf{x}_t\right)=\operatorname{LayerNorm}\left(\mathbf{x}_t+\operatorname{Sublayer}\left(\mathbf{x}_t\right)\right) \text {, }
$$
where Sublayer is either a multi-head self-attention or a feedforward sublayer.


## Notes from Tay
(see [[@tayEfficientTransformersSurvey2022]])
- transformers are a multi-layered architecture formed by stacking transformer blocks on top of one another.
- Transformer blocks are characterized by a multi-head sel-attention mechanism, a poistion-wise feed-forward network, layer norm modules ([[@baLayerNormalization2016]]) and residual connectors ([[@heDeepResidualLearning2015]])
- The input is passed through an embedding layer and converts one-hot tokens into a $d_{\text{model}}$ dimensional embedding. The tensor is composed with a positional encoding and passed through a multi-headed self-attention module. 
- Inputs and outputs oft he multi-headed self-attention module are connected by residual connectors and a layer normalization layer. the output of the multi-headed self-attention module is then passed to a two-layered feed forward network which has it inputs / outputs similarily connected in a residual fashion with layer normalization. 

## Notes from Harvard🎓
http://nlp.seas.harvard.edu/2018/04/03/attention.html
- self-attention is a an attention mechaism of realting different positions of a single sequence to compute a representation of the sequence.
- Encoder maps an input sequenc of symbol representations (x1, .... xn) to a a seuqence of continuous representations z=(z1, ..., zn). The decoder generates from z an output sequence(y1,...ym) of symbols one element at a time. Each step is auto-regressive as generated symbols are used as additional input when generating text.
- Dropout is applied to the output of each sublayer, before it is added to the sub-layer input and normalized. Requires all sublayers of the model to have a fixed dimensions to make residual connnections (which add elemnt-wisely?) possible.
- Each layer has two sub-layers. The first is a multi-headed self attention mechanism and the second is a simple, point-wise feed-forward network.
- the decoder inserts a third layer, which performs multi-head attention over the output of the encoder stack. Again layer normalization and residual connections are used.
- To prevent the decoder from attending to subsequent  poistions a musk is used, so that the predictions only depend on known outputs at positions less than i.

## Notes from Tunstall
[[@tunstallNaturalLanguageProcessing2022]]
- it's based on the encoder-decoder architecture, which is commonly used in machine translation, where a sequence is translated from language to another.
- **Encoder:** converts an input sequence of tokens into sequence of embedding vectors which is the context.
-  **Decoder:** use encoder's hidden state to iteratively generate an output sequence of tokens (auto-regressively?)
- Both encoder and decoder consists of multiple stacked transformer blocks
- the encoder's output is fed to the decoder layer and the decoder gnerates a prediction for the most probable next token in the sequence. The output is then fed back into the decoder to generate the next token until the end of EOS token is reached.
- Enoder-only architecture that is well suited for text-classification and named-entity recognition. A sequence of text is converted into a rich numerical representation => bidirectional attention
- Other types => decoder-only => autoregressive attention and encoder-decoder => good for machine translation and summarization
- The encoder feeds the input through sublayers:
	- multi-head self-attention layer
	- fully-connected feed-forward layer
- The purpose of the encoder is to update the input embeddings and to produce representations that encode some contextual information of the sequence. Input dimensions and output dimensions are the same. 
- skip connections and layer normalization is used to train deep neural networks efficiently.

## Notes from ML6
(see: https://blog.ml6.eu/transformers-for-tabular-data-hot-or-not-e3000df3ed46)
- Why it makes sense to use embeddings / transformers for tabular data:
	- In a lot of tabular “languages”, there are meaningful feature interactions. The value of one feature impacts the way another feature should be interpreted.Decision trees naturally lend themselves to model these kinds of interactions because of their sequential decision making process. A decision deeper in the tree depends on all previous decisions since the root, so previous feature values impact the current feature interpretation. That’s why a transformer also explicitly models token interactions through its multi head self-attention mechanism. In that way, the model produces _contextual embeddings_.
	- use powerful semi-supervised training techniques from natural language processing
	- A final advantage of transformers is that they excel in handling missing and noisy features.

## Notes from Rothman
[[@rothmanTransformersNaturalLanguage2021]]
- multi-head attention sub-layer contains eight attention heads and is followed by post-layer normalization, which will add residual connections to the output of the sublayer and normalize
- performing attention using a single head is slow. Given the size of the embeddings, we would have to make huge computations. A better way is to divide the dimensions (embedding dim) among the heads.  Output of the multi-headed attention module must be concatenated. 
- Inside each head of the attention mechanism, each word vector has three representations a Query vector, key and value.

## Notes from e2ml
(https://e2eml.school/transformers.html#second_order_matrix_mult)
- De-embedding is done the same way embeddings are done, with a projection from one space to another, that is, a matrix multiplication.
- The softmax is helpful here for three reasons. First, it converts our de-embedding results vector from an arbitrary set of values to a probability distribution. Softmax exaggerates the difference between values. Preserves though if some words are equally likely. And we can perform backpropagation.
- Linear layers are used to project the matricces. To make multi-headed self-attention even possible.
- Skip connections add a copy of the input to the output of a set of calculations. The input to the attention block are added back in to its outputs. The inputs to the element-wise forward blcoks are added to the inputs. Makes the overall pipeline robust.
- Skip connections help to smooth out saddle points and ridges of the gradient. The problem is taht attention is a filter, that blocks out most what tries to pass through and may lead to large areas where the gradient is flat. Slopes of loss function hills are much smoother uniform, if skip connections are used. Also, it could happen, that an attention filter forgets entirely about the most recent word. Skip connections therefore enforce the signal and add the word back in.
- Inputs are shifted to have a zero mean and scaled to std dev of 1. It's needed cause it matters, for e. g., softmax what ranges values have and if they are balanced.
- layer normalization maintains a consistent distribution of signal values each step of the way throughout many-layered neural nets.
- Intuively, multiple attention layers allow for multiple paths to good set of transformer params. More layers lead to better results, but improvements are marginal with more than six layers.
- We can not make any judgements about the performance of the encoder, as the result is only a sequence of vectors in the embedded space.
- Cross-attention is similar to self-attention but with the exception taht the key matrix, value  matrix are from the fianl encoder layer
- As all layers get the same embedded source sequence in the decoder, it can be said that succesive layers provide redundancy and cooperate to perform the same task.

## Notes from Huggingface 🤗
https://huggingface.co/course/chapter1/4
-   **Encoder (left)**: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
-   **Decoder (right)**: The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

## Notes from Baeldung 🍁
https://www.baeldung.com/cs/transformer-text-embeddings
- input sequence is input to the encoding block to obtain rich embeddings for each token, which is then fed to the decoding block to obtain the output
- initial layers in the encoder capture more basic patterns, latter blocks capture more sophisticated ones (as only filtered signal is passed?)
- Encoder takes one vector per token in the sequence as in put and returns a vector per token of the same shape. Thus, intuitively, the encoder returns the same input vectors, but enriched with more complex information.
- Self-attention layer detects related tokens in the sequence. (sequential)
- Next are the add and normalization layers, as the entire sequence is required and normalized. FFN follows (parallel) and another add and normalization. The only part that can be normalized are the feed-forward parts.
- decoder is like the encoder but with an additional encoder-decoder-attention layer. The attention mechanism provides insights into which tokens of the input sequence are more relevant to the current output token. It is followed by an add and normalize layer.
- the current decoder input will be processed producing and output, which will feed the next decoder
- The last decoder is connected to the output layer, generating the next output token, until the EOS token is reached.
- The decoder outputs a stack of float vectors. This vector is connect to a linear layer to project the output vector into a vector the size of the vocabulary. By applying softmax, we obtain the probabilities for every token to be the next token in the sequence. The token with the highest probability is chosen.

## Notes on Talk with Łukasz Kaiser 🎙️
(see here: https://www.youtube.com/watch?v=rBCqOTEfxvg)

- RNNs suffer from vanishing gradients
- Some people used CNNs, but path length is still logarithmic (going down a tree). Is limited to position.
- Attention: make a query with your vector and look at similar things in the past. Looks at everything, but choose things, that are similar.
- Encoder attention allows to go from one word to another. (Encoder Self-Attention)
- MaskedDecoder Self-Attention (is a single matrix multiply with a mask) to mask out all prev. elements not relevant
- Attention A(Q, K, V) (q = query vector) (K, V matrices= memory) (K = current word working on, V = all words generated before). You want to use q to find most similar k and get values that correspond to keys. (QK^T) gives a probability distribution over keys, which is then multiplied with values
- n^2 * d complexity
- to preserve the order of words they use multi-head attention
- attention heads can be interpreted (see winograd schemas)

## Notes from talk with Lucas Beyer / Google🎙️

^54aa8a

(see https://www.youtube.com/watch?v=EixI6t5oif0)
- attention was originally introduced in the Bahdu paper. But was not the most central part.
- attention is like a (convoluted (soft)) dictionary lookup. like in a dict we have keys and values and want to query the dictionary. keys and values are a vector of quotes. measure the similarity with the dotproduct. we measure similarity between query and key (attention weights) and the result is normalized. We take weighted average of all values weighted by the attention weights. Note output can also be average over multiple 
![[attention-visualization.png]]
- q (word to translate), k, v (words in source language)
- We not just have one query, but multiple. Also an attention matrix. We use multi-head attention
- Multi-head attention splits the queries along the embedding dimension. Also outputs are split. Works empirically better. Requires less compute. (only implementation details. Not the gist of attention.)
- Architecture is heavily inspired by the translation task / community. This is helpful, as it resulted in encoder / decoder architecture.
- Every token from the input sequence is linearily projected. Each vector looks around to see what vectors are there and calculates the output. (self-attention)
- Every token individually is sent to a oint-wise MLP. It's done individually for every token. Stores knowledge. There is a paper. (references in [[@gevaTransformerFeedForwardLayers2021]] are the best I could find?) Gives the model processing power to think about what it has seen." Larger hidden size gives better results.
- skip processing. We have input and update it with our processing. (See residual stream in (mathematical foundations of Transformers))
- Layer norm is technically important.
- It's not clear, which variant of layer-norm is better.
- Decoder learns to sample from all possible outputs of the target language. 10 most likely translation etc. Computationally infeasible. To solve we look at one token at a time. Decoder works auto-regressively. Choose most likely token. and update all the inputs / things we have computed so far.
- All inputs are passed into the decoder at once to reduce training times. We multiply with mask matrix to lookup future tokens. In generation time we can not implement this trick and have to implement token by token.
- Cross-attention. Tokens from decoder become queries and keys and values come from the encoder. Look at the tokens from the source language (keys, values). 
- Flexible architecture. Needs loads of data. Are computationally efficient. That's true.

## Notes from Zhang
(see here: [[🧠Deep Learning Methods/@zhangDiveDeepLearning2021]])

- framework for designing attention mechanisms consists of:
    - volitional (~free) cues = queries
    - sensory inputs  = keys
    - nonvolitional cue of sensory input = keys
- attention pooling mechanism  enables a given query (volitional cue) to interact with keys (nonvolitional cues) which guides a biased selection over values (sensory inputs)
- self attention enjoys both parallel computation and the shortest maximum path length. Which makes it appealing to design deep architectures by using self-attention. Do not require a convolutional layer or recurrent layer.
- It's an instance of an encoder-decoder architecture. Input and output sequence embeddings are added with positional encoding before being fed into the encoder and the decoder that stack modules based on self-attention.
