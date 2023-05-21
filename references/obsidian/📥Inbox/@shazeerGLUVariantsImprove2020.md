*title:* GLU Variants Improve Transformer
*authors:* Noam Shazeer
*year:* 2020
*tags:* #activations #transformer #accuracy #tabtransformer #GLU #deep-learning 
*status:* #📦 
*related:*
- [[@vaswaniAttentionAllYou2017]] (architecture used)
- [[@glorotDeepSparseRectifier2011]] (activation function in comparsion)
*code:*
- None, but they base their implementation on another paper.
*review:*

## Notes 📍
- Authors propose and test different variants of the $\operatorname{GELU}$ activation in the classical transformer architecture ([[@vaswaniAttentionAllYou2017]]).
- Their new variants like $\operatorname{FFN}_{\text {GEGLU }}$ lead to better perplexities for the de-noising objective in pre-training and improvements in downstream learning tasks. Activation functions come at no additional complexity. $\operatorname{FFN}_{\text {GEGLU }}$ (based on gelu (gaussian error linear unit)) and $\operatorname{FFN}_{\text {SwiGLU }}$ (based on swish) among the best in terms of accuracy. 
-  The activation function is inherently used in the position-wise feed-forward network (FFN) of the transformer e. g., $\operatorname{FFN}_{\text {GEGLU }}\left(x, W, V, W_2\right)=(\operatorname{GELU}(x W) \otimes x V) W_2$. The FFN itself consists of of two linear layers with an activation function in between. Originally this was $\operatorname{ReLU}$. The FFN takes a vector $x$ (the hidden representation at a particular position in the sequence) and passes it through two learnt linear transformations, (represented by the matrices $W_1$ and $W_2$ and bias vectors $b_1$ and $b_2$ ).
- All gated Linear Units (GLU) consist of a component-wise product of two linear projections, of which one is first passed through an activation. With sigmoid activation this would be: $\mathrm{GLU}(x, W, V, b, c)=\sigma(x W+b) \otimes(x V+c)$.

## Annotations 📖
“Gated Linear Units consist of the component-wise product of two linear projections, one of which is first passed through a sigmoid function.” ([Shazeer, 2020, p. 1](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=1&annotation=2UQCS39Y))

“We test these variants in the feedforward sublayers of the Transformer  sequence-to-sequence model, and find that some of them yield quality improvements over the typically-used ReLU or GELU activations.” ([Shazeer, 2020, p. 1](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=1&annotation=Y6VPNJPP))

“The FFN takes a vector x (the hidden representation at a particular position in the sequence) and passes it through two learnt linear transformations, (represented by the matrices W1 and W2 and bias vectors b1 and b2).” ([Shazeer, 2020, p. 1](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=1&annotation=Q47E2F3E))

“[Dauphin et al., 2016] introduced Gated Linear Units (GLU), a neural network layer defined as the componentwise product of two linear transformations of the input, one of which is sigmoid-activated.” ([Shazeer, 2020, p. 1](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=1&annotation=X743SY7B))

“GEGLU(x, W, V, b, c) = GELU(xW + b) ⊗ (xV + c)” ([Shazeer, 2020, p. 2](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=2&annotation=3D8YVEAK))

“In this paper, we propose additional variations on the Transformer FFN layer which use GLU or one of its variants in place of the first linear transformation and the activation function.” ([Shazeer, 2020, p. 2](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=2&annotation=2ZVS3MJW))

“FFNGEGLU(x, W, V, W2) = (GELU(xW ) ⊗ xV )W2” ([Shazeer, 2020, p. 2](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=2&annotation=VDV8GTFB))

“All of these layers have three weight matrices, as opposed to two for the original FFN. To keep the number of parameters and the amount of computation constant, we reduce the number of hidden units dff (the second dimension of W and V and the first dimension of W2) by a factor of 2 3 when comparing these layers to the original two-matrix version” ([Shazeer, 2020, p. 2](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=2&annotation=USJHG6XY))

“In a transfer-learning setup, the new variants seem to produce better perplexities for the de-noising objective used in pre-training, as well as better results on many downstream language-understanding tasks. These architectures are simple to implement, and have no apparent computational drawbacks.” ([Shazeer, 2020, p. 3](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=3&annotation=8Q53SBBN))

“FFNGEGLU 73.96” ([Shazeer, 2020, p. 4](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=4&annotation=L9AKNG5J))

“FFNSwiGLU 74.56” ([Shazeer, 2020, p. 4](zotero://select/library/items/QJWAK9LR)) ([pdf](zotero://open-pdf/library/items/6ZX9BFUF?page=4&annotation=54HT2H72))