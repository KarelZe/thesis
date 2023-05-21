*title:* Efficient Transformers: A Survey
*authors:* Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler
*year:* 2022
*tags:* 
*status:* #📥
*related:*
*code:*
*review:*

## Notes 📍

### Architecture
- transformers are a multi-layered architecture formed by stacking transformer blocks on top of one another.
- Transformer blocks are characterised by a multi-head sel-attention mechanism, a poistion-wise feed-forward network, layer norm modules ([[@baLayerNormalization2016]]) and residual connectors ([[@heDeepResidualLearning2015]])
- The input is passed through an embedding layer and converts one-hot tokens into a $d_{\text{model}}$ dimensional embedding. The tensor is composed with a positional encoding and passed through a multi-headed self-attention module. 
- Inputs and outputs oft he multi-headed self-attention module are connected by residual connectors and a layer normalisation layer. the output of the multi-headed self-attention module is then passed to a two-layered feed forward network which has it inputs / outputs similarily connected in a residual fashion with layer normalisation. 
- Each Transformer block can be expressed as:
$$
\begin{aligned}
& \left.X_A=\text { LayerNorm(MultiheadAttention }(X, X)\right)+X \\
& X_B=\text { LayerNorm }\left(\text { PositionFFN }\left(X_A\right)\right)+X_A
\end{aligned}
$$
where $X$ is the input of the Transformer block and $X_B$ is the output of the Transformer block. Note that the MultiheadAttention() function accepts two argument tensors, one for query and the other for key-values. If the first argument and second argument is the same input tensor, this is the MultiheadSelfAttention mechanism.


## MHSA
The Transformer model leverages a multi-headed self-attention mechanism. The key idea behind the mechanism is for each element in the sequence to learn to gather from other tokens in the sequence. The operation for a single head is defined as:
$$
A_h=\operatorname{Softmax}\left(\alpha Q_h K_h^{\top}\right) V_h,
$$
where $X$ is a matrix in $\mathbb{R}^{N \times d}, \alpha$ is a scaling factor that is typically set to $\frac{1}{\sqrt{d}}, Q_h=$ $X \boldsymbol{W}_q, K_h=X \boldsymbol{W}_k$ and $V_h=X \boldsymbol{W}_v$ are linear transformations applied on the temporal dimension of the input sequence, $\boldsymbol{W}_q, \boldsymbol{W}_k, \boldsymbol{W}_v \in \mathbb{R}^{d \times \frac{d}{H}}$ are the weight matrices (parameters) for the query, key, and value projections that project the input $X$ to an output tensor of $d$ dimensions, and $N_H$ is the number of heads. Softmax is applied row-wise.

The outputs of heads $A_1 \cdots A_H$ are concatenated together and passed into a dense layer. The output $Y$ can thus be expressed as $Y=W_o\left[A_1 \cdots A_H\right]$, where $W_o$ is an output linear projection. Note that the computation of $A$ is typically done in a parallel fashion by considering tensors of $\mathbb{R}^B \times \mathbb{R}^N \times \mathbb{R}^H \times \mathbb{R}^{\frac{d}{H}}$ and computing the linear transforms for all heads in parallel.

The attention matrix $A=Q K^{\top}$ is chiefly responsible for learning alignment scores between tokens in the sequence. In this formulation, the dot product between each element/token in the query $(Q)$ and key $(K)$ is taken. This drives the self-alignment process in self-attention whereby tokens learn to gather from each other. (Tay; p. 4)

## Compute cost MHSA

The computation costs of Transformers is derived from multiple factors. Firstly, the memory and computational complexity required to compute the attention matrix is quadratic in the input sequence length, i.e., $N \times N$. In particular, the $Q K^{\top}$ matrix multiplication operation alone consumes $N^2$ time and memory. This restricts the overall utility of self-attentive models in applications which demand the processing of long sequences. Memory restrictions are tend to be applicable more to training (due to gradient updates) and are generally of lesser impact on inference (no gradient updates). The quadratic cost of self-attention impacts (Tay; p. 4 f.)


## Position-wise FFN

The outputs of the self-attention module are then passed into a two-layered feed-forward network with ReLU activations. This feed-forward layer operates on each **position independently**. This is expressed as follows:
$$
F_2\left(\operatorname{Re} L U\left(F_1\left(X_A\right)\right)\right)
$$
where $F_1$ and $F_2$ are feed-forward functions of the form $W x+b$.

See also [[🤖Transformer#^54aa8a]] he says that is the memory of the transformer, and that every token is processed separately. 

ReLU was proposed in [[@glorotDeepSparseRectifier2011]].

## Annotations 📖

“The new tensor is then additively composed with positional encodings and passed through a multiheaded self-attention module.” ([Tay et al., 2022, p. 3](zotero://select/library/items/SLWQVGHF)) ([pdf](zotero://open-pdf/library/items/PDDJFS9K?page=3&annotation=9KZA4TCJ))

“Positional encodings can take the form of a sinusoidal input (as per (Vaswani et al., 2017)) or be trainable embeddings” ([Tay et al., 2022, p. 3](zotero://select/library/items/SLWQVGHF)) ([pdf](zotero://open-pdf/library/items/PDDJFS9K?page=3&annotation=KIXCJ9U6))

“The inputs and output of the multi-headed self-attention module are connected by residual connectors and a layer normalisation layer. The output of the multi-headed selfattention module is then passed to a two-layered feed-forward network which has its inputs/outputs similarly connected in a residual fashion with layer normalisation. The sublayer residual connectors with layer norm is expressed as: X = LayerNorm(FS(X)) + X where FS is the sub-layer module which is either the multi-headed self-attention or the position-wise feed-forward layers.” ([Tay et al., 2022, p. 3](zotero://select/library/items/SLWQVGHF)) ([pdf](zotero://open-pdf/library/items/PDDJFS9K?page=3&annotation=LIKW8XJH))

“2.1 Multi-Head Self-Attention The Transformer model leverages a multi-headed self-attention mechanism. The key idea behind the mechanism is for each element in the sequence to learn to gather from other tokens in the sequence. The operation for a single head is defined as: Ah = Softmax(αQhK> h )Vh, where X is a matrix in RN×d, α is a scaling factor that is typically set to 1 √d , Qh = XWq, Kh = XWk and Vh = XWv are linear transformations applied on the temporal dimension of the input sequence, Wq, Wk, Wv ∈ Rd× d H are the weight matrices (parameters) for the query, key, and value projections that project the input X to an output tensor of d dimensions, and NH is the number of heads. Softmax is applied row-wise. The outputs of heads A1 · · · AH are concatenated together and passed into a dense layer. The output Y can thus be expressed as Y = Wo[A1 · · · AH ], where Wo is an output linear projection. Note that the computation of A is typically done in a parallel fashion by considering tensors of RB × RN × RH × R d H and computing the linear transforms for all heads in parallel. The attention matrix A = QK> is chiefly responsible for learning alignment scores between tokens in the sequence. In this formulation, the dot product between each element/token in the query (Q) and key (K) is taken. This drives the self-alignment process in self-attention whereby tokens learn to gather from each other. 2.2 Position-wise Feed-forward Layers The outputs of the self-attention module are then passed into a two-layered feed-forward network with ReLU activations. This feed-forward layer operates on each position independently. This is expressed as follows: F2(ReLU (F1(XA))) where F1 and F2 are feed-forward functions of the form W x + b. 2.3 

Putting it all together 
Each Transformer block can be expressed as: XA = LayerNorm(MultiheadAttention(X, X)) + X XB = LayerNorm(PositionFFN(XA)) + XA where X is the input of the Transformer block and XB is the output of the Transformer block. Note that the MultiheadAttention() function accepts two argument tensors, one for query and the other for key-values. If the first argument and second argument is the same input tensor, this is the MultiheadSelfAttention mechanism.” ([Tay et al., 2022, p. 4](zotero://select/library/items/SLWQVGHF)) ([pdf](zotero://open-pdf/library/items/PDDJFS9K?page=4&annotation=9YTWUXLT))

“The Transformer model leverages a multi-headed self-attention mechanism. The key idea behind the mechanism is for each element in the sequence to learn to gather from other tokens in the sequence.” ([Tay et al., 2022, p. 4](zotero://select/library/items/SLWQVGHF)) ([pdf](zotero://open-pdf/library/items/PDDJFS9K?page=4&annotation=W8Z7PXZG))

“2.4 On the compute cost of Transformers The computation costs of Transformers is derived from multiple factors. Firstly, the memory and computational complexity required to compute the attention matrix is quadratic in the” ([Tay et al., 2022, p. 4](zotero://select/library/items/SLWQVGHF)) ([pdf](zotero://open-pdf/library/items/PDDJFS9K?page=4&annotation=WWNRMPR8))

“Efficient Transformers: A Survey input sequence length, i.e., N × N . In particular, the QK> matrix multiplication operation alone consumes N 2 time and memory. This restricts the overall utility of self-attentive models in applications which demand the processing of long sequences. Memory restrictions are tend to be applicable more to training (due to gradient updates) and are generally of lesser impact on inference (no gradient updates). The quadratic cost of self-attention impacts speed1 in both training and inference. The compute costs of the self-attention mechanism contributes partially to the overall compute cost of the Transformer. A non-trivial amount of compute still stems from the two layer feed-forward layers at every Transformer block (approximately half the compute time and/or FLOPs). The complexity of the FFN is linear with respect to sequence length but is generally still costly. Hence, a large portion of recent work have explored sparsity (Lepikhin et al., 2020; Fedus et al., 2021) as a means to scale up the FFN without incurring compute costs.” ([Tay et al., 2022, p. 5](zotero://select/library/items/SLWQVGHF)) ([pdf](zotero://open-pdf/library/items/PDDJFS9K?page=5&annotation=ARZR6SPM))

“In the encoder mode, there is no restriction or constraint that the self-attention mechanism has to be causal, i.e., dependent solely on the present and past tokens.” ([Tay et al., 2022, p. 5](zotero://select/library/items/SLWQVGHF)) ([pdf](zotero://open-pdf/library/items/PDDJFS9K?page=5&annotation=SCK5VP9X))

“In the encoder-decoder setting, self-attention used in the decoder (i.e. across decoding positions) must be causal since each auto-regressive decoding step can only depend on previous tokens, whereas the selfattention used in the encoder need not.” ([Tay et al., 2022, p. 5](zotero://select/library/items/SLWQVGHF)) ([pdf](zotero://open-pdf/library/items/PDDJFS9K?page=5&annotation=52AJK9KK))

“Relative Positional Encodings Transformer-XL introduces novel relative position encodings. In this scheme, absolute positional encodings are not added to the content embeddings. Instead, they are only considered while computing attention weights where they can be replaced with relative position encodings. Since the relative position encodings are not directly relevant to the efficiency of the model, we refer interested readers to Dai et al. (2019) for more details.” ([Tay et al., 2022, p. 24](zotero://select/library/items/SLWQVGHF)) ([pdf](zotero://open-pdf/library/items/PDDJFS9K?page=24&annotation=NMBNLED6))