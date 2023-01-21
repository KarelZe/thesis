*title:* Efficient Transformers: A Survey
*authors:* Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler
*year:* 2022
*tags:* 
*status:* #📥
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖

### Architecture
- transformers are a multi-layered architecture formed by stacking transformer blocks on top of one another.
- Transformer blocks are characterized by a multi-head sel-attention mechanism, a poistion-wise feed-forward network, layer norm modules ([[@baLayerNormalization2016]]) and residual connectors ([[@heDeepResidualLearning2015]])
- The input is passed through an embedding layer and converts one-hot tokens into a $d_{\text{model}}$ dimensional embedding. The tensor is composed with a positional encoding and passed through a multi-headed self-attention module. 
- Inputs and outputs oft he multi-headed self-attention module are connected by residual connectors and a layer normalization layer. the output of the multi-headed self-attention module is then passed to a two-layered feed forward network which has it inputs / outputs similarily connected in a residual fashion with layer normalization. 
- Each Transformer block can be expressed as:
$$
\begin{aligned}
& \left.X_A=\text { LayerNorm(MultiheadAttention }(X, X)\right)+X \\
& X_B=\text { LayerNorm }\left(\text { PositionFFN }\left(X_A\right)\right)+X_A
\end{aligned}
$$
where $X$ is the input of the Transformer block and $X_B$ is the output of the Transformer block. Note that the MultiheadAttention() function accepts two argument tensors, one for query and the other for key-values. If the first argument and second argument is the same input tensor, this is the MultiheadSelfAttention mechanism.


## MHSA

## Position-wise FFN

The outputs of the self-attention module are then passed into a two-layered feed-forward network with ReLU activations. This feed-forward layer operates on each **position independently**. This is expressed as follows:
$$
F_2\left(\operatorname{Re} L U\left(F_1\left(X_A\right)\right)\right)
$$
where $F_1$ and $F_2$ are feed-forward functions of the form $W x+b$.

See also [[🤖Transformer#^54aa8a]] he says that is the memory of the transformer, and that every token is processed separately. 

ReLU was proposed in [[@glorotDeepSparseRectifier2011]].