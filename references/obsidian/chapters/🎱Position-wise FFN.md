
#tansformer #layernorm #feed-forward #position-wise-ffn #activation_function 

Besides the attention sub-layer ([[🅰️Attention]]), the transformer block contains a two-layered feed-forward network, which is applied to each embedding separately and identically ([[@vaswaniAttentionAllYou2017]]) (p. 5). The network consists of a linear transformation, followed by a non-linear activation function, and second linear layer. For the $l$-th layer, the network is given by:
$$
\tag{1}
X = X+W_{\mathrm{mlp} 2}^l \operatorname{ReLU}\left(W_{\mathrm{mlp} 1}^l X+b_{\mathrm{mlp} 1}^l 1^{\top}\right)+b_{\mathrm{mlp} 2}^l 1^{\top}.
$$

%%
```python
class PositionWiseFFN(nn.Module): """Positionwise feed-forward network.""" def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs): super(PositionWiseFFN, self).__init__(**kwargs) self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens) self.relu = nn.ReLU() self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs) def forward(self, X): return self.dense2(self.relu(self.dense1(X)))
```
%%
with $W_{\text{mlp1}}^{l}, b_{\text{mlp1}}^{l},W_{\text{mlp2}}^{l}, b_{\text{mlp2}}^{l}$ <mark style="background: #FFB8EBA6;"> (dims?) </mark>being learnable parameters identical for all embeddings in the layer. In the Transformer, the point-wise feed-forward network acts as a persistent memory, capable of storing general information on the task and independent from the immediate context ([[@sukhbaatarAugmentingSelfattentionPersistent2019]]) (p. 3). (see also [[🤖Transformer#^54aa8a]] and [[@gevaTransformerFeedForwardLayers2021]]). [[@vaswaniAttentionAllYou2017]] (p. 9) set the hidden dimension to be two to eight magnitudes of the embedding dimension. The large capacity, strengthens the model's ability to retain information but also contributes significantly to the high computational requirements and memory footprint of transformers and has been in focus of research (See [[@tayEfficientTransformersSurvey2022]] (p. 5); [[@kitaevReformerEfficientTransformer2020]] (p. 1)).  

The classical transformer of [[@vaswaniAttentionAllYou2017]] (p. 5) uses the *Rectified Linear Units* $\operatorname{ReLU}$ ([[@glorotDeepSparseRectifier2011]]; p. 318) activation (see Equation (1)) given by $\max(0,X)$. Later variants (see e. g., [[@devlinBERTPretrainingDeep2019]] or [[@radfordImprovingLanguageUnderstanding]]) commonly replace the $\operatorname{ReLU}$ with *Gaussian Error Linear Units* $\operatorname{GELU}$ ([[@hendrycksGaussianErrorLinear2020]], p. 2) activation, which has empirically proven to improve the performance and convergence behaviour of transformers ( [[@narangTransformerModificationsTransfer2021]], p. 16; and [[@shazeerGLUVariantsImprove2020]] p. 4). Both activation functions are also available as *gated linear units* ([[@dauphinLanguageModelingGated2017]], p. ), which are the element-wise product of two linear transformations of which one is activated, and may be a drop-in for the first linear layer and activation function ([[@shazeerGLUVariantsImprove2020]]; p. 2).

Like the [[🅰️Attention]] sub-layer, the feed-forward sub-layer is surrounded by residual connections ([[@heDeepResidualLearning2015]]) and followed by a layer-normalization ([[@baLayerNormalization2016]] (p. 4)) layer. Sometimes additional dropout ([[@srivastavaDropoutSimpleWay]] p. 1,930) is applied to prevent the model from overfitting. 

**Notes:**
[[🎱Position-wise FFN notes]]