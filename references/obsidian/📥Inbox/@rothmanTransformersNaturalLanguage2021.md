*title:* Transformers for Natural Language Processing
*authors:* Denis Rothman
*year:* 2021
*tags:* 
*status:* #📥
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖

The original encoder layer structure remains the same for all of the $N=6$ layers of the Transformer model. Each layer contains two main sub-layers: a multi-headed attention mechanism and a fully connected position-wise feedforward network.
Notice that a residual connection surrounds each main sub-layer, Sublayer $(x)$, in the Transformer model. These connections transport the unprocessed input $x$ of a sublayer to a layer normalization function. This way, we are certain that key information such as positional encoding is not lost on the way. The normalized output of each layer is thus:
LayerNormalization $(x+$ Sublayer $(x))$
Though the structure of each of the $\mathrm{N}=6$ layers of the encoder is identical, the content of each layer is not strictly identical to the previous layer.

“model = 512” ([Rothman, 2021, p. 11](zotero://select/library/items/QDB3526T)) ([pdf](zotero://open-pdf/library/items/GLMXG7M9?page=34&annotation=7WEJLZ8T))

“For each word embedding vector, we need to find a way to provide information to i in the range(0,512) dimensions of the word embedding vector of black and brown.” ([Rothman, 2021, p. 11](zotero://select/library/items/QDB3526T)) ([pdf](zotero://open-pdf/library/items/GLMXG7M9?page=34&annotation=BY5QNPJ6))

“unit sphere to represent positional encoding with sine and cosine values that will thus remain small but very useful” ([Rothman, 2021, p. 11](zotero://select/library/items/QDB3526T)) ([pdf](zotero://open-pdf/library/items/GLMXG7M9?page=34&annotation=VPM77IMT))

“The authors of the Transformer found a simple way by merely adding the positional encoding vector to the word embedding vector:” ([Rothman, 2021, p. 15](zotero://select/library/items/QDB3526T)) ([pdf](zotero://open-pdf/library/items/GLMXG7M9?page=38&annotation=DNEHJSJ3))

“f we go back and take the word embedding of black, for example, and name it y1=black, we are ready to add it to the positional vector pe(2) we obtained with positional encoding functions. We will obtain the positional encoding pc(black) of the input word black: pc(black)=y1+pe(2) The solution is straightforward. However, if we apply it as shown, we might lose the information of the word embedding, which will be minimized by the positional encoding vector.” ([Rothman, 2021, p. 15](zotero://select/library/items/QDB3526T)) ([pdf](zotero://open-pdf/library/items/GLMXG7M9?page=38&annotation=ZUDH7GSS))

“One of the many possibilities is to add an arbitrary value to y1, the word embedding of black: y1\*math.sqrt(dmodel) We can now add the positional vector to the embedding vector of the word black, both of which are the same size (512):” ([Rothman, 2021, p. 16](zotero://select/library/items/QDB3526T)) ([pdf](zotero://open-pdf/library/items/GLMXG7M9?page=39&annotation=RSU5TDDK))