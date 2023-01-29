
![[context-xl-transformer.png]]
(found in [[@daiTransformerXLAttentiveLanguage2019]])


## Attention

<mark style="background: #ADCCFFA6;">An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. (unknown)</mark>


## Dot-product attention

<mark style="background: #BBFABBA6;">“Dot-product attention. The standard attention used in the Transformer is the scaled dot-product attention (Vaswani et al., 2017). The input consists of queries and keys of dimension dk, and values of dimension dv. The dot products of the query with all keys are computed, scaled by √dk, and a softmax function is applied to obtain the weights on the values. In practice, the attention function on a set of queries is computed simultaneously, packed together into a matrix Q. Assuming the keys and values are also packed together into matrices K and V , the matrix of outputs is defined as: Attention(Q, K, V ) = softmax( QKT √dk )V” ([Kitaev et al., 2020, p. 2](zotero://select/library/items/D93TNTMS)) ([pdf](zotero://open-pdf/library/items/5F5L22PR?page=2&annotation=YLGCST6K))</mark>


## Multi-headed attention
- [[@vaswaniAttentionAllYou2017]] propose to run several attention heads in parallel, instead of using a single attention heads. They write it improves performance and has a similar computational cost to the single-headed version. Multiple heads may also learn different patterns, not all are equally important, and some can be pruned. This means given a specific context there attention heads learn to attend to different patterns. 


<mark style="background: #FFF3A3A6;">“Multi-head attention. In the Transformer, instead of performing a single attention function with dmodel-dimensional keys, values and queries, one linearly projects the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. Attention is applied to each of these projected versions of queries, keys and values in parallel, yielding dvdimensional output values. These are concatenated and once again projected, resulting in the final values. This mechanism is known as multi-head attention.” ([Kitaev et al., 2020, p. 2](zotero://select/library/items/D93TNTMS)) ([pdf](zotero://open-pdf/library/items/5F5L22PR?page=2&annotation=P89D5VB5))</mark>

<mark style="background: #FFB86CA6;">The Transformer model leverages a multi-headed self-attention mechanism. The key idea behind the mechanism is for each element in the sequence to learn to gather from other tokens in the sequence. The operation for a single head is defined as:
$$
A_h=\operatorname{Softmax}\left(\alpha Q_h K_h^{\top}\right) V_h,
$$
where $X$ is a matrix in $\mathbb{R}^{N \times d}, \alpha$ is a scaling factor that is typically set to $\frac{1}{\sqrt{d}}, Q_h=$ $X \boldsymbol{W}_q, K_h=X \boldsymbol{W}_k$ and $V_h=X \boldsymbol{W}_v$ are linear transformations applied on the temporal dimension of the input sequence, $\boldsymbol{W}_q, \boldsymbol{W}_k, \boldsymbol{W}_v \in \mathbb{R}^{d \times \frac{d}{H}}$ are the weight matrices (parameters) for the query, key, and value projections that project the input $X$ to an output tensor of $d$ dimensions, and $N_H$ is the number of heads. Softmax is applied row-wise.

The outputs of heads $A_1 \cdots A_H$ are concatenated together and passed into a dense layer. The output $Y$ can thus be expressed as $Y=W_o\left[A_1 \cdots A_H\right]$, where $W_o$ is an output linear projection. Note that the computation of $A$ is typically done in a parallel fashion by considering tensors of $\mathbb{R}^B \times \mathbb{R}^N \times \mathbb{R}^H \times \mathbb{R}^{\frac{d}{H}}$ and computing the linear transforms for all heads in parallel.

The attention matrix $A=Q K^{\top}$ is chiefly responsible for learning alignment scores between tokens in the sequence. In this formulation, the dot product between each element/token in the query $(Q)$ and key $(K)$ is taken. This drives the self-alignment process in self-attention whereby tokens learn to gather from each other. (Tay; p. 4)</mark>

<mark style="background: #D2B3FFA6;">The computation costs of Transformers is derived from multiple factors. Firstly, the memory and computational complexity required to compute the attention matrix is quadratic in the input sequence length, i.e., $N \times N$. In particular, the $Q K^{\top}$ matrix multiplication operation alone consumes $N^2$ time and memory. This restricts the overall utility of self-attentive models in applications which demand the processing of long sequences. Memory restrictions are tend to be applicable more to training (due to gradient updates) and are generally of lesser impact on inference (no gradient updates). The quadratic cost of self-attention impacts (Tay; p. 4 f.)
</mark>

- interesting blog post on importance of heads and pruning them: https://lena-voita.github.io/posts/acl19_heads.html
- Heads have a different importance and many can even be pruned: https://proceedings.neurips.cc/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf
- It's just a logical concept. A single data matrix is used for the Query, Key, and Value, respectively, with logically separate sections of the matrix for each Attention head. Similarly, there are not separate Linear layers, one for each Attention head. All the Attention heads share the same Linear layer but simply operate on their ‘own’ logical section of the data matrix.
- https://transformer-circuits.pub/2021/framework/index.html


## Notes from Peltarion 📺
1. Calculate dot product between two vectors, if two embeddings are more correlated / similar. Scalar prodcut is higher. Embeddings relate to each other. 
2. Square root is used to avoid getting large values
3. Non-linear softmax activation. Exponentially amplifies large values and brings small values to zero. Performs normalization.
4. New embeddings are contextualized, as each embedding now contains a  fraction of the other embeddings. If a embedding has hardly any relationship to other tokens, its embedding is similar to the input embedding.

- Typically projections are used to make calculation feasible. We create key, value, and query vectors. 
- multi-head atttention uses different projections to learn different aspect e. g., propisition-location relationships etc. (from https://peltarion.com/blog/data-science/self-attention-video)

## Notes from Hedu AI 📺
See: https://youtu.be/mMa2PmYJlCo
- We judge the meaning of the word by the context in which the word appears
- simple attention focuses only on words with respect to a query. self-attention takes into account the relationship between any word in the sentence.
- Linear layers change the dimensionality through projection. Usually done to save computation cost.
- We got linear layers for query, key, and value. Copies of the embedding are fed to the queries, keys, and values. Copies! Each linear layer has its own set of weights. 
- Similarity is calculated between two vectors using cosine similarity
$$
\text { similarity }(Q, K)=\frac{Q \cdot K^T}{\text { scaling }}
$$
- Output of the dot product is the attention filter. Attention filter contains the attention scores. Scale them and squash them using softmax activation.
- Multiplication with values and attention filter gives a filter value matrix which assigns a high focus to features that are more important.
- Multi-headed attention is used to learn multiple features to learn different linguistic phenomenon. Output is concatenated and projected back. 
- Multi-head attention layer in the decoder takes three inputs. Query, key come from the encoder, and values.
- last layer has the capacity of the vocabulary. 
- raw scores are called logits. Finally softmax is applied to obtain the probabilities.
- In training model gets provided with source and target dialog. Target dialog is masked to come up with own answers.
- Masking operation is a filter matrix, whereby neg inf is added to all words afterwords. Neg inf become 0 in softmax layer.


The study of the transformer architecture has focused on the role and function of self-attention layers (Voita et al., 2019; Clark et al., 2019; Vig and Belinkov, 2019) and on inter-layer differences (i.e. lower vs. upper layers) (Tenney et al., 2019; Jawahar et al., 2019). (look up citations in [[@gevaTransformerFeedForwardLayers2021]])

## Self-attention
- https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/
- Detailed explanation and implementation. Check my understanding against it: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- See [[@michelAreSixteenHeads2019]] for discussion if all heads are necessary
- https://ai.stanford.edu/blog/contextual/
- good blog post for intuition https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
- good blog post for inuition on attention: https://storrs.io/attention/
Nice overview over attention and self-attention:
- https://slds-lmu.github.io/seminar_nlp_ss20/attention-and-self-attention-for-nlp.html
- https://angelina-yang.medium.com/whats-the-difference-between-attention-and-self-attention-in-transformer-models-2846665880b6
- Course on nlp / transformers: https://phontron.com/class/anlp2022/schedule.html
- https://www.borealisai.com/research-blogs/tutorial-16-transformers-ii-extensions/
- discuss problems with computational complexity and that approximations exist
- nice explanation of transformers, such as dot-product attention https://t.co/WOlBY3suy4
- low-level  overview. Fully digest these ideas: https://transformer-circuits.pub/2021/framework/index.html
