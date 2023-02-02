
![[context-xl-transformer.png]]
(found in [[@daiTransformerXLAttentiveLanguage2019]])

[[@dosovitskiyImageWorth16x162021]]
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

## Notes from Angelina Yang 🔗
(see here: https://angelina-yang.medium.com/whats-the-difference-between-attention-and-self-attention-in-transformer-models-2846665880b6 and https://www.youtube.com/clip/UgkxwDACXNvxoVHOisH-BgWt22_gGH7mOYt8)
**Difference self-attention and cross-attention:** **_Attention_** _connecting between the encoder and the decoder is called_ **_cross-attention_** _since keys and values are generated by a_ **_different_** _sequence than queries. If the keys, values, and queries are generated from the_ **_same_** _sequence, then we call it_ **_self-attention_**_. The attention mechanism allows output to focus attention on input when producing output while the self-attention model allows inputs to interact with each other._

## Notes from Carnegie Melon 👨‍🎓
(see here: https://phontron.com/class/anlp2022/assets/slides/anlp-06-attention.pdf)

- Idea of Badanau et al. Encode each word in a sentence as a vector. When decoding, perform a linear combination of these vectors, weighted by the attention weights. Use the weighted combination for picking the next word.
- Uses a query vector ( decoder state) and key vector (all encoder states). For each query-key pair the weight is calculated. Finally, result is normalized using softmax. The values are then combined as the weighted sum.  Keys are weighted by the attention weights.
- Scale of dot product increases with dimension of vectors, thats why scaled dot product normalizes by the size of the vector. $$
a(\boldsymbol{q}, \boldsymbol{k})=\frac{\boldsymbol{q}^{\boldsymbol{\top}} \boldsymbol{k}}{\sqrt{|\boldsymbol{k}|}}
$$
- Self-attention was first done in Cheng et all. Each element in the sentence attends to ther elements. Leads to context-sensitive emebeddings.
- Idea of multi-headed attention is, to use multiple attention heads, so that the model can focus on different parts of the sentence. Heads are learned independently.
- Training tricks: layer norm helps to ensure layers remain in a reasonable range, specialized training schedule: adjust default learning  rate of Adam, label smoothing insert uncertainity in the training process, and masking
- Perform training in as few operations as possible using big matrix multiplies. Mask some parts of the output

## Notes from storrs.io🕸️
(see here: https://storrs.io/multihead-attention/)
- Multi-head attention allows for the neural network to control the mixing of information between pieces of an input sequence, leading to the creation of richer representations, which in turn allows for increased performance on machine learning tasks.
- The key concept behind self attention is that it allows the network to learn how best to route  information between pieces of a an input sequence (known as _tokens_).
- Attention applies to **any** type of data that can be formatted as a sequence.
- We got queries, keys, and  values. We then take those queries and match them against a series of _keys_ that describe _values_ that we want to know something about. The similarity of a given query to the keys determines how much information from each value to retrieve for that particular query
- The output of the first matrix multiplication, where we take the similarity of each query to each of the keys, is known as the _attention matrix_.
- The attention matrix depicts how much each token in the sequence is paying attention to each of the keys (hence the _n_ x _n_ shape)
- These values are then passed through a softmax function to scale the values to a probability distribution that adds up to 1, and also sharpens the distribution so the higher values are even higher and lower values even lower.
- After we have our attention values, we need to index the information contained in the value matrix V, because, after all, now that we know _how_ we want to route information, we need to actually route the information contained in the values
- Multi-head attention allows for the network to learn multiple ways to "index" the value matrix _V_. In vanilla attention (without multiple heads) the network only learns one way to route information. In theory this allows for the network to learn a richer final representation of the input data.
- To start, the keys, queries, and values input matrices are simply broken up by feature into different _heads_, each of which are responsible for learning a different representation.
- Layer normalization is identical to batch correction, except instead of acting on the sample dimension _n,_ LayerNorm normalizes the feature dimension _d_.

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


## Notes from towards datascience
(see here: https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

- self-attention in the encoder: input sequence pays attention to itself
- self-attention in the decoder: target sequence pays attention to itself
- encoder-decoder attention: target sequence pays attention to input sequence
- attention layer takes 3 inputs: query, key, and values
- encoder self-attention produces the encoded representation for each word in the input sequence and the attention scores for each word
- decoder self-attention produces an encoded representation of each word in the target sequence, which incorporates attention scores for each word
- the encoder-decoder attention produces a representation of the target sequence and representation of the input sequence. The representation of the attention scores for each target sequence word capture the influence from the input sequence as well. Thus,  all self-attention and corss-attention add their own attention scores to the word representation.
- Calculation of the attention head is repeated multiple times in parallel with attention heads. Each attention module splits queries, keys and values N-ways. All of the similar Attention calculations are combined to produce the final Attention scores. Multi-headed attention give the transformer greater power at encoding multiple relationships and nuances for each word.
- Each linear layer has its own weights and produces the Q, K, V matrices.
- Attention heads split the data only logically. All attention heads share the same linear layer but simply operate on their own logical section of the data matrix. The splitting is done with a logical split across the attention heads. The dimension is given by the query size = Embedding size / no of heads
- attention scores from each the heads need to be combined. Reverse the splitting through reshaping into (batch, sequence, head * query size)
- Multi-headed attention: each section of the embedding can learn different aspect of the meanings of each word, as it realtes to other words, in the sequence. Thus, the transformer can learn richer representations.

## Notes from Tenney et al
[[@tenneyBERTRediscoversClassical2019]]

## Notes from Jawahar et al 
[[@jawaharWhatDoesBERT2019]]

## Notes from Michel



## Self-attention
- https://www.borealisai.com/research-blogs/tutorial-14-transformers-i-introduction/
- https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/
- https://www.borealisai.com/research-blogs/tutorial-16-transformers-ii-extensions/
- Detailed explanation and implementation. Check my understanding against it: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- See [[@michelAreSixteenHeads2019]] for discussion if all heads are necessary
- good blog post for intuition https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853


Nice overview over attention and self-attention:
- https://slds-lmu.github.io/seminar_nlp_ss20/attention-and-self-attention-for-nlp.html

- low-level  overview. Fully digest these ideas: https://transformer-circuits.pub/2021/framework/index.html
