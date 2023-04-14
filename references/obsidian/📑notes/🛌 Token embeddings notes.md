

To this end, embeddings are non-exclusive to Transformer-based architectures, and can be used in other deep learning-based approaches, and even classical machine learning models, like [[🐈Gradient Boosting]]. Covering these combinations is outside the scope of this work. We refer the reader to [[@gorishniyEmbeddingsNumericalFeatures2022]] for an in-depth comparison. Next, our focus is on two concrete examples of Transformers for tabular data.


## Short and concise notion of embeddings

Let SN = {wi} N i=1 be a sequence of N input tokens with wi being the i th element. The corresponding word embedding of SN is denoted as EN = {xi} N i=1, where xi ∈ R d is the d-dimensional word embedding vector of token wi without position information. The self-attention first incorporates position information to the word embeddings and transforms them into queries, keys, and value representations. qm = fq(xm, m) kn = fk(xn, n) vn = fv(xn, n), (1) where qm, kn and vn incorporate the mth and n th positions through fq, fk and fv, respectively. The query and key values are then used to compute the attention weights, while the output is computed as the weighted sum over the value 2 RoFormer representation. am,n = exp( q | m√ kn d ) PN j=1 exp( q | √mkj d ) om = X N n=1 am,nvn (2) The existing approaches of transformer-based position encoding mainly focus on choosing a suitable function to form Equation (1).
(from [[@suRoFormerEnhancedTransformer2022]])

## Embeddings and Grokking

4 DISCUSSION We have seen that in the datasets we studied, small algorithmic binary operation tables, effects such as double descent or late generalization, and improvements to generalization from interventions like weight decay can be striking. This suggests that these datasets could be a good place to investigate aspects of generalization. For example, we plan to test whether various proposed measures of minima flatness correlate with generalization in our setting. We have also seen that visualizing the embedding spaces of these neural networks can show natural kinds of structure, for example in problems of modular arithmetic the topology of the embeddings tends to be circles or cylinders. We also see that the network tends to idiosyncratically organize the embeddings by various residues. Whilst the properties of these mathematical objects are familiar to us, we speculate that such visualizations could one day be a useful way to gain intuitions about novel mathematical objects. 

3.4 QUALITATIVE VISUALIZATION OF EMBEDDINGS In order to gain some insight into networks that generalize, we visualized the matrix of the output layer for the case of modular addition and S5. In Figure 3 we show t-SNE plots of the row vectors. For some networks we find clear reflections of the structure of the underlying mathematical objects in the plots. For example the circular topology of modular addition is shown with a ‘number line’ formed by adding 8 to each element. The structure is more apparent in networks that were optimized with weight decay. https://arxiv.org/pdf/2201.02177.pdf

![[visualization-token-embedding.png]]
(from [[@powerGrokkingGeneralizationOverfitting2022]])


## Visualization

![[Pasted image 20230414105714.png]]
found in https://www.nature.com/articles/s42256-022-00532-1

![[visualization-of-word-embeddings.png]]


## Difference equivariant / invariant:
(see https://datascience.stackexchange.com/a/99892/142202)
![[equivariance-invariance.png]]


TODO: Use dot-product instead, to be coherent to the idea used in attention? Not scaled by magnitude?
See here. https://datascience.stackexchange.com/questions/744/cosine-similarity-versus-dot-product-as-distance-metrics


```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
x = np.array([0.01, 0.20, 0.134]).reshape(-1,1)
y = np.array([0.07, 0.157, 0.139]).reshape(-1,1)
cosine_sim = cosine_similarity(x, y)
print(cosine_sim)

[[1. 1. 1.] [1. 1. 1.] [1. 1. 1.]]

```


## Numerical embeddings: Why an how?
See: https://blog.ayoungprogrammer.com/2018/01/deep-recurrent-neural-networks-for-mathematical-sequence-prediction.html
Encoding numerical inputs for neural networks is difficult because the representation space is very large and there is no easy way to embed numbers into a smaller space without losing information. Some of the ways to currently handle this is:

-   Scale inputs from minimum and maximum values to [-1, 1]
-   One hot for each number
-   One hot for different bins (e.g. [0-0], [1-2], [3-7], [8 – 19], [20, infty])

In small integer number ranges, these methods can work well, but they don’t scale well for wider ranges. In the input scaling approach, precision is lost making it difficult to distinguish between two numbers close in value. For the binning methods, information about the mathematical properties of the numbers such as adjacency and scaling is lost.

The desideratum of our embeddings of numbers to vectors are as follows:

-   able to handle numbers of arbitrary length
-   captures mathematical relationships between numbers (addition, multiplication, etc.)
-   able to model sequences of numbers

In this blog post, we will explore a novel approach for embedding numbers as vectors that include these desideratum.


## Notes from Phuong and Hutter
(see [[@phuongFormalAlgorithmsTransformers2022]])
![[token-embedding.png]]

## Notes from e2eml

In an embedding, those word points are all taken and rearranged (**projected**, in linear algebra terminology) into a lower-dimensional space. The picture above shows what they might look like in a 2-dimensional space for example. Now, instead of needing _N_ numbers to specify a word, we only need 2. These are the (_x_, _y_) coordinates of each point in the new space. Here's what a 2-dimensional embedding might look like for our toy example, together with the coordinates of a few of the words.

![](https://e2eml.school/images/transformers/embedded_words.png)

A good embedding groups words with similar meanings together. A model that works with an embedding learns patterns in the embedded space. That means that whatever it learns to do with one word automatically gets applied to all the words right next to it. This has the added benefit of reducing the amount of training data needed. Each example gives a little bit of learning that gets applied across a whole neighborhood of words

A good embedding groups words with similar meanings together. **A model that works with an embedding learns patterns in the embedded space.** That means that whatever it learns to do with one word automatically gets applied to all the words right next to it. This has the added benefit of reducing the amount of training data needed. Each example gives a little bit of learning that gets applied across a whole neighborhood of words.

## Notes from Chris Olah
From https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/:

I’d like to start by tracing a particularly interesting strand of deep learning research: word embeddings. In my personal opinion, word embeddings are one of the most exciting area of research in deep learning at the moment, although they were originally introduced by Bengio, _et al._ more than a decade ago.(see [[@bengioNeuralProbabilisticLanguage]]) Beyond that, I think they are one of the best places to gain intuition about why deep learning is so effective.


A word embedding $W:$ words $\rightarrow \mathbb{R}^n$ is a paramaterized function mapping words in some language to high-dimensional vectors (perhaps 200 to 500 dimensions). For example, we might find:
$$
\begin{aligned}
& W(\text { "cat" })=(0.2,-0.4,0.7, \ldots) \\
& W(\text { "mat" })=(0.0,0.6,-0.1, \ldots)
\end{aligned}
$$
(Typically, the function is a lookup table, parameterized by a matrix, $\theta$, with a row for each word: $\left.W_\theta\left(w_n\right)=\theta_{n-}\right)$
$W$ is initialized to have random vectors for each word. It learns to have meaningful vectors in order to perform some task.
WW is initialized to have random vectors for each word. It learns to have meaningful vectors in order to perform some task.

Word embeddings exhibit an even more remarkable property: analogies between words seem to be encoded in the difference vectors between words. For example, there seems to be a constant male-female difference vector:
$$
\begin{aligned}
& W(\text { ''woman" })-W(\text { ''man") }) \simeq W(\text { ''aunt" })-W(\text { ''uncle" }) \\
& W(\text { ''woman" })-W(\text { 'man" }) \simeq W(\text { 'queen" })-W(\text { ('king") }
\end{aligned}
$$
This may not seem too surprising. After all, gender pronouns mean that switching a word can make a sentence grammatically incorrect. You write, "she is the aunt" but "he is the uncle." 

Example is adapted from [[@mikolovLinguisticRegularitiesContinuous2013]]


## Notes from Vaswani
From [[@vaswaniAttentionAllYou2017]] :
“Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.” (Vaswani et al., 2017, p. 5)

## Notes from Rothman
From [[@rothmanTransformersNaturalLanguage2021]]:

The embedding sub-layer works like other standard *transduction models*. A tokenizer will transform a sentence into tokens.

“The Transformer contains a learned embedding sub-layer. Many embedding methods can be applied to the tokenized input. I chose the skip-gram architecture of the word2vec embedding approach Google made available in 2013 to illustrate the embedding sublayer of the Transformer.” (Rothman, 2021, p. 9)

“A skip-gram will focus on a center word in a window of words and predicts context words. For example, if word(i) is the center word in a two-step window, a skipgram model will analyze word(i-2), word(i-1), word(i+1), and word(i+2). Then the window will slide and repeat the process. A skip-gram model generally contains an input layer, weights, a hidden layer, and an output containing the word embeddings of the tokenized input words.” (Rothman, 2021, p. 9)

“To verify the word embedding produced for these two words, we can use cosine similarity to see if the word embeddings of the words black and brown are similar. Cosine similarity uses Euclidean (L2) norm to create vectors in a unit sphere. The dot product of the vectors we are comparing is the cosine between the points of those two vectors.” (Rothman, 2021, p. 10)

“The Transformer's subsequent layers do not start empty-handed. They have learned word embeddings that already provide information on how the words can be associated. However, a big chunk of information is missing because no additional vector or information indicates a word's position in a sequence.” (Rothman, 2021, p. 10)