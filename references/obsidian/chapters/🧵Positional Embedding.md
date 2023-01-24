related: 
#transformer #positional-encoding #linear-algebra #linear-projection #cylic-data

![[classical_transformer_architecture.png]]
(own drawing after [[@daiTransformerXLAttentiveLanguage2019]])

[[@vaswaniAttentionAllYou2017]] (p. 6) propose to inject information on the token's position within the sequence through a *positional embedding*, that is added to the token embedding from Section [[🛌Token Embedding]]. Like token embeddings, positional embeddings can also be learned. Due to better, extrapolation capabilities, [[@vaswaniAttentionAllYou2017]] (p. 6), propose an embedding $W_p: \mathbb{N} \rightarrow \mathbb{R}^{d_{\mathrm{e}}}$ based on sine and cosine signals to encode the *absolute* position of the token:
$$
\tag{1}
\begin{aligned}
W_p[2 i-1, t] & =\sin \left(t / \ell_{\max }^{2 i / d_e}\right), \\
W_p[2 i, t] & =\cos \left(t / \ell_{\max }^{2 i / d_e}\right) .
\end{aligned}
$$
with $0<i \leq d_{\mathrm{e}} / 2$, the maximum sequence length $\ell_{\max}$, which is arbitrarily set to $\ell_{\max}=10,000$, and $t$ is again the position of the token in the sequence. As shown in Equation (1) the frequency decreases across the position dimension and alternates between sine and cosine for the embedding dimension. Each embedding thus contains a pattern, easily distinguishable by the model.

<mark style="background: #FFB8EBA6;">(TODO: make clear, why it is needed in the first place)</mark>

Using trigonometric functions for the positional embedding is favourable, due to being zero-centered, and resulting in values in the *limited* range of $[-1,1]$. These properties are long known to promote convergence of neural networks (cp. [[@lecunEfficientBackProp2012]] (p. 8 f)). The reason for encoding with both the sine and cosine is more subtle. [[@vaswaniAttentionAllYou2017]] (p. 6) hypothesize, that beside learning the *absolute position* i. e., fifth place in sequence, also enables to model attend to *relative positions*, i. e., two places from a given token. A detailed proof is layed out in  [[@zhangDiveDeepLearning2021]] (p. 410) 

%%
TODO: Think about including the proof in the thesis:

%% 

![[viz-of-pos-encoding.png]]
(pos encoding copied from [[@zhangDiveDeepLearning2021]]; p. 409; transpose image due to my matrix notation) ^1f2fe5

![[positional-encoding-different-view.png]]
(copied from: https://www.borealisai.com/research-blogs/tutorial-16-transformers-ii-extensions/)

We visualize the positional embedding in Figure [[#^1f2fe5]] with an embedding dimension of $d_e=32$ and 60 tokens. One can clearly see the pattern describing the position.

The positional embedding of a token is finally added to the token embedding to form a token's initial embedding $e$. For the $t$-th token of a sequence $x$, the embedding becomes:
$$
e=W_e[:, x[t]]+W_p[:, t] .
$$
Due to the sheer depth of of the network with multiple Transformer blocks, the positional information would easily vanish during back-propagation. To enforce it, the architecture relies on residual connections [[@heDeepResidualLearning2015]] (p. 3).

<mark style="background: #FFB8EBA6;">(TODO: What is intuition behind adding a positional encoding -> shift in space (see talk of lucas beyer / rothman) )</mark>

%%
Nice visuals: https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers

ResNet paper on residual learning / residual connections. Discusses in general the problems that arise with learning deep neural networks: https://arxiv.org/pdf/1512.03385.pdf
Nice explanation: https://stats.stackexchange.com/a/565203/351242
%%

Later works, like [[@daiTransformerXLAttentiveLanguage2019]] (p. 4 f.) <mark style="background: #FFB8EBA6;">(also found in [[@tayEfficientTransformersSurvey2022]]; p. 24)</mark>, remove the positional encoding in favour of *relative position encoding*, that is only considered during computation. <mark style="background: #FFB8EBA6;">(see also [[@tunstallNaturalLanguageProcessing2022]] (p. 74). There is a short section that describes *relative, positional embeddings*).</mark> 

%%
nice blog post with proof:
https://www.borealisai.com/research-blogs/tutorial-16-transformers-ii-extensions/
%%

Also, if the order of the input is arbitrary, a positional encoding may be dropped [[@huangTabTransformerTabularData2020]]  (p. 3). We come back to this observation in chapter [[🤖TabTransformer]].

<mark style="background: #FFB86CA6;">“Positional Encoding Transformers for vision and language typically employ positional encodings along with the patch/word embeddings to retain spatial information. These encodings are necessary when all features in a data point are of same type, hence these models use the same function to embed all inputs. This is not the case with most of the datasets used in this paper; each feature may be of a different type and thus possesses a unique embedding function. However, when we train the model on MNIST (treated as tabular data), positional encodings are used since all pixels are of the same type and share a single embedding function.” ([Somepalli et al., 2021, p. 15](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=15&annotation=FB8RJ78J))</mark>

---

Resources:
- nice visualizations: https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
- introduced in [[@vaswaniAttentionAllYou2017]]
- nice consistent notation in [[@phuongFormalAlgorithmsTransformers2022]]
- nice summary and well explained in [[@zhangDiveDeepLearning2021]]. Use it to crosscheck my understanding!
- [[@daiTransformerXLAttentiveLanguage2019]] removes the positional encoding altogether.

**Notes:**
- The encoder (the self-attention and feed-forward layers) are said to be permutation equivariant—if the input is permuted then the corresponding output of the layer is permuted in exactly the same way.
- Add information about the position in sequence to the embedding
- The vector $p_i$ is of dimension $d_{\text{model}}$  and is added (not concat'd! -> to keep no of params small?) to the embedding vector.
- Note that there is a subtle difference between positional embedding and encoding. When reading [[@rothmanTransformersNaturalLanguage2021]] I noticed, that word `positional encoding = embedding + positional vector`
- There are obviously different types of *positional embeddings* e. g., learnable embeddings, absolute positional representations and relative positional representations. ([[@tunstallNaturalLanguageProcessing2022]]) In the [[@vaswaniAttentionAllYou2017]] an *absolute positional encoding* is used. Sine and cosine signals are sued to encode the position of tokens. 
- Positional encoding can be learned or fix. Both seem to work similarily. 
- According to this [reddit post](https://www.reddit.com/r/learnmachinelearning/comments/9e4j4q/positional_encoding_in_transformer_model/), [[@vaswaniAttentionAllYou2017]] chose the fixed encoding with sine and cosine, to enable learning about the relative position in a sequence. It's also documented in the paper, even though it is covered only briefly.
- *Absolute positional embeddings* work well if the dataset is small.

**Visualization:**
![[viz-pos-encoding.png]]
(Code available at https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb)

Visualization of  diffferent columns:
![[freq-of-encoding.png]]
(Image from: https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html)

## Notes from Dive into Deep Learning
(see [[@zhangDiveDeepLearning2021]])
Suppose that the input representation $\mathbf{X} \in \mathbb{R}^{n \times d}$ contains the $d$-dimensional embeddings for $n$ tokens of a sequence. The positional encoding outputs $\mathbf{X}+\mathbf{P}$ using a positional embedding matrix $\mathbf{P} \in \mathbb{R}^{n \times d}$ of the same shape, whose element on the $i^{\text {th }}$ row and the $(2 j)^{\text {th }}$ or the $(2 j+1)^{\text {th }}$ column is
$$
\begin{aligned}
p_{i, 2 j} & =\sin \left(\frac{i}{10000^{2 j / d}}\right), \\
p_{i, 2 j+1} & =\cos \left(\frac{i}{10000^{2 j / d}}\right) .
\end{aligned}
$$
At first glance, this trigonometric-function design looks weird. Before explanations of this design, let's first implement it in the following PositionalEncoding class.

In the positional embedding matrix $\mathbf{P}$, rows correspond to positions within a sequence and columns represent different positional encoding dimensions. In the example below, we can see that the $6^{\mathrm{th}}$ and the $7^{\text {th }}$ columns of the positional embedding matrix have a higher frequency than the $8^{\text {th }}$ and the $9^{\text {th }}$ columns. The offset between the $6^{\text {th }}$ and the $7^{\text {th }}$ (same for the $8^{\text {th }}$ and the $9^{\text {th }}$ ) columns is due to the alternation of sine and cosine functions.

In binary representations, a higher bit has a lower frequency than a lower bit. Similarly, as demonstrated in the heat map below, the positional encoding decreases frequencies along the encoding dimension by using trigonometric functions. Since the outputs are float numbers, such continuous representations are more space-efficient than binary representations.

Besides capturing absolute positional information, the above positional encoding also allows a model to easily learn to attend by relative positions. This is because for any fixed position offset $\delta$, the positional encoding at position $i+\delta$ can be represented by a linear projection of that at position $i$.

This projection can be explained mathematically. Denoting $\omega_j=1 / 10000^{2 j / d}$, any pair of $\left(p_{i, 2 j}, p_{i, 2 j+1}\right)$ in (11.6.2) can be linearly projected to $\left(p_{i+\delta, 2 j}, p_{i+\delta, 2 j+1}\right)$ for any fixed offset $\delta$ :
$$
\begin{aligned}
& {\left[\begin{array}{cc}
\cos \left(\delta \omega_j\right) & \sin \left(\delta \omega_j\right) \\
-\sin \left(\delta \omega_j\right) & \cos \left(\delta \omega_j\right)
\end{array}\right]\left[\begin{array}{c}
p_{i, 2 j} \\
p_{i, 2 j+1}
\end{array}\right] } \\
= & {\left[\begin{array}{c}
\cos \left(\delta \omega_j\right) \sin \left(i \omega_j\right)+\sin \left(\delta \omega_j\right) \cos \left(i \omega_j\right) \\
-\sin \left(\delta \omega_j\right) \sin \left(i \omega_j\right)+\cos \left(\delta \omega_j\right) \cos \left(i \omega_j\right)
\end{array}\right] } \\
= & {\left[\begin{array}{c}
\sin \left((i+\delta) \omega_j\right) \\
\cos \left((i+\delta) \omega_j\right)
\end{array}\right] } \\
= & {\left[\begin{array}{c}
p_{i+\delta, 2 j} \\
p_{i+\delta, 2 j+1}
\end{array}\right] }
\end{aligned}
$$
where the $2 \times 2$ projection matrix does not depend on any position index $i$.

## Notes from Efficient Transformers
(see [[@tayEfficientTransformersSurvey2022]])

The input first passes through an embedding layer that converts each one-hot token representation into a dmodel dimensional embedding, i.e., RB × RN × Rdmodel. The new tensor is then additively composed with positional encodings and passed through a multiheaded self-attention module. Positional encodings can take the form of a sinusoidal input (as per (Vaswani et al., 2017)) or be trainable embeddings.

## Amirhossein Kazemnejad
https://kazemnejad.com/blog/transformer_architecture_positional_encoding

- through breaking up the recurrence, transformers achieve a massive speed-up and can capture longer dependencies in a sentence, but models still require the order of words.
- Assign a number in the range of $[0,1]$ is suboptimal, as a time-step delta does not have a consistent mean across sentences (with different length) .

A positional encoding should be satisfied:
-   It should output a unique encoding for each time-step (word’s position in a sentence)
-   Distance between any two time-steps should be consistent across sentences with different lengths.
-   Our model should generalize to longer sentences without any efforts. Its values should be bounded.
-   It must be deterministic.

- The positional encoding is not part of the model itself. We we enhance the model’s input to inject the order of words.

Let $t$ be the desired position in an input sentence, $\overrightarrow{p_t} \in \mathbb{R}^d$ be its corresponding encoding, and $\boldsymbol{d}$ be the encoding dimension (where $d \equiv{ }_2 0$ ) Then $f: \mathbb{N} \rightarrow \mathbb{R}^d$ will be the function that produces the output vector $\overrightarrow{p_t}$ and it is defined as follows:
$$
{\overrightarrow{p_t}}^{(i)}=f(t)^{(i)}:= \begin{cases}\sin \left(\omega_k \cdot t\right), & \text { if } i=2 k \\ \cos \left(\omega_k . t\right), & \text { if } i=2 k+1\end{cases}
$$
where
$$
\omega_k=\frac{1}{10000^{2 k / d}}
$$
As it can be derived from the function definition, the frequencies are decreasing along the vector dimension. Thus it forms a geometric progression from $2 \pi$ to $10000 \cdot 2 \pi$ on the wavelengths.

You can also imagine the positional embedding $\overrightarrow{p_t}$ as a vector containing pairs of sines and cosines for each frequency (Note that $d$ is divisble by 2 ):
$$
\overrightarrow{p_t}=\left[\begin{array}{c}
\sin \left(\omega_1 \cdot t\right) \\
\cos \left(\omega_1 \cdot t\right) \\
\sin \left(\omega_2 . t\right) \\
\cos \left(\omega_2 \cdot t\right) \\
\vdots \\
\sin \left(\omega_{d / 2} \cdot t\right) \\
\cos \left(\omega_{d / 2} \cdot t\right)
\end{array}\right]_{d \times 1}
$$
**Intuition:**
$$
\begin{align}
  0: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{0}} & & 
  8: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{0}} \\
  1: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{1}} & & 
  9: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{1}} \\ 
  2: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{0}} & & 
  10: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{0}} \\ 
  3: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{1}} & & 
  11: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{1}} \\ 
  4: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{0}} & & 
  12: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{0}} \\
  5: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{1}} & & 
  13: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{1}} \\
  6: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{0}} & & 
  14: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{0}} \\
  7: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{1}} & & 
  15: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{1}} \\
\end{align}
$$
Number in binary format. The LSB bit is alternating on every number, the second-lowest bit is rotating on every two numbers, and so on. BUT, binary encoding would be wasteful and not generalize.

- Another property of sinusoidal position encoding is that the distance between neighboring time-steps are symmetrical and decays nicely with time.
![[dotproduct-pos-embedding.png]]

Reason for using sine and cosine. Personally, I think, only by using both sine and cosine, we can express the sine(x+k) and cosine(x+k) as a linear transformation of sin(x) and cos(x). It seems that you can’t do the same thing with the single sine or cosine. If you can find a linear transformation for a single sine/cosine, please let me know in the comments section.

## Notes from uvadlc
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

- positional information is added as an input feature to the network
- learning an embedding for every possible position is possible, but has difficulties with regard to handling varying input sequence lengths.
[[@vaswaniAttentionAllYou2017]] use:
$$
P E_{(\text {pos }, i)}= \begin{cases}\sin \left(\frac{p o s}{10000^{i / d_{\text {model }}}}\right) & \text { if } i \bmod 2=0 \\ \cos \left(\frac{p o s}{\left.10000^{(i-1) / d_{\text {model }}}\right)}\right. & \text { otherwise }\end{cases}
$$
$\boldsymbol{P E} E_{(p o s, i)}$ represents the position encoding at position $p o s$ in the sequence, and hidden dimensionality $i$. These values, concatenated for all hidden dimensions, are added to the original input features (in the Transformer visualization above, see "Positional encoding"), and constitute the position information. We distinguish between even $(i \bmod 2=0)$ and uneven $(i \bmod 2=1$ ) hidden dimensionalities where we apply a sine/cosine respectively. 
The intuition behind this encoding is that you can represent $P E_{(p o s+k, i)}$ as a linear function of $P E_{(p o s,:)}$, which might allow the model to easily attend to relative positions. The wavelengths in different dimensions range from $2 \pi$ to $10000 \cdot 2 \pi$.

## Notes from Tunstall
[[@tunstallNaturalLanguageProcessing2022]]R

“Naturally, we want to avoid being so wasteful with our model parameters since models are expensive to train, and larger models are more difficult to maintain. A common approach is to limit the vocabulary and discard rare words by considering, say, the 100,000 most common words in the corpus. Words that are not part of the vocabulary are classified as “unknown” and mapped to a shared UNK token. This means that we lose some potentially important information in the process of word tokenization, since the model has no information about words associated with UNK.” ([Tunstall, 2022, p. 32](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=56&annotation=8I4JJSUZ))

“Positional embeddings are based on a simple, yet very effective idea: augment the token embeddings with a position-dependent pattern of values arranged in a vector. If the pattern is characteristic for each position, the attention heads and feed-forward layers in each stack can learn to incorporate positional information into their transformations.” ([Tunstall, 2022, p. 73](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=97&annotation=SH9B5BKC))

“Absolute positional representations Transformer models can use static patterns consisting of modulated sine and cosine signals to encode the positions of the tokens. This works especially well when there are not large volumes of data available.” ([Tunstall, 2022, p. 74](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=98&annotation=WYXPF9R8))

“Relative positional representations Although absolute positions are important, one can argue that when computing an embedding, the surrounding tokens are most important. Relative positional representations follow that intuition and encode the relative positions between tokens. This cannot be set up by just introducing a new relative embedding layer at the beginning, since the relative embedding changes for each token depending on where from the sequence we are attending to it” ([Tunstall, 2022, p. 74](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=98&annotation=P3WC3ZNQ))

## Notes from e2eml school
https://e2eml.school/transformers.html#positional_encoding
- Without a positional embedding words the distance of words from each other in the embedding space would be independent from the order in which they appear in the sentence. A **perturbation** is added to the word embedding to account for the word order in the sequence.  
- The original word embedding is the center of a circle and a pertubation is added depending on its position in the input space. For each position, the word is moved the same distance but at a different angle, resulting in a circular pattern as you move through the sequence. Words that are close to each other in the sequence have similar perturbations, but words that are far apart are perturbed in different directions.
- The combination of all these circular wiggles of different frequencies gives a good representation of the absolute position of a word within the sequence.


## Notes from Jonathan Kernes
https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3

Why not use an absolute encoding?
- Count through: values become really large. Weights for neural networks should be centered and balanced around zero. (See [[🧠Deep Learning Methods/@lecunEfficientBackProp2012]])
- Count through and normalize: can no langer handle arbitrary sequence lengths
- binary encoding (encoding numbers as binary number) would result in a matrix. Can not handle large numbers which would require more places than columns in matrix. Could be centered around zero with $f(x)=2x-1$
- We try to find an embedding manifold. -> A curve in the $d_{\text{model}}$ embedding space, that increases position in a continous way.
- $\sin$ function exists on $[-1,1]$ and is normalized.
- Think of the encodings in different columns as dials. Each dial gives finegrained or coarse grained control over levels. E. g. one finetuning, another one more coarse-grained etc.
- Note that the frequency is different for every column in the embedding matrix (e. g., the degree a dial would change result). 
- We now have our first prototype positional encoding tensor. It consists of a matrix denoted by $\mathbf{M}$, where the $y$-axis of the matrix is the discrete position $x_{-} i$ in the sequence $(0,1, \ldots, \mathrm{n}-1)$, and the $\mathrm{x}$-axis is the vector embedding dimension. Elements of the matrix $\mathbf{M}$ are given by:
$$
M_{i j}=\sin \left(2 \pi i / 2^j\right)=\sin \left(x_i \omega_j\right)
$$
![[absolute-position-positional-encoding.png]]
- It's problematic that the curve is closed. e. g., $n+1$ is nearby to $1$. To address this issue. 
increase. We can lower all of the frequencies, and in this way keep ourselves far away from the boundary.
$$
M_{i j}=\sin \left(x_i \omega_0^{j / d_{\text {model }}}\right)
$$
The slowed down dial version. $omega_0$ is the min frequency. Check the edge cases: $j=0$ gives the largest frequency, $j=d_model$ gives the smallest.
The original authors choose a minimum frequency of $1 / 10,000$ for this purpose. they then increase frequencies exponentially via $omega_min ${ range \left(\mathrm{d}\right) / \mathrm{d}}$ to achieve the monotonicity criteria. 

- We want to line up the position vector with the position vector of the input, so that a query and key matches them perfectly. -> Modify encoding so that it can be translated with a linear transformation. The following should hold:
$$
\operatorname{PE}(x+\Delta x)=\operatorname{PE}(x) \cdot \mathbf{T}(\Delta x)
$$
$\mathbf{T}(\Delta x)$ is a linear transformation and $\operatorname{PE}(x)$ the positional encoding.

$[\cos (\ldots) \sin (\ldots)]$ pair. In our new PE matrix, the row-vectors look like the following:
$$
\mathbf{v}^{(i)}=\left[\cos \left(\omega_0 x_i\right), \sin \left(\omega_0 x_i\right), \ldots, \cos \left(\omega_{n-1} x_i\right), \sin \left(\omega_{n-1} x_i\right)\right]
$$
We now build the full linear transformation by using a bunch of blockdiagonal linear transformations. Each block will have a different matrix, since the frequencies that the block acts on are different. For example, in order to translate the $k$ th dial with frequency omega_k by dx units, we would need a total angle shift of delta=omega_ $k^* d x$. The $T$ matrix can now be written as:
![[block-matrix-pe.png]]
If you take the transpose of this, you can directly insert it into our previous $\mathrm{PE}(\mathrm{x}+\mathrm{dx})=\mathrm{PE}(\mathrm{x})^* \mathrm{~T}$ equation, thereby proving by construction the existence of a translation matrix!

- The row-vector is an alternating series of sines and cosines, with frequencies that decrease according to a geometric series
- There exists a matrix multiplication that can shift the position of any row-vector we want.

## Notes by Timo Denk
https://timodenk.com/blog/linear-relationships-in-the-transformers-positional-encoding/
## Problem Statement

Let E∈Rn×dmodel be a matrix that contains dmodel-dimensional column vectors Et,: which encode the position t in an input sequence of length n. The function e:{1,…,n}→Rdmodel induces this matrix and is defined as  

(1)e(t)=Et,::=[sin⁡(tf1)cos⁡(tf1)sin⁡(tf2)cos⁡(tf2)⋮sin⁡(tfdmodel2)cos⁡(tfdmodel2)],

where the frequencies are  

(2)fm=1λm:=100002mdmodel.

The paper states that a linear transformation T(k)∈Rdmodel×dmodel exists, for which  

(3)T(k)Et,:=Et+k,:

holds for any positional offset k∈{1,…,n} at any valid position t∈{1,…,n−k} in the sequence.

## Derivation

That is true, because a definition for T(k) can be found with no dependence on t:  

(4)T(k)=[Φ1(k)0⋯00Φ2(k)⋯000⋱000⋯Φdmodel2(k)],

where 0 denotes 2×2 all-zero matrices and the #dmodel2 transposed [rotation matrices](https://en.wikipedia.org/wiki/Rotation_matrix) Φ(k) positioned on the main diagonal are defined by  

(5)Φm(k)=[cos⁡(rmk)−sin⁡(rmk)sin⁡(rmk)cos⁡(rmk)]⊺,

with wave length rm (not to be confused with the encoding wave length λm).

Now we can reformulate Eq. 3, which we want to prove, as  

(6)[cos⁡(rmk)sin⁡(rmk)−sin⁡(rmk)cos⁡(rmk)]⏟Φm(k)[sin⁡(λmt)cos⁡(λmt)]=[sin⁡(λm(t+k))cos⁡(λm(t+k))].

Expanded, that is  

(6a)sin⁡(λk+λt)=sin⁡(rk)cos⁡(λt)+cos⁡(rk)sin⁡(λt)(6b)cos⁡(λk+λt)=cos⁡(rk)cos⁡(λt)−sin⁡(rk)sin⁡(λt),

and we seek to determine r in dependence of λ and k, while eliminating t.

The [addition theorems](https://timodenk.com/blog/trigonometric-functions-formulary/)

(7a)sin⁡(α+β)=sin⁡αcos⁡β+cos⁡αsin⁡β(7b)cos⁡(α+β)=cos⁡αcos⁡β–sin⁡αsin⁡β

help solving the problem at hand. When applying the addition theorems to the expanded form (Eq. 6a, 6b), it follows

(8a)α=λk=rk(8b)β=λt=λt,

and consequently r=λ. Applying that finding and the definition of λm (Eq. 2) to Eq. 5, we get  

(9)Φm(k)=[cos⁡(λmk)sin⁡(λmk)−sin⁡(λmk)cos⁡(λmk)],

where λm=10000−2mdmodel. With that, T(k) fully specified and depends solely on m, dmodel, and k. The position within the sequence, t, is not a parameter, q.e.d.


From [tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L403-L449):
```python
def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4, start_index=0): """Gets a bunch of sinusoids of different frequencies. Each channel of the input Tensor is incremented by a sinusoid of a different frequency and phase. This allows attention to learn to use absolute and relative positions. Timing signals should be added to some precursors of both the query and the memory inputs to attention. The use of relative position is possible because sin(x+y) and cos(x+y) can be expressed in terms of y, sin(x) and cos(x). In particular, we use a geometric sequence of timescales starting with min_timescale and ending with max_timescale. The number of different timescales is equal to channels / 2. For each timescale, we generate the two sinusoidal signals sin(timestep/timescale) and cos(timestep/timescale). All of these sinusoids are concatenated in the channels dimension. Args: length: scalar, length of timing signal sequence. channels: scalar, size of timing embeddings to create. The number of different timescales is equal to channels / 2. min_timescale: a float max_timescale: a float start_index: index of first position Returns: a Tensor of timing signals [1, length, channels] """ position = tf.to_float(tf.range(length) + start_index) num_timescales = channels // 2 log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / tf.maximum(tf.to_float(num_timescales) - 1, 1)) inv_timescales = min_timescale * tf.exp( tf.to_float(tf.range(num_timescales)) * -log_timescale_increment) scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0) signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1) signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]]) signal = tf.reshape(signal, [1, length, channels]) return signal
```