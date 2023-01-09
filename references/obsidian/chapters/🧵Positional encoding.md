Related: 
#transformer #positional-encoding #linear-algebra

**ressources:**
- https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
- introduced in [[@vaswaniAttentionAllYou2017]]
- nice consistent notation in [[@phuongFormalAlgorithmsTransformers2022]]

**Notes:**
- Add information about the position in sequence to the embedding
- The vector $p_i$ is of dimension $d_{\text{model}}$  and is added (not concat'd! -> to keep no of params small?) to the embedding vector.
- Note that there is a subtle difference between positional embedding and encoding. When reading [[@rothmanTransformersNaturalLanguage2021]] I noticed, that word `positional encoding = embedding + positional vector`
- There are obviously different types of *positional embeddings* e. g., learnable embeddings, absolute positional representations and relative positional representations. ([[@tunstallNaturalLanguageProcessing2022]]) In the [[@vaswaniAttentionAllYou2017]] an *absolute positional encoding* is used. Sine and cosine signals are sued to encode the position of tokens. 
- Positional encoding can be learned or fix. Both seem to work similarily. 
- According to this [reddit post](https://www.reddit.com/r/learnmachinelearning/comments/9e4j4q/positional_encoding_in_transformer_model/), [[@vaswaniAttentionAllYou2017]] chose the fixed encoding with sine and cosine, to enable learning about the relative position in a sequence.
- *Absolute positional embeddings* work well if the dataset is small.

**Visualization:**
![[viz-pos-encoding.png]]
(Code available at https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb)

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