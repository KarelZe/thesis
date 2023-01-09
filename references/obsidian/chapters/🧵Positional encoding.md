
- Only describe shortly. Positional encoding is not important for tabular data, as the ordering of features is invariant.
- Both architectures (TabTransformer and FTTransformer) don't include one.

- Add information about the position in sequence to the embedding
- The vector $p_i$ is of dimension $d_{\text{model}}$  and is added (not concat'd! -> to keep no of params small?) to the embedding vector.
- Note that there is a subtle difference between positional embedding and encoding. When reading [[@rothmanTransformersNaturalLanguage2021]] I noticed, that word `positional encoding = embedding + positional vector`
- There are obviously different types of *positional embeddings* e. g., learnable embeddings, absolute positional representations and relative positional representations. ([[@tunstallNaturalLanguageProcessing2022]]) In the [[@vaswaniAttentionAllYou2017]] an *absolute positional encoding* is used. Sine and cosine signals are sued to encode the position of tokens. 
- *Absolute positional embeddings* work well if the dataset is small.


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

Code to plot positional encoding:
```python
#### TensorFlow only version ####

def positional_encoding(max_position, d_model, min_freq=1e-4):

position = tf.range(max_position, dtype=tf.float32)

mask = tf.range(d_model)

sin_mask = tf.cast(mask%2, tf.float32)

cos_mask = 1-sin_mask

exponent = 2*(mask//2)

exponent = tf.cast(exponent, tf.float32)/tf.cast(d_model, tf.float32)

freqs = min_freq**exponent

angles = tf.einsum('i,j->ij', position, freqs)

pos_enc = tf.math.cos(angles)*cos_mask + tf.math.sin(angles)*sin_mask

return pos_enc

#### Numpy version ####

def positional_encoding(max_position, d_model, min_freq=1e-4):

position = np.arange(max_position)

freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)

pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)

pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])

pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])

return pos_enc

### Plotting ####

d_model = 128

max_pos = 256

mat = positional_encoding(max_pos, d_model)

plt.pcolormesh(mat, cmap='copper')

plt.xlabel('Depth')

plt.xlim((0, d_model))

plt.ylabel('Position')

plt.title("PE matrix heat map")

plt.colorbar()

plt.show()
```