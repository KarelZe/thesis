In practice, the order of words is important for the overall meaning of a sentence. As such, [[@vaswaniAttentionAllYou2017]] (p. 6) proposes to inject information on the token's position within the sequence through a *positional embedding*, that is added to the token embedding. Like token embeddings, positional embeddings can also be learned (cp. [[@devlinBERTPretrainingDeep2019]]). Due to better, extrapolation capabilities, [[@vaswaniAttentionAllYou2017]] (p. 6), propose an embedding $W_p: \mathbb{N} \rightarrow \mathbb{R}^{d_{\mathrm{e}}}$ based on sine and cosine signals to encode the *absolute* position of the token:
$$
\tag{1}
\begin{aligned}
W_p[2 i-1, t] & =\sin \left(t / \ell_{\max }^{2 i / d_e}\right), \\
W_p[2 i, t] & =\cos \left(t / \ell_{\max }^{2 i / d_e}\right).
\end{aligned}
$$
with $0<i \leq d_{\mathrm{e}} / 2$, the maximum sequence length $\ell_{\max}$, which is arbitrarily set to $\ell_{\max}=10{,}000$, and $t$ is again the position of the token in the sequence. As shown in Equation (1) the frequency decreases across the position dimension and alternates between sine and cosine for the embedding dimension. Each embedding thus contains a pattern, easily distinguishable by the model. We visualize the positional encoding in Figure [[#^1f2fe5]] with an embedding dimension of $d_e=96$ and 64 tokens. One can see the alternating pattern between even and odd columns and the unique pattern for each token's position. 

![[positional-encoding.png]]
<mark style="background: #FFB8EBA6;">(Similar to [[@zhangDiveDeepLearning2021]] (p. 409); check y-labels. I thought that patterns fade for the latter positions in the sequence. See also here: https://www.borealisai.com/research-blogs/tutorial-16-transformers-ii-extensions/)</mark> ^1f2fe5

Using trigonometric functions for the positional embedding is favourable, due to being zero-centred, and resulting in values in the closed range of $[-1,1]$. These properties are long known to promote the convergence of neural networks (cp. [[@lecunEfficientBackProp2012 1]] (p. 8 f) or [[📥Inbox/@ioffeBatchNormalizationAccelerating2015]] (p. 2)). 

The reason for encoding with both the sine and cosine is more subtle, as either one would suffice for absolute embeddings. [[@vaswaniAttentionAllYou2017]] (p. 6) hypothesize, that besides learning the *absolute position* i. e., fifth place in sequence, providing both sine and cosine also enables the model to attend to *relative positions*, i. e., two places from a given token. The detailed proof is given in  [[@zhangDiveDeepLearning2021]] (p. 410).

The positional embedding is finally added element-wisely to the token embedding to form a token's initial embedding $e$. For the $t$-th token of a sequence $x$, the embedding becomes:
$$
\tag{3}
e=W_e[:, x[t]]+W_p[:, t] .
$$

Intuitionally, adding the positional encoding leads to a rotation of the token embedding in the embedding space. As the positional embedding is different for every location within the sequence, otherwise identical tokens, now have a unique embedding. Positional embeddings are not the only way to fix the location, however. Later works, like [[@daiTransformerXLAttentiveLanguage2019]] (p. 4 f.), remove the positional encoding in favour of a *relative position encoding*, which is only considered during computation.

**Notes:**
[[🧵Positional encoding notes]]