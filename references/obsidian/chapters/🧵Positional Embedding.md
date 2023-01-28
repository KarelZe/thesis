<mark style="background: #FFB8EBA6;">(TODO: make clear, why it is needed in the first place)</mark>. [[@vaswaniAttentionAllYou2017]] (p. 6) proposes to inject information on the token's position within the sequence through a *positional embedding*, that is added to the token embedding. Like token embeddings, positional embeddings can also be learned. Due to better, extrapolation capabilities, [[@vaswaniAttentionAllYou2017]] (p. 6), propose an embedding $W_p: \mathbb{N} \rightarrow \mathbb{R}^{d_{\mathrm{e}}}$ based on sine and cosine signals to encode the *absolute* position of the token:
$$
\tag{1}
\begin{aligned}
W_p[2 i-1, t] & =\sin \left(t / \ell_{\max }^{2 i / d_e}\right), \\
W_p[2 i, t] & =\cos \left(t / \ell_{\max }^{2 i / d_e}\right).
\end{aligned}
$$
with $0<i \leq d_{\mathrm{e}} / 2$, the maximum sequence length $\ell_{\max}$, which is arbitrarily set to $\ell_{\max}=10,000$, and $t$ is again the position of the token in the sequence. As shown in Equation (1) the frequency decreases across the position dimension and alternates between sine and cosine for the embedding dimension. Each embedding thus contains a pattern, easily distinguishable by the model.

Using trigonometric functions for the positional embedding is favorable, due to being zero-centered, and resulting in values in the *limited* range of $[-1,1]$. These properties are long known to promote the convergence of neural networks (cp. [[@lecunEfficientBackProp2012]] (p. 8 f)). The reason for encoding with both the sine and cosine is more subtle. [[@vaswaniAttentionAllYou2017]] (p. 6) hypothesize, that besides learning the *absolute position* i. e., fifth place in sequence, also enables the model to attend to *relative positions*, i. e., two places from a given token. A detailed proof is laid out in  [[@zhangDiveDeepLearning2021]] (p. 410) 

![[viz-of-pos-encoding.png]]
(pos encoding copied from [[@zhangDiveDeepLearning2021]]; p. 409; transpose image due to my matrix notation) ^1f2fe5
We visualize the positional embedding in Figure [[#^1f2fe5]] with an embedding dimension of $d_e=32$ and 60 tokens. One can clearly see the pattern describing the position.

The positional embedding of a token is finally added to the token embedding to form a token's initial embedding $e$. For the $t$-th token of a sequence $x$, the embedding becomes:
$$
e=W_e[:, x[t]]+W_p[:, t] .
$$
Due to the sheer depth of the network with multiple Transformer blocks, the positional information would easily vanish during back-propagation. To enforce it, the architecture relies on residual connections [[@heDeepResidualLearning2015]] (p. 3).
<mark style="background: #FFB8EBA6;">(TODO: What is the intuition behind adding a positional encoding -> shift in space (see talk of lucas beyer / rothman) )</mark>
Later works, like [[@daiTransformerXLAttentiveLanguage2019]] (p. 4 f.), remove the positional encoding in favour of a *relative position encoding*, which is only considered during computation.

**Notes:**
[[🧵Positional encoding notes]]