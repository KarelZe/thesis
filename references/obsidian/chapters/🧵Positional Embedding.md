related: 
#transformer #positional-encoding #linear-algebra #linear-projection #cylic-data

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

**Notes:**
[[🧵Positional encoding notes]]