[[🅰️attention notes]]

## Agenda
- What attention is?
	- Where was it introduced?
	- Explain attention in on *catchy* sentence.
	- Why it is key to the transformer?
- What is self-attention?
	- How is it different from the standard attention?
	- Why was it proposed in the first place?
- What is multi-headed attention?
	- Why do we decompose?
	- How do different heads attend to different characteristics?
	- What is the importance of different heads?
	- Why can some heads be pruned?
- What is cross-attention?
	- Why is it needed to mask attention in the decoder?
	- What is the appropriate masking strategy?


While not explored systematically for the tabular domain, the roles of different attention heads have been studied intensively in transformer-based machine translation (see e. g., [[@voitaAnalyzingMultiHeadSelfAttention2019]], [[@tangAnalysisAttentionMechanisms2018]]).  [[@voitaAnalyzingMultiHeadSelfAttention2019]] observe that attention heads have varying importance and serve distinct purposes like learning positional or syntactic relations between tokens. Also, all attention layers contribute to the model's prediction. 


Attention is a concept originally introduced in [[@bahdanauNeuralMachineTranslation2016]]


In practice,  the , Thus, the d-model dimensional keys is distributed logically. 


As the output 

Finally, a softmax function is applied to 

**Multi-headed attention:**
Rather than relying on a single attention function, [[@vaswaniAttentionAllYou2017]] (4. f) introduce multiple so-called *attention heads*, which perform attention in parallel on different linear projections of the queries, keys, and values. The 

Doing so gives the model the flexibility


$$
\tilde{V}=\boldsymbol{V} \cdot \operatorname{softmax}\left(\boldsymbol{S} / \sqrt{d_{\mathrm{attn}}}\right)
$$

The output of the $h$ individual attention heads is finally concatenated and passed into a dense layer, whose purpose is to linearly project the output back into

$$
\begin{aligned}
Y &= \left[Y^1 ; Y^2 ; \ldots ; Y^H\right] \\
\tilde{V}&=W_o Y+b_o 1^{\top}
\end{aligned}
$$
In practice, both the split across attention heads and the concatenation of the head's result is done only logically with each of the attention heads operating on the same data matrix, but in different subsections. The output $\tilde{V}$ is then passed to 
