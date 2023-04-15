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



2.1.1 Self-attention (Vaswani et al., 2017) introduced the self-attention mechanism. This mechanism is able to retrieve the contextual information of a word in a sentence. The module compares every word in the sentence to every other word in the same sentence, including itself. During this comparison, the word embeddings of all words in the sentence are reweighed to include the contextual relevance. The logic behind comparing the word with itself is to determine the exact meaning 4 Figure 2.1: The calculated attention for a sentence. Thicker lines indicate that more attention should be paid to that word.a aRetrieved from a presentation by (Vaswani et al., 2017): https://www.slideshare.net/ilblackdragon/attention-is-all-you-need of the word, since a single word can have different meanings, depending on the context of the word, i.e. I am going to the bank to retrieve money, and the bank of a river. Figure 2.1 contains calculated attention scores for a sentence. The attentions scores are represented by the lines. How thicker a line, the more attention should be paid to this word. For a model to be capable of learning the contextual connections between words, three weight matrices are introduced, called the query weight matrix, key weight matrix and value weight matrix. These weight matrices are learned during training, and using matrices enables simultaneous calculations for the whole sequence. The input word embeddings are all multiplied with each of the weight matrices separately, to get three new matrices, the query, key and value matrix. Equations 2.1 - 2.3 show these multiplications, whereas X is the word embedding matrix of the input, WQ is the query weight matrix, W K is the key weight matrix, WV is the value weight matrix. Q = X × WQ (2.1) K = X × W K (2.2) V = X × WV (2.3) The next step is to deduce to which words the Transformer should focus on, pay attention to, for a specific word. Ideally, all these words refer to this specific word. Calculating the dot product similarity is the first step to calculate the attention scores. The result of the dot product similarity indicates to which words 5 the Transformer should pay attention to. A higher score indicates that more attention should be paid to this specific word. These products are calculated by multiplying the query matrix with the transpose of the key matrix. After calculating all these dot products, the values are normalized with the softmax function, to focus on the relevant words, and to drown-out irrelevant words. Equation 2.4 shows the equation used to calculate these values, where dk is the dimension of key vectors. Dividing by this dimension leads to more stable gradients. Attention(Q, K) = sof tmax( Q × KT √ dk ) (2.4) Finally, to calculate the new word embeddings, the context vector, the previously calculated attention matrix is multiplied by the originally calculated value matrix. The result of this final multiplication, Equation 2.5, are the newly weighted word embeddings, Z. Z = Attention(Q, K) × V (2.5) Finally, one of these self-attention blocks might not be capable of paying attention to several important words and might not make an observable change to their respective word embeddings. To resolve this problem, multi-headed attention is introduced. Multi-head attention exists out of several attention blocks which run in parallel, each calculating attention scores independently. Therefore, using this mechanism expands the model’s ability to focus on different positions. The three weight matrices are not shared between the different selfattention blocks, and are all randomly initialized. The results of these selfattention blocks are concatenated together to retrieve the new word embeddings, the new context vectors.
(https://ai4lt.anthropomatik.kit.edu/downloads/Masterarbeiten/2022/Master_Thesis_Tim_Debets_Final.pdf)



While not explored systematically for the tabular domain, the roles of different attention heads have been studied intensively in transformer-based machine translation (see e. g., [[@voitaAnalyzingMultiHeadSelfAttention2019]], [[@tangAnalysisAttentionMechanisms2018]]).  [[@voitaAnalyzingMultiHeadSelfAttention2019]] observe that attention heads have varying importance and serve distinct purposes like learning positional or syntactic relations between tokens. Also, all attention layers contribute to the model's prediction. 


Heads have a different importance and many can even be pruned: [[@michelAreSixteenHeads2019]]

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
