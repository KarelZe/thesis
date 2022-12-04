*title:* Formal Algorithms for Transformers
*authors:* Mary Phuong, Marcus Hutter
*year:* 2021
*tags:* #transformer #deep-learning #attention #encoder #embeddings #multiheaded-attention #sgd #adam #tokenization 
*status:* #📦 
*related:*
- [[@vaswaniAttentionAllYou2017]]
- [[@huangTabTransformerTabularData2020]]
- [[@huangTabTransformerTabularData2020]]
- [[@devlinBERTPretrainingDeep2019]]
# Notes 

# Annotations

“It covers what transformers are, how they are trained, what they are used for, their key architectural components, and a preview of the most prominent models. The reader is assumed to be familiar with basic ML terminology and simpler neural network architectures such as MLPs.” ([Phuong and Hutter, 2022, p. 1](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=1&annotation=77QFWC4I))

“It aims to be a self-contained, complete, precise and compact overview of transformer architectures and formal algorithms (but not results).” ([Phuong and Hutter, 2022, p. 1](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=1&annotation=8C5HP6PY))

“n practice, the distribution estimate is often decomposed via the chain rule as ˆ 𝑃¹𝒙º = ˆ 𝑃𝜽 ¹𝑥 »1¼º ˆ 𝑃𝜽¹𝑥 »2¼ j 𝑥 »1¼º ˆ 𝑃𝜽 ¹𝑥 »¼ j 𝒙 »1 : 1¼º, where 𝜽 consists of all neural network parameters to be learned. The goal is to learn a distribution over a single token 𝑥 »𝑡¼ given its preceding tokens 𝑥 »1 : 𝑡 1¼ as context.” ([Phuong and Hutter, 2022, p. 3](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=3&annotation=7H4SEVMT))

“Given a vocabulary 𝑉 and a set of classes »𝑁C¼, let ¹𝒙𝑛 𝑐𝑛º 2 𝑉 »𝑁C¼ for 𝑛 2 »𝑁data¼ be an i.i.d. dataset of sequence-class pairs sampled from 𝑃¹𝒙 𝑐º. The goal in classification is to learn an estimate of the conditional distribution 𝑃¹𝑐j𝒙º.” ([Phuong and Hutter, 2022, p. 4](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=4&annotation=46X5JAU9))

“here are in fact many ways to do subword tokenization. One of the simplest and most successful ones is Byte Pair Encoding [Gag94, SHB16] used in GPT-2 [RWC ̧19].” ([Phuong and Hutter, 2022, p. 4](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=4&annotation=57PTDTJ3))

“Token embedding. The token embedding learns to represent each vocabulary element as a vector in ℝ𝑑e; see Algorithm 1” ([Phuong and Hutter, 2022, p. 4](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=4&annotation=UC6DXFP7))

“The positional embedding learns to represent a token’s position in a sequence as a vector in ℝ𝑑e. For example, the position of the first token in a sequence is represented by a (learned) vector 𝑾𝒑 »: 1¼, the position of the second token is represented by another (learned) vector 𝑾𝒑 »: 2¼, etc. The purpose of the positional embedding is to allow a Transformer to make sense of word ordering; in its absence the representation would be permutation invariant and the model would perceive sequences as “bags of words” instead.” ([Phuong and Hutter, 2022, p. 5](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=5&annotation=Q2WE7H5S))

“The positional embedding of a token is usually added to the token embedding to form a token’s initial embedding. For the 𝑡-th token of a sequence 𝒙, the embedding is 𝒆 = 𝑾𝒆 »: 𝑥 »𝑡¼¼ ̧ 𝑾𝒑 »: 𝑡¼” ([Phuong and Hutter, 2022, p. 5](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=5&annotation=DKMMC563))

“Attention. Attention is the main architectural component of transformers. It enables a neural Algorithm 2: Positional embedding. Input: 2 »max¼, position of a token in the sequence. Output: 𝒆𝒑 2 ℝ𝑑e, the vector representation of the position. Parameters: 𝑾𝒑 2 ℝ𝑑emax , the positional embedding matrix. 1 return 𝒆𝒑 = 𝑾𝒑 »: ¼ network to make use of contextual information (e.g. preceding text or the surrounding text) for predicting the current token.” ([Phuong and Hutter, 2022, p. 5](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=5&annotation=B84PMF5K))

“On a high level, attention works as follows: the token currently being predicted is mapped to a query vector 𝒒 2 ℝ𝑑attn, and the tokens in the context are mapped to key vectors 𝒌𝑡 2 ℝ𝑑attn and value vectors 𝒗𝑡 2 ℝ𝑑value . The inner products 𝒒ᵀ𝒌𝑡 are interpreted as the degree to which token 𝑡 2 𝑉 is important for predicting the current token 𝑞 – they are used to derive a distribution over the context tokens, which is then used to combine the value vectors. An intuitive explanation how this achieves attention can be found at The precise algorithm is given in Algorithm 3” ([Phuong and Hutter, 2022, p. 5](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=5&annotation=7SA5TCXB))

“he attention algorithm presented so far (Algorithm 4) describes the operation of a single attention head. In practice, transformers run multiple attention heads (with separate learnable parameters) in parallel and combine their outputs; this is called multi-head attention; see Algorithm 5” ([Phuong and Hutter, 2022, p. 6](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=6&annotation=GIP6PKMQ))

“Layer normalisation explicitly controls the mean and variance of individual neural network activations; the pseudocode is given in Algorithm 6” ([Phuong and Hutter, 2022, p. 7](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=7&annotation=8B7ERQGB))

“It uses the GELU nonlinearity instead of ReLU: GELU¹𝑥º = 𝑥 ℙ𝑋N ¹01º »𝑋 𝑥¼ (5) (When called with vector or matrix arguments, GELU is applied element-wise.)” ([Phuong and Hutter, 2022, p. 8](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=8&annotation=A4VRNZJQ))

“EDTraining() Algorithm 11 shows how to train a sequence-to-sequence transformer (the original Transformer .” ([Phuong and Hutter, 2022, p. 8](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=8&annotation=AZPM2ABG))

“Gradient descent. The described training Algorithms 11 to 13 use Stochastic Gradient Descent” ([Phuong and Hutter, 2022, p. 8](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=8&annotation=4CQKYJTC))

“Formal Algorithms for Transformers (SGD) 𝜽 𝜽 𝜂 rloss¹𝜽º to minimize the log loss (aka cross entropy) as the update rule. Computation of the gradient is done via automatic differentiation tools; see  In practice, vanilla SGD is usually replaced by some more refined variation such as RMSProp or AdaGrad or others . Adam  is used most often these days.” ([Phuong and Hutter, 2022, p. 9](zotero://select/library/items/DYN5Q8UB)) ([pdf](zotero://open-pdf/library/items/9X32MT2H?page=9&annotation=NHTHE4DY))