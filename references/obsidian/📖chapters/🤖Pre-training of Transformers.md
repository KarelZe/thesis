
**Why pretraining?**
Whilst Transformers could be combined with self-training, a more common approach is to pre-train Transformers on unlabelled data, and then finetune the Transformer on the remaining labelled data.

“An important advantage of deep models over GBDT is that they can potentially achieve higher performance via pretraining their parameters with a properly designed objective. These pretrained parameters, then, serve as a better than random initialization for subsequent finetuning for downstream tasks. For computer vision and NLP domains, pretraining is a de facto standard and is shown to be necessary for the state-of-the-art performance [18, 10].” (Rubachev et al., 2022, p. 1)

In several instances, pre-training has shown to improve improve performance

Pre-training accelerates convergence and improves model performance through a better weight initialization over a random initialization. 
(successfully adapted -> Tabnet, TabTransformer, Saint ) (easy to optimize, leave architecture unaltered)

“Pretraining provides substantial gains over well-tuned supervised baselines in the fully supervised setup.” (Rubachev et al., 2022, p. 2)

performance gains over Transformers trained on labelled data only.

We base our selection on ([[@rubachevRevisitingPretrainingObjectives2022]]), who convincingly benchmark several pre-training objectives for tabular data. Among the best-performing approaches is the gls-*masked language modelling* objective proposed by ([[@devlinBERTPretrainingDeep2019]] 4174). 

(not target aware)

**How masked language modelling works**
Originally proposed for pre-training a language model, masked language modelling gls-(mlm) randomly masks 15 sunitx-percentage of the tokens, i. e., word, in the input sequence and replaces them with a $\mathtt{[MASK]}$ token. The $\mathtt{[MASK]}$ token, is a specialized token within the vocabulary  (cp. [[💤Embeddings For Tabular Data]]). The model learns to predict the masked token through the tokens in the bidirectional context. Like before, the cross-entropy loss is used to predict the most probable token ([[@devlinBERTPretrainingDeep2019]] 4174). 

A caveat of *mlm* is, that the mask token only appears during pre-training, but not in fine-tuning, creating a mismatch between the pre-training and fine-tuning task. (Devlin offer some alternatives)
(framed as multi-classification task)

**How it transfers to my problem**
Applied to trade classification on tabular datasets, masked language modelling transfers to randomly masking elements in $\mathbf{x}_{i}$. () The classification task is trivialized for derived features or features with a high dependence on other features in the context. In a broader sense, gls-mlm can help learning expressive representations of the input, even if the true label is unavailable. Given previous research, we expect pre-training to improve the performance of the FT-Transformer and match the one of the gls-gbm.

**Notes:**
[[🤖Pretraining notes]]


