*title:* On Layer Normalization in the Transformer Architecture
*authors:* Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu
*year:* 2020
*tags:* #transformer #layernorm #post-norm #pre-norm
*status:* #📦 
*related:*
- [[@liuUnderstandingDifficultyTraining2020]]
- [[@wangLearningDeepTransformer2019]]
*code:*
*review:*

## Notes 📍

## Annotations 📖

“To train a Transformer however, one usually needs a carefully designed learning rate warm-up stage, which is shown to be crucial to the final performance but will slow down the optimization and bring more hyperparameter tunings. In this paper, we first study theoretically why the learning rate warm-up stage is essential and show that the location of layer normalization matters. Specifically, we prove with mean field theory that at initialization, for the original-designed Post-LN Transformer, which places the layer normalization between the residual blocks, the expected gradients of the parameters near the output layer are large. Therefore, using a large learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful for avoiding this problem. On the other hand, our theory also shows that if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization. This motivates us to remove the warm-up stage for the training of Pre-LN Transformers. We show in our experiments that Pre-LN Transformers without the warm-up stage can reach comparable results with baselines while requiring significantly less training time and hyper-parameter tuning on a wide range of applications.” ([Xiong et al., 2020, p. 1](zotero://select/library/items/JKKHGAAC)) ([pdf](zotero://open-pdf/library/items/2E5NZTRP?page=1&annotation=WV34HEM3))

“Residual connection and layer normalization Besides the two sub-layers described above, the residual connection and layer normalization are also key components to the Transformer. For any vector v, the layer normalization is computed as LayerNorm(v) = γ v−μ σ + β, in which μ, σ are the mean and standard deviation of the elements in v, i.e., μ = 1 d ∑d k=1 vk and σ2 = 1 d ∑d k=1(vk − μ)2. Scale γ and bias vector β are parameters” ([Xiong et al., 2020, p. 3](zotero://select/library/items/JKKHGAAC)) ([pdf](zotero://open-pdf/library/items/2E5NZTRP?page=3&annotation=7FI3EMJF))

“Different orders of the sub-layers, residual connection and layer normalization in a Transformer layer lead to variants of Transformer architectures. One of the original and most popularly used architecture for the Transformer and BERT (Vaswani et al., 2017; Devlin et al., 2018) follows “selfattention (FFN) sub-layer → residual connection → layer normalization”, which we call the Transformer with PostLayer normalization (Post-LN Transformer), as illustrated in Figure 1.” ([Xiong et al., 2020, p. 3](zotero://select/library/items/JKKHGAAC)) ([pdf](zotero://open-pdf/library/items/2E5NZTRP?page=3&annotation=NNLEUS79))