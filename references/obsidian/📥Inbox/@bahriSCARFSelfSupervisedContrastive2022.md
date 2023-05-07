*title:* SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption
*authors:* Dara Bahri, Heinrich Jiang, Yi Tay, Donald Metzler
*year:* 2022
*tags:* #semi-supervised #self-supervised #deep-learning 
*status:* #📥
*related:*
- [[@yoonVIMEExtendingSuccess2020]]
- [[@devlinBERTPretrainingDeep2019]]
- [[@somepalliSaintImprovedNeural2021]]
# Notes Sebastian Raschka
-   No code available, so results are not directly reproducible and must be taken with a grain of salt.
-   The paper proposes a contrastive loss for self-supervised learning for tabular data
-   Experiments were done on 69 classification datasets, and the results showed that the self-supervised approach is an improvement compared to purely supervised approaches.

## Visualization
![[SCARF.png]]
## Annotations

“In many machine learning tasks, unlabeled data is abundant but labeled data is costly to collect, requiring manual human labelers. The goal of self-supervised learning is to leverage large amounts of unlabeled data to learn useful representations for downstream tasks such as classification.” ([Bahri et al., 2022, p. 1](zotero://select/library/items/JZ2AZEJD)) ([pdf](zotero://open-pdf/library/items/TL73PSVV?page=1&annotation=43LSA5FV))

“Despite the importance of self-supervised learning, there is surprisingly little work done in finding methods that are applicable across domains and in particular, ones that can be applied to tabular data.” ([Bahri et al., 2022, p. 1](zotero://select/library/items/JZ2AZEJD)) ([pdf](zotero://open-pdf/library/items/TL73PSVV?page=1&annotation=37T425RZ))

“In this paper, we propose SCARF, a simple and versatile contrastive pre-training procedure. We generate a view for a given input by selecting a random subset of its features and replacing them by random draws from the features’ respective empirical marginal distributions.” ([Bahri et al., 2022, p. 1](zotero://select/library/items/JZ2AZEJD)) ([pdf](zotero://open-pdf/library/items/TL73PSVV?page=1&annotation=6YJP36AV))

“We show that not only does SCARF pre-training improve classification accuracy in the fully-supervised setting but does so also in the presence of label noise and in the semi-supervised setting where only a fraction of the available training data is labeled. Moreover, we show that combining SCARF pre-training with other solutions to these problems further improves them, demonstrating the versatility of SCARF and its ability to” ([Bahri et al., 2022, p. 1](zotero://select/library/items/JZ2AZEJD)) ([pdf](zotero://open-pdf/library/items/TL73PSVV?page=1&annotation=BNM52RKK))


“Recently, Yao et al. (2020) adapted the contrastive framework to large-scale recommendation systems in a way similar to our approach. The key difference is in the way the methods generate multiple views. Yao et al. (2020) proposes masking random features in a correlated manner and applying a dropout for categorical features, while our approach involves randomizing random features based on the features’ respective empirical marginal distribution (in an uncorrelated way).” ([Bahri et al., 2022, p. 3](zotero://select/library/items/JZ2AZEJD)) ([pdf](zotero://open-pdf/library/items/TL73PSVV?page=3&annotation=8MD5CLIW))

“Lastly, also similar to our work is VIME (Yoon et al., 2020), which proposes the same corruption technique for tabular data that we do. They pre-train an encoder network on unlabeled data by attaching “mask estimator” and “feature estimator” heads on top of the encoder state and teaching the model to recover both the binary mask that was used for corruption as well as the original uncorrupted input, given the corrupted input. The pre-trained encoder network is subsequently used for semisupervised learning via attachment of a task-specific head and minimization of the supervised loss as well as an auto-encoder reconstruction loss. VIME was shown to achieve state-of-art results on genomics and clinical datasets. The key differences with our work is that we pre-train using a contrastive loss, which we show to be more effective than the denoising auto-encoder loss that partly constitutes VIME. Furthermore, after pre-training we fine-tune all model weights, including the encoder (unlike VIME, which only fine-tunes the task head), and we do so using task supervision only” ([Bahri et al., 2022, p. 3](zotero://select/library/items/JZ2AZEJD)) ([pdf](zotero://open-pdf/library/items/TL73PSVV?page=3&annotation=ZGG37Q2J))