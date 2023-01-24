*title:* Saint: Improved Neural Networks for Tabular Data Via Row Attention and Contrastive Pre-Training
*authors:* Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, Tom Goldstein
*year:* 2021
*tags:* #supervised-learning #semi-supervised #deep-learning #tabular-data #supervised-learning #transformer #attention
*status:* #📦 
*related:*
- [[@arikTabNetAttentiveInterpretable2020]] (other transformer)
- [[@huangTabTransformerTabularData2020]] (other transformer)
 [[@vaswaniAttentionAllYou2017]]
- [[@huangTabTransformerTabularData2020]]
- [[@arikTabNetAttentiveInterpretable2020]]
- [[@grinsztajnWhyTreebasedModels2022]]
*code:* [https://github.com/somepago/saint](https://github.com/somepago/saint)
*review:* https://openreview.net/forum?id=nL2lDlsrZU (Was rejected at ICLR. Fairness of the comparison was doubted.) 

## Notes📍
- Authors propose the Self-Attention and Intersample Attention Transforer (SAINT), a transformer-based architecture for tabular data with an attention mechanism over rows and columns, that preserves locality. 
- Additionally, they propose contrastive self-supervised pre-training when labeled data is scarce. This makes the architecture suitable for semi-supervised learning.
- According to the authors their method consistently outperforms other dl and gradient boosting methods.
- Authors claim that there architecture is robust to missing or corrupted features due to the contrastive pre-training.
- They provide an interesting enhancement to TabTransformer ([[@huangTabTransformerTabularData2020]]) by embedding not just categorical data but also continous data.
- Why tabular data is hard:
	- heterogenous features e. g., continous, categorical, ordinal. Values can be correlated or independent.
	- No positional information e.g., columns can be in an arbitrary order
	- Data can come from multiple discrete and continous distributions. 
	- Correlations between features must be learned.
- **Saint-Architecture:** 
	- Architecture seems to be overly complex. (?)
![[saint-architecture.png]]
	- Saint projects categorical and continous variables into a dense vector space and passes the values as tokens into a transformer encoder. Self-attention is used to attend to individual features within each sample and intersample attentation is used to relate a row to other rows in the table.  This is similar to nearest neighbour classification, where the distance metric is learned from data.
	- Every feature is projected into a $d$ dimensional space using a separate fully-connected layer. Separate as tokens can be the same but have a different meaning in each column. The embeddings of the projections are then passed into the transformer encoder. Done similarily for categorical data only in the TabTransformer ([[@huangTabTransformerTabularData2020]]).
	- Uses to self-supervised contrastive pre-training to profit from learning on unlabeled data.
	- The saint architecture consists of $L$ stages out of which each stage consists of a self-attention transformer block (eq. encoder of [[@vaswaniAttentionAllYou2017]]) and a intersample transformer block (similar to self-attention transformer block, but with intersample attention). Each transformer block uses skip connections, layer normalization and $\operatorname{GeLU}(\cdot)$. 
	- Intersample attention is described below.
	- SAINT is interpreteable (with one?) transformer stage.
- Requires adjustments to the architecture e. g., loss function for pretraining and finetuning? -> can this be used as a criteria to render out the architecture.
- **Intersample attention:** Intersample attention is computed across different rows in a batch of data. They concat the embeddings of each feature and compute the attention over samples. The advantage is that missing or noise features in one row can borrow features from another data sample in the batch. -> Problematic for financial data with a low-signal to noise ratio?
![[intersample-attention.png]]
- **Contrastrive learning:**
	- In contrastive pre-training the distance between two views of the same point is minimized while the distance between two other points is maximized.
	- Contrastive learning has been used in other domains, where models are pre-trained to be invariant to reodering, cropping etc.
	- Transfer to tabular data is difficult. [[@yoonVIMEExtendingSuccess2020]] uses mixup for continous data. The authors here use CutMix to augment samples in the input space and mixup in the embedding space. (see original papers or [[@heBagTricksImage2018]] for mixup), which mixes two samples.
- **Results:**
	- Pre-trained SAINT (semi-supervised variant) improves over SAINT (supervised variant). The benefit of pre-training is less pronounced if all samples are labelled.
	- Off all architectures tested (e. g. XGBoost, CatBoost, LightGBM, MLP, VIME, TabNet, TabTransformer), SAINT performs best. All architecture perform better, when trained on more instances.
**Extensions to TabTransformer:**
- They modify TabTransformer by embedding continous features into $d$ dimensions using a single layer $\operatorname{ReLU}$ MLP, that is originally just used for categorical features and pass the embedded features through a transformer lbock. Otherwise the architecture and tuning is the same. Helps to boost the AUROC score from 89.38 to 91.72.
## Notes Sebastian Raschka
-   The Self-Attention and Intersample Attention Transformer (SAINT) hybrid architecture is based on self-attention that applies attention across both rows and columns.
-   Also proposes a self-supervised learning technique for pre-training under scarce data regimes.
-   When looking at the average performance across all nine datasets, the proposed SAINT method tends to outperform gradient-boosted trees. The datasets ranged from 200 to 495,000 examples.
## Annotations

“Our method, SAINT, performs attention over both rows and columns, and it includes an enhanced embedding method. We also study a new contrastive self-supervised pre-training method for use when labels are scarce.” ([Somepalli et al., 2021, p. 1](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=1&annotation=KM89W4QU))

“SAINT consistently improves performance over previous deep learning methods, and it even outperforms gradient boosting methods, including XGBoost, CatBoost, and LightGBM, on average over a variety of benchmark tasks.” ([Somepalli et al., 2021, p. 1](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=1&annotation=CRU8WRY7))

“First, tabular data often contain heterogeneous features that represent a mixture of continuous, categorical, and ordinal values, and these values can be independent or correlated. Second, there is no inherent positional information in tabular data, meaning that the order of columns is arbitrary.” ([Somepalli et al., 2021, p. 1](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=1&annotation=IRK5ZD7E))

“Tabular models must handle features from multiple discrete and continuous distributions, and they must discover correlations without relying on the positional information.” ([Somepalli et al., 2021, p. 1](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=1&annotation=3GTX92EN))

“SAINT projects all features – categorical and continuous into a combined dense vector space. These projected values are passed as tokens into a transformer encoder which uses attention in the following two ways. First, there is “self-attention,” which attends to individual features within each data sample. Second, we propose a novel “intersample attention,” which enhances the classification of a row (i.e., a data sample) by relating it to other rows in the table. Intersample attention is akin to a nearest-neighbor classification, where the distance metric is learned end-to-end rather than fixed. In addition to this hybrid attention mechanism, we also leverage self-supervised contrastive pre-training to boost performance for semi-supervised problems.” ([Somepalli et al., 2021, p. 2](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=2&annotation=M3AJUVXC))

“TabNet [1] uses neural networks to mimic decision trees by placing importance on only a few features at each layer. The attention layers in that model do not use the regular dot-product self-attention common in transformer-based models, rather there is a type of sparse layer that allows only certain features to pass through” ([Somepalli et al., 2021, p. 2](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=2&annotation=W546LBI8))

“Transformer models for more general tabular data include TabTransformer [18], which uses a transformer encoder to learn contextual embeddings only on categorical features.” ([Somepalli et al., 2021, p. 2](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=2&annotation=NXRU9FJZ))

“The continuous features are concatenated to the embedded features and fed to an MLP. The main issue with this model is that continuous data do not go through the self-attention block. That means any information about correlations between categorical and continuous features is lost.” ([Somepalli et al., 2021, p. 2](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=2&annotation=K685SR8C))

“Self-Supervised Learning Self-supervision via a ‘pretext task’ on unlabeled data coupled with finetuning on labeled data is widely used for improving model performance in language and computer vision. Some of the tasks previously used for self-supervision on tabular data include masking, denoising, and replaced token detection. Masking (or Masked Language Modeling(MLM)) is when individual features are masked and the model’s objective is to impute their value [1, 18, 32]. Denoising injects various types of noise into the data, and the objective there is to recover the original values [43, 49]. Replaced token detection (RTD) inserts random values into a given feature vector and seeks to detect the location of these replacements [18, 20]. Contrastive pre-training, where the distance between two views of the same point is minimized while maximizing the distance between two different points [5, 12, 15], is another pretext task that applies to tabular data. In this paper, to the best of our knowledge, we are the first to adopt contrastive learning for tabular data. We couple this strategy with denoising to perform pre-training on a plethora of datasets with varied volumes of labeled data, and we show that our method outperforms traditional boosting methods.” ([Somepalli et al., 2021, p. 3](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=3&annotation=NYSIVQZX))

“Contrastive learning, in which models are pre-trained to be invariant to reordering, cropping, or other label-preserving “views” of the data [5, 12, 15, 32, 43], is a powerful tool in the vision and language domains that has never (to our knowledge) been applied to tabular data.” ([Somepalli et al., 2021, p. 5](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=5&annotation=HQX7N9D5))

“Existing self-supervised objectives for tabular data include denoising [43], a variation of which was used by VIME [49], masking, and replaced token detection as used by TabTransformer [18]. We find that, while these methods are effective, superior results are achieved by contrastive learning.” ([Somepalli et al., 2021, p. 5](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=5&annotation=LXSCK4TP))

“It is difficult to craft invariance transforms for tabular data. The authors of VIME [49] use mixup in the non-embedded space as a data augmentation method, but this is limited to continuous data. We instead use CutMix [50] to augment samples in the input space and we use mixup [51] in the embedding space. These two augmentations combined yield a challenging and effective self-supervision task” ([Somepalli et al., 2021, p. 5](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=5&annotation=XZYZV3VZ))

“Semi-supervised setting We perform 3 sets of experiments with 50, 200, and 500 labeled data points (in each case the rest are unlabeled). See Table 3 for numerical results. In all cases, the pre-trained SAINT model (with both self and intersample attention) performs the best. Interestingly, we note that when all the training data samples are labeled, pre-training does not contribute appreciably, hence the results with and without pre-training are fairly close.” ([Somepalli et al., 2021, p. 8](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=8&annotation=ZJHH2RKT))

“Effect of embedding continuous features To understand the effect of learning embeddings for continuous data, we perform a simple experiment with TabTransformer. We modify TabTransformer by embedding continuous features into d dimensions using a single layer ReLU MLP, just as they use on categorical features, and we pass the embedded features through the transformer block. We keep the entire architecture and all training hyper-parameters the same for both TabTransformer and its modified version. The average AUROC of the original TabTransformer is 89.38. Just by embedding the continuous features, the performance jumps to 91.72. This experiment shows that embedding the continuous data is important and can boost the performance of the model significantly.” ([Somepalli et al., 2021, p. 8](zotero://select/library/items/PCV7XCHY)) ([pdf](zotero://open-pdf/library/items/N8H76CQW?page=8&annotation=JK3TUUXE))