## Self-Training

- changes to the algorithm raises concerns about comparability
- alternative is to wrap gradient-boosting with a wrapper algorithm, which leaves the algorithm unalatered and yet achieves a similar goal
- keep it short


## Pre-Training + Transformer
Next, we look into extending FT-Transformer for the semi-supervised setting. Whilst Transformers could be combined with self-training, a more promising approach is to pre-train Transformers on unlabelled data, and then fine-tune the network on the remaining labelled instances. Various studies report unanimously performance improvements from pre-training attention-based architectures, including ([[@somepalliSAINTImprovedNeural2021]]8), ([[@arikTabNetAttentiveInterpretable2020]]7), and ([[@huangTabTransformerTabularData2020]]7). An exception are ([[@levinTransferLearningDeep2022]]7), who find no improvements from pre-training the FT-Transformer.

Until now we assumed the parameters e.g., weights and biases, of the Transformer to be initialized randomly. The joint goal of pre-training objectives is to initialize a neural network with weights that capture expressive representations of the input and thereby improve convergence and model performance over a random initialization when fine-tuning on a specific task. (Verify + cite [[@erhanWhyDoesUnsupervised]]) Pre-training objectives for tabular data differ vastly in their methodology and are often a direct adaption from other domains and include pre-training through masking, token replacement, or constrastive learning. As such, ([[@huangTabTransformerTabularData2020]]7) use gls-mlm, whereby features are randomly masked and the objective is to reconstruct the original input. Pre-training by gls-rtd aims to recover a binary mask, used for random feature replacement ([[@huangTabTransformerTabularData2020]]7). ([[@bahriSCARFSelfSupervisedContrastive2022]]3) and ([[@yoonVIMEExtendingSuccess2020]]4--5) reconstruct both the binary feature mask and to recover the original input. ([[@somepalliSAINTImprovedNeural2021]]3) alter the methodology of ([[@yoonVIMEExtendingSuccess2020]]4--5) through a contrastive loss function.  

With a multitude of methods, tested on differing datasets and neural architectures, a fair comparison between pre-training methods is complicated.  However, [[@rubachevRevisitingPretrainingObjectives2022]]2--3) provide guidance for selecting objectives in their work. Among the pre-training objectives that they convincingly benchmarked, the gls-mlm objective proposed by ([[@devlinBERTPretrainingDeep2019]]4174) was found to be one of the best-performing approaches. The gls-mlm objective is easy to optimize and does not require modifications to the model architecture, which is an important property when comparing the supervised and semi-supervised variant. This makes gls-mlm a compelling choice for pre-training on unlabelled data.

**Notes:**
[[🍪Selection of semisupervised Approaches notes]]

Interesting resources on pre-training:
- https://arxiv.org/pdf/2109.07437.pdf
- https://phontron.com/class/anlp2022/assets/slides/anlp-07-pretraining.pdf