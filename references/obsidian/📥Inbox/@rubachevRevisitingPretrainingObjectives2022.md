*title:* Revisiting Pretraining Objectives for Tabular Deep Learning
*authors:* Ivan Rubachev, Artem Alekberov, Yury Gorishniy, Artem Babenko
*year:* 2022
*tags:* #mlp #pre-training #tabular #tabular-data 
*status:* #📦 
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖

“Pretraining in deep learning. For domains with structured data, like natural images or texts, pretraining is currently an established stage in the typical pipelines, which leads to higher general performance and better model robustness [18, 10].” ([Rubachev et al., 2022, p. 2](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=2&annotation=LARIGNA9))

“Pretraining for the tabular domain. Numerous pretraining methods were recently proposed in several recent works on tabular DL [2, 42, 9, 39, 37, 27]. However, most of these works do not focus on the pretraining objective per se and typically introduce it as a component of their tabular DL pipeline. Moreover, the experimental setup varies significantly between methods. Therefore,” ([Rubachev et al., 2022, p. 2](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=2&annotation=ZVQA27B7))

“it is difficult to extract conclusive evidence about pretraining effectiveness from the literature. To the best of our knowledge, there is only one systematic study on the tabular pretraining [4], but its experimental evaluation is performed only with the simplest MLP models, and we found that the superiority of the contrastive pretraining, reported in [4], does not hold for tuned models in our setup, where contrastive objective is comparable to the simpler self-prediction objectives.” ([Rubachev et al., 2022, p. 3](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=3&annotation=LEYWPPJY))

“We use MLP as a simple deep baseline to compare and ablate the methods. Our implementation of MLP exactly follows [16], the model is regularized by dropout and weight decay. As more advanced deep models, we evaluate MLP equipped with numerical feature embeddings, specifically, target-aware piecewise linear encoding (MLP-T-LR) and embeddings with periodic activations (MLPPLR) from [14]. These models represent the current state-of-the-art solution for tabular DL [14], and are of interest as most prior work on pretraining in tabular DL focus on pretraining with the simplest” ([Rubachev et al., 2022, p. 3](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=3&annotation=SRNGIC7T))

“MLP models in evaluation.” ([Rubachev et al., 2022, p. 4](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=4&annotation=DWZBMVGD))

“retraining. Pretraining is always performed directly on the target dataset and does not exploit additional data. The learning process thus comprises two stages. On the first stage, the model parameters are optimized w.r.t. the pretraining objective. On the second stage, the model is initialized with the pretrained weights and finetuned on the downstream classification or regression task. We focus on the fully-supervised setup, i.e., assume that target labels are provided for all dataset objects. Typically, pretraining stage involves the input corruption: for instance, to generate positive pairs in contrastive-like objectives or to corrupt the input for reconstruction in self-prediction based objectives. We use random feature resampling as a proven simple baseline for input corruption in tabular data [4, 42].” ([Rubachev et al., 2022, p. 4](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=4&annotation=FWB2KK4R))

“Pretraining is beneficial for the state-of-the-art models. Models with the numerical feature embeddings also benefit from pretraining with either reconstruction or mask prediction demonstrating the top performance on the downstream task. However, the improvement is typically less noticeable compared to the vanilla MLPs.” ([Rubachev et al., 2022, p. 4](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=4&annotation=D3VWZHY5))

“There is no universal solution between self-prediction objectives. We observe that for some datasets the reconstruction objective outperforms the mask prediction (OT, WE, CO, MI), while on others the mask prediction is better (GE, CH, HI, AD). We also note that the mask prediction objective sometimes leads to unexpected performance drops for models with numerical embeddings (WE, MI), we do not observe significant performance drops for the reconstruction objective.” ([Rubachev et al., 2022, p. 4](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=4&annotation=6C5KD9P8))

“The main takeaway: simple pretraining strategies based on self-prediction lead to significant improvements in the downstream accuracy compared to the tuned supervised baselines learned from scratch across different tabular DL models and datasets. In practice, we recommend trying both” ([Rubachev et al., 2022, p. 4](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=4&annotation=JZ3SNEN7))

“reconstruction and mask prediction as tabular pretraining baselines, as either one might show superior performance depending on the dataset being used.” ([Rubachev et al., 2022, p. 5](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=5&annotation=ZGACJLMC))

“Here we demonstrate that early stopping the pretraining by the value of the pretraining objective on the hold-out validation set is comparable to the early stopping by the downstream metric on the hold-out validation set after finetuning. See Table 12 for the results. This is an important practical observation, as computing pretraining objective is much faster than the full finetuning of the model, especially on large scale datasets.” ([Rubachev et al., 2022, p. 15](zotero://select/library/items/BR4F57P5)) ([pdf](zotero://open-pdf/library/items/AUF4PL5H?page=15&annotation=BY4QFNWP))