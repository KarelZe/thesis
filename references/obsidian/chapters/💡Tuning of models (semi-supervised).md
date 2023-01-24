- Justify training of semi-supervised model from theoretical perspective with findings in chapter [[#^c77130]] . 
- Use learning curves from [[#^d50f5d]].
- for pre-training using ELECTRA see: https://blog.ml6.eu/how-a-pretrained-tabtransformer-performs-in-the-real-world-eccb12362950
- For pre-training objectives see: https://github.com/puhsu/tabular-dl-pretrain-objectives/
- For implementation of masked language modelling see https://nn.labml.ai/transformers/mlm/index.html
- form implementation of semi-supervised catboost see: https://github.com/catboost/catboost/issues/525

“For deep models with transfer learning, we tune the hyperparameters on the full upstream data using the available large upstream validation set with the goal to obtain the best performing feature extractor for the pre-training multi-target task. We then fine-tune this feature extractor with a small learning rate on the downstream data. As this strategy offers considerable performance gains over default hyperparameters, we highlight the importance of tuning the feature extractor and present the comparison with default hyperparameters in Appendix B as well as the details on hyperparameter search spaces for each model.” ([Levin et al., 2022, p. 6](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/QCVUFCDQ?page=6&annotation=PICSZEZU)) [[@levinTransferLearningDeep2022]]