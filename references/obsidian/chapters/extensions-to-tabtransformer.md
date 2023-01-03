 See [[@huangTabTransformerTabularData2020]] for extensions.
- Authors use unsupervised pretraining and supervised finetuning. They also try out techniques like pseudo labelling from [[@leePseudolabelSimpleEfficient 1]] for semi supervised learning among others.
- For pratical implementation see: - For pratical implementation see [Self Supervised Pretraining - pytorch_widedeep](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/self_supervised_pretraining.html)
- single output neuron, fused loss describe what loss function is used

- See paper [[@huangTabTransformerTabularData2020]]
- TabTransformer can't capture correlations between categorical and continous features. See [[🧠Deep Learning Methods/Transformer/@somepalliSAINTImprovedNeural2021]]
- Investigate whether my dataset even profits from this type of architecture?
- See about embedding continous features in [[@somepalliSAINTImprovedNeural2021]]
- On considertion of different weights at an algorithm level see [[@huyenDesigningMachineLearning]] (p. 111)

Video about TabTransformer:
https://m.youtube.com/watch?v=-ZdHhyQsvRc

Reading group on TabTransformer: https://m.youtube.com/watch?v=59uGzJaVzYc
