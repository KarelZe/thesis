![[ft_transformer.png]]
(own drawing somewhat inspired by figure [[@gorishniyRevisitingDeepLearning2021]] and page 5)

- Variant of the classical transformer, but for tabular data. Published in [[@gorishniyRevisitingDeepLearning2021]]
- Firstly, Feature Tokenizer transforms features to embeddings. The embeddings are then processed by the Transformer module and the final representation of the (CLS) token is used for prediction.
- Very likely interpretable... 
- Analysed in [[@grinsztajnWhyTreebasedModels2022]]
- Work out differences between the three:
![[feature-tokenizer.png]]

![[comparison-ft-tab-transformer.png]]


On different embeddings check my understanding against:
- https://towardsdatascience.com/transformers-for-tabular-data-part-3-piecewise-linear-periodic-encodings-1fc49c4bd7bc

- Variant of the classical transformer, but for tabular data. Published in [[@gorishniyRevisitingDeepLearning2021]]
- Firstly, Feature Tokenizer transforms features to embeddings. The embeddings are then processed by the Transformer module and the final representation of the (CLS) token is used for prediction.
- Very likely interpretable... 
- Work out differences between the three:


On different embeddings check my understanding against:
- https://towardsdatascience.com/transformers-for-tabular-data-part-3-piecewise-linear-periodic-encodings-1fc49c4bd7bc
- https://www.kaggle.com/code/antonsruberts/tabtransformer-w-pre-training


## Notes from W&B Paper Reading Group
(See here: https://www.youtube.com/watch?v=59uGzJaVzYc)

- there is a lack of benchmarks
- deep learning models are interesting for multi-modal use cases
- feature tokenizer is just a look-up table as well
- distillation, learning rate warmup, learning rate decay is not used in paper,  but could improve training times and maybe accuracy.
- there is a paper that studies that studies ensembeling for deep learning (Fort et al, 2020)
- there is no universal solution of gbdt and deep learning models
- deep learning is less interpretable