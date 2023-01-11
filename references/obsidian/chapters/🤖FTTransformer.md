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