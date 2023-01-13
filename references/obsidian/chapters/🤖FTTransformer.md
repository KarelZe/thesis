![[ft_transformer.png]]
(own drawing somewhat inspired by figure [[@gorishniyRevisitingDeepLearning2021]] and page 5) ^6a12a6

The FTTransformer of [[@gorishniyRevisitingDeepLearning2021]] is another adaption of the classical Transformer ([[@vaswaniAttentionAllYou2017]]) and Bert [[@devlinBERTPretrainingDeep2019]] for the tabular domain. Opposed to the [[TabTransformer]], the FTTransformer, introduces a  *feature tokenizer*, for embedding both continuous and categorical inputs, that are contextualized in a stack of Transformer layers, as shown in Figure [[#^6a12a6]]. 


For embedding 


Conceptually different from the aforementioned approaches, the Transformer-blocks are arranged in norm-last ...

Also deriving 

While this linear mapping is conceptually simple, later works of [[@gorishniyRevisitingDeepLearning2021]] suggests, that line
![[Pasted image 20230113154921.png]]

![[Pasted image 20230113154446.png]]

![[Pasted image 20230113154737.png]]



A special $[\texttt{CLS}]$ token of dimension $\mathbb{R}^{e_d}$ is appended the column embeddings with $X = \left[e_1, e_2, \ldots e_{n}, \texttt{[CLS]}\right]$ , where $X \in \mathbb{R}^{e_d \times n +1}$.

$X$ is passed through a sequence 


Is there a mask somewhere


![[Pasted image 20230113160607.png]]

- Firstly, Feature Tokenizer transforms features to embeddings. The embeddings are then processed by the Transformer module and the final representation of the (CLS) token is used for prediction.
- Very likely interpretable... 
- Analysed in [[@grinsztajnWhyTreebasedModels2022]]

![[Pasted image 20230113150825.png]]

![[Pasted image 20230113151427.png]]
(from revisiting [[@gorishniyRevisitingDeepLearning2021]])

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