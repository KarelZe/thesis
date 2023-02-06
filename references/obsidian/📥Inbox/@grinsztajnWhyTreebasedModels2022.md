
title: Why do tree-based models still outperform deep learning on typical tabular data?
authors: Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux
year: 2022
tags :  #gbm  #dt #neural_network #transformer
status : #📥  
related: 
- [[@friedmanGreedyFunctionApproximation2001]]
- [[@hastietrevorElementsStatisticalLearning2009]]
- [[@breimanRandomForests2001]]
- [[@arikTabNetAttentiveInterpretable2020]]
- [[@huangTabTransformerTabularData2020]]
Code: 
- https://github.com/LeoGrin/tabular-benchmark
- https://openreview.net/forum?id=Fp7__phQszn
## Notes Sebastian Raschka
-   The main takeaway is that tree-based models (random forests and XGBoost) outperform deep learning methods for tabular data on medium-sized datasets (10k training examples).
-   The gap between tree-based models and deep learning becomes narrower as the dataset size increases (here: 10k -> 50k).
-   Solid experiments and thorough investigation into the role of uninformative features: uninformative features harm deep learning methods more than tree-based methods.
-   Small caveats: some of the recent tabular methods for deep learning were not considered; “large” datasets are only 50k training examples (small in many industry domains.)
-   Experiments based on 45 tabular datasets; numerical and mixed numerical-categorical; classification and regression datasets; 10k training examples with balanced classes for main experiments; 50k datasets for “large” dataset experiments.

## Annotations

“This leads to a series of challenges which should guide researchers aiming to build tabular-specific neural network: 1. be robust to uninformative features, 2. preserve the orientation of the data, and 3. be able to easily learn irregular functions.” ([Grinsztajn et al., 2022, p. 1](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=1&annotation=MVSXTHHL))

“Our contributions are as follow: 1. We create a new benchmark for tabular data, with a precise methodology for choosing and preprocessing a large number of representative datasets.” ([Grinsztajn et al., 2022, p. 2](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=2&annotation=XDQV9NKQ))

<mark style="background: #FF5582A6;">“The paper closest to our work is Gorishniy et al. [2021], benchmarking novel algorithms, on 11 tabular datasets. We provide a more comprehensive benchmark, with 45 datasets, split across different settings (medium-sized / large-size, with/without categorical features), accounting for the hyperparameter tuning cost, to establish a standard benchmark.” ([Grinsztajn et al., 2022, p. 2](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=2&annotation=YXJLM6JN)) “FT_Transformer : a simple Transformer model combined with a module embedding categorical and numerical features, created in Gorishniy et al. [2021]. We choose this model because it was benchmarked in a convincing way against tree-based models and other tabular-specific models. It can thus be considered a “best case” for Deep learning models on tabular data.” ([Grinsztajn et al., 2022, p. 5](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=5&annotation=AHYUCL2P))</mark>

“Tuning hyperparameters does not make neural networks state-of-the-art Tree-based models are superior for every random search budget, and the performance gap stays wide even after a large number of random search iterations. This does not take into account that each random search iteration is generally slower for neural networks than for tree-based models (see A.2).” ([Grinsztajn et al., 2022, p. 6](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=6&annotation=K2FYJND8))

“Such results suggest that the target functions in our datasets are not smooth, and that neural networks struggle to fit these irregular functions compared to tree-based models. This is in line with Rahaman et al. [2019], which finds that neural networks are biased toward low-frequency functions. Models based on decision trees, which learn piece-wise constant functions, do not exhibit such a bias.” ([Grinsztajn et al., 2022, p. 6](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=6&annotation=G8TL6WFN))

“MLP-like architectures are not robust to uninformative features In the two experiments shown in Fig. 4, we can see that removing uninformative features (4a) reduces the performance gap between MLPs (Resnet) and the other models (FT Transformers and tree-based models), while adding uninformative features widens the gap. This shows that MLPs are less robust to uninformative features, and, given the frequency of such features in tabular datasets, partly explain the results from Sec. 4.2.” ([Grinsztajn et al., 2022, p. 7](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=7&annotation=TQSG939L))

<mark style="background: #ABF7F7A6;">“Why are MLPs much more hindered by uninformative features, compared to other models? One answer is that this learner is rotationally invariant in the sense of Ng [2004]: the learning procedure which learns an MLP on a training set and evaluate it on a testing set is unchanged when applying a rotation (unitary matrix) to the features on both the training and testing set. On the contrary, tree-based models are not rotationally invariant, as they attend to each feature separately, and neither are FT Transformers, because of the initial FT Tokenizer, which implements a pointwise operation theoretical link between this concept and uninformative features is provided by Ng [2004], which shows that any rotationallly invariant learning procedure has a worst-case sample complexity that grows at least linearly in the number of irrelevant features. Intuitively, to remove uninformative features, a rotationaly invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost.” ([Grinsztajn et al., 2022, p. 8](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=8&annotation=W6LGGVAC))

“Our findings shed light on the results of Somepalli et al. [2021] and Gorishniy et al. [2022], which add an embedding layer, even for numerical features, before MLP or Transformer models this layer breaks rotation invariance. The fact that very different types of embedding seem to improve performance suggests that the sheer presence of an embedding which breaks the invariance is a key part of these improvements. We note that a promising avenue for further research would be to find other ways to break rotation invariance which might be less computationally costly than embeddings.” ([Grinsztajn et al., 2022, p. 9](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=9&annotation=3DIWPNHH))</mark>

“Study how both tree-based models and neural networks cope with specific challenges such as missing data or high-cardinality categorical features, thus extending to neural networks prior empirical work [Cerda et al., 2018, Cerda and Varoquaux, 2020, Perez-Lebel et al., 2022].” ([Grinsztajn et al., 2022, p. 9](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=9&annotation=PCA3SDUE))
