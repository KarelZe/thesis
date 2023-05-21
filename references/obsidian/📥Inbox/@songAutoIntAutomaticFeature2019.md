*title:* AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
*authors:* Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, Jian Tang
*year:* 2019
*tags:* #self-attention #embeddings #categorical #continous 
*status:* #📦 
*related:*
- [[@gorishniyRevisitingDeepLearning2021]] (similar idea)
- [[@gorishniyEmbeddingsNumericalFeatures2022]] (reference this paper)
*code:*
*review:*

## Notes 📍

## Annotations 📖
<mark style="background: #ADCCFFA6;">“4.3 Embedding Layer Since the feature representations of the categorical features are very sparse and high-dimensional, a common way is to represent them into low-dimensional spaces (e.g., word embeddings). Specifically, we represent each categorical feature with a low-dimensional vector, i.e., ei = Vixi, (2) where Vi is an embedding matrix for field i, and xi is an one-hot vector. Often times categorical features can be multi-valued, i.e., xi is a multi-hot vector. Take film watching prediction as an example, there could be a feature field Genre which describes the types of a film and it may be multi-valued (e.g., Drama and Romance for film “Titanic”). To be compatible with multi-valued inputs, we further modify the Equation 2 and represent the multi-valued feature field as the average of corresponding feature embedding vectors: ei = 1 q Vixi, (3) where q is the number of values that a sample has for i-th field and xi is the multi-hot vector representation for this field. To allow the interaction between categorical and numerical features, we also represent the numerical features in the same lowdimensional feature space. Specifically, we represent the numerical feature as em = vmxm, (4) where vm is an embedding vector for field m, and xm is a scalar value. By doing this, the output of the embedding layer would be a concatenation of multiple embedding vectors, as presented in Figure 2.” ([Song et al., 2019, p. 4](zotero://select/library/items/2PWVWL5T)) ([pdf](zotero://open-pdf/library/items/HBV6667L?page=4&annotation=UVTMKE9G))</mark>