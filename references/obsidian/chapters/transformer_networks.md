- Go "deep" instead of wide
- Explain how neural networks can be adjusted to perform binary classification.
- use feed-forward networks to discuss central concepts like loss function, back propagation etc.
- Discuss why plain vanilla feed-forward networks are not suitable for tabular data. Why do the perform poorly?
- How does the chosen layer and loss function to problem framing
- How are neural networks optimized?
- Motivation for Transformers
- For formal algorithms on Transformers see [[@phuongFormalAlgorithmsTransformers2022]]
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://www.youtube.com/watch?v=EixI6t5oif0
- https://transformer-circuits.pub/2021/framework/index.html
- On efficiency of transformers see: https://arxiv.org/pdf/2009.06732.pdf
- Mathematical foundation of the transformer architecture: https://transformer-circuits.pub/2021/framework/index.html
- Detailed explanation and implementation. Check my understanding against it: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- On implementation aspects see: https://arxiv.org/pdf/2007.00072.pdf
- batch nromalization is not fully understood. See [[@zhangDiveDeepLearning2021]] (p. 277)


feature importance evaluation is a non-trivial problem due to missing ground truth. See [[@borisovDeepNeuralNetworks2022]] paper for citation
- nice visualization / explanation of self-attention. https://peltarion.com/blog/data-science/self-attention-video

- intuition behind multi-head and self-attention e. g. cosine similarity, key and querying mechanism: https://www.youtube.com/watch?v=mMa2PmYJlCo&list=PL86uXYUJ7999zE8u2-97i4KG_2Zpufkfb
- visualization of attention using a dedicated programming language https://github.com/srush/raspy

![[feature_attributions.png]]
(compare kernelshap with feature attributions from attention [[@borisovDeepNeuralNetworks2022]])
(from [[@borisovDeepNeuralNetworks2022]])

![[attention-map-somepalli.png]]
[[@somepalliSAINTImprovedNeural2021]]


In absence of a ground truth for the true feature attribution, the *optimal* approach is to be determined.

While not explored systematically for the tabular domain, the specific roles of different attention heads have been studied in transformer-based machine translation (see e. g., [[@voitaAnalyzingMultiHeadSelfAttention2019]], [[@tangAnalysisAttentionMechanisms2018]]).  [[@voitaAnalyzingMultiHeadSelfAttention2019]] observe, that heads in the multi-headed self-attention mechanism serve distinct purposes like learning positional or syntactic relations between tokens.  Transferring their result back to the tabular domain, averaging over multiple heads may lead to undesired obfuscation effects. 

![[chefer-attention-maps-calculation.png]]
(Adapted from [[@cheferTransformerInterpretabilityAttention2021]]) Note, that the relevance maps are different from the raw attention values. Also requires gradients. Removed for another approach. Do not use as it is outdated.)

![[Pasted image 20230108080816.png]]
