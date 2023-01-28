
![[context-xl-transformer.png]]
(found in [[@daiTransformerXLAttentiveLanguage2019]])


What is attention in general?
How does attention work?
What is self-attention and how is it computed?
What is cross-attention?
What is problematic about attention?


- What is Attention in general?
- Where does it originate from? Where was it introduced?


The study of the transformer architecture has focused on the role and function of self-attention layers (Voita et al., 2019; Clark et al., 2019; Vig and Belinkov, 2019) and on inter-layer differences (i.e. lower vs. upper layers) (Tenney et al., 2019; Jawahar et al., 2019). (look up citations in [[@gevaTransformerFeedForwardLayers2021]])

- What is learned in lower layers? What is learned in upper layers?

## Self-attention
- https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/
- Detailed explanation and implementation. Check my understanding against it: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- intuition behind multi-head and self-attention e. g. cosine similarity, key and querying mechanism: https://www.youtube.com/watch?v=mMa2PmYJlCo&list=PL86uXYUJ7999zE8u2-97i4KG_2Zpufkfb
- See [[@michelAreSixteenHeads2019]] for discussion if all heads are necessary

- See [[@zhangDiveDeepLearning2021]] for intuitive explanation
- See [[@tayEfficientTransformersSurvey2022]] for computational complexity
- https://ai.stanford.edu/blog/contextual/
- Why does it make sense? 
- good blog post for intuition https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
- good blog post for inuition on attention: https://storrs.io/attention/
Nice overview over attention and self-attention:
- https://slds-lmu.github.io/seminar_nlp_ss20/attention-and-self-attention-for-nlp.html
https://angelina-yang.medium.com/whats-the-difference-between-attention-and-self-attention-in-transformer-models-2846665880b6
- visualization of attention using a dedicated programming language https://github.com/srush/raspy
- nice visualization / explanation of self-attention. https://peltarion.com/blog/data-science/self-attention-video
- intuition behind multi-head and self-attention e. g. cosine similarity, key and querying mechanism: https://www.youtube.com/watch?v=mMa2PmYJlCo&list=PL86uXYUJ7999zE8u2-97i4KG_2Zpufkfb
- Course on nlp / transformers: https://phontron.com/class/anlp2022/schedule.html
- See also [[@elhage2021mathematical]]
- https://www.borealisai.com/research-blogs/tutorial-16-transformers-ii-extensions/
- discuss problems with computational complexity and that approximations exist

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

- nice explanation of transformers, such as dot-product attention https://t.co/WOlBY3suy4

- Very high level overview: https://www.youtube.com/watch?app=desktop&v=SZorAJ4I-sA
- low-level  overview. Fully digest these ideas: https://transformer-circuits.pub/2021/framework/index.html
- notebook with nice visuals: https://github.com/dvgodoy/PyTorchStepByStep/blob/master/Chapter10.ipynb

- efficiency [[@kitaevReformerEfficientTransformer2020]]

## Multi-headed attention
- [[@vaswaniAttentionAllYou2017]] propose to run several attention heads in parallel, instead of using a single attention heads.
- They write it improves performance and has a similar computational cost to the single-headed version
- multiple heads may also learn different patterns, not all are equally important, and some can be pruned. This means given a specific context there attention heads learn to attend to different patterns. 
- interesting blog post on importance of heads and pruning them: https://lena-voita.github.io/posts/acl19_heads.html
- Heads have a different importance and many can even be pruned: https://proceedings.neurips.cc/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf
- It's just a logical concept. A single data matrix is used for the Query, Key, and Value, respectively, with logically separate sections of the matrix for each Attention head. Similarly, there are not separate Linear layers, one for each Attention head. All the Attention heads share the same Linear layer but simply operate on their ‘own’ logical section of the data matrix.
- https://transformer-circuits.pub/2021/framework/index.html

## Go Beyond
- Self-attention is central to the transformer architecture, yet other attention mechanisms can be used
- Why do authors propose a different mechanism for tabular data?
- All subsequent architectures use the self-attention mechanism