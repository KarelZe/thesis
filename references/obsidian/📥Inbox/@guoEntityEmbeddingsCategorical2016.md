*title:* Entity Embeddings of Categorical Variables
*authors:* Cheng Guo, Felix Berkhahn
*year:* 2016
*tags:* #categorical #embeddings
*status:* #📦 
*related:*
- [[@borisovDeepNeuralNetworks2022]] (cite this paper)
- [[@chengWideDeepLearning2016]] (use categorical embeddings)
- [[@songAutoIntAutomaticFeature2019]] (use categorical embeddings)
*code:*
*review:*

## Notes 📍

## Annotations 📖

“We map categorical variables in a function approximation problem into Euclidean spaces, which are the entity embeddings of the categorical variables. The mapping is learned by a neural network during the standard supervised training process. Entity embedding not only reduces memory usage and speeds up neural networks compared with one-hot encoding, but more importantly by mapping similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables.” ([Guo and Berkhahn, 2016, p. 1](zotero://select/library/items/5CUI2BTM)) ([pdf](zotero://open-pdf/library/items/TMTTKAZP?page=1&annotation=CYY7CN4R))

“As entity embedding defines a distance measure for categorical variables it can be used for visualizing categorical data and for data clustering.” ([Guo and Berkhahn, 2016, p. 1](zotero://select/library/items/5CUI2BTM)) ([pdf](zotero://open-pdf/library/items/TMTTKAZP?page=1&annotation=AWIQWR2X))

“Therefore, naively applying neural networks on structured data with integer representation for category variables does not work well. A common way to circumvent this problem is to use onehot encoding, but it has two shortcomings: First when we have many high cardinality features one-hot encoding often results in an unrealistic amount of computational resource requirement. Second, it treats different values of categorical variables completely independent of each other and often ignores the informative relations between them.” ([Guo and Berkhahn, 2016, p. 1](zotero://select/library/items/5CUI2BTM)) ([pdf](zotero://open-pdf/library/items/TMTTKAZP?page=1&annotation=XLJYFFSJ))

“The most common variable types in structured data are continuous variables and discrete variables. Continuous variables such as temperature, price, weight can be represented by real numbers. Discrete variables such as age, color, bus line number can be represented by integers. Often the integers are just used for convenience to label the different states and have no information in themselves. For example if we use 1, 2, 3 to represent red, blue and yellow, one can not assume that ”blue is bigger than red” or ”the average of red and yellow are blue” or anything that introduces additional information based on the properties of integers. These integers are called nominal numbers. Other times there is an intrinsic ordering in the integer index such as age or month of the year. These integers are called cardinal number or ordinal numbers. Note that the meaning or order may not be more useful for the problem than only considering the” ([Guo and Berkhahn, 2016, p. 3](zotero://select/library/items/5CUI2BTM)) ([pdf](zotero://open-pdf/library/items/TMTTKAZP?page=3&annotation=BW76G7AJ))

“With entity embedding we want to put similar values of a categorical variable closer to each other in the embedding space. If we use a real number to define similarity of the values then entity embedding is closely related to the embedding of finite metric space problem in topology” ([Guo and Berkhahn, 2016, p. 4](zotero://select/library/items/5CUI2BTM)) ([pdf](zotero://open-pdf/library/items/TMTTKAZP?page=4&annotation=EEMP9PII))

“Neural networks with one-hot encoding give slightly better results than entity embedding for the shuffled data while entity embedding is clearly better than one-hot encoding for the non-shuffled data. The explanation is that entity embedding, by restricting the network in a much smaller parameter space in a meaningful way, reduces the chance that the network converges to local minimums far from the global minimum. More intuitively, entity embeddings force the network to learn the intrinsic properties of each of the feature as well as the sales distribution in the feature space” ([Guo and Berkhahn, 2016, p. 6](zotero://select/library/items/5CUI2BTM)) ([pdf](zotero://open-pdf/library/items/TMTTKAZP?page=6&annotation=S5JSEQQM))