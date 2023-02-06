*title:* Survey on categorical data for neural networks
*authors:* John T. Hancock, Taghi M. Khoshgoftaar
*year:* 2020
*tags:* #categorical #embedding #learned-embeddings
*status:* #📦 
*related:*
- [[@guoEntityEmbeddingsCategorical2016]]
*code:*
*review:*

## Notes 📍

## Annotations 📖

“Entity embedding [3] is an example of an automatic technique” ([Hancock and Khoshgoftaar, 2020, p. 3](zotero://select/library/items/G6I6BSG6)) ([pdf](zotero://open-pdf/library/items/KXEQH6PK?page=3&annotation=QIKWMJ6J))

“We refer to these techniques as entity embedding algorithms. The output of an entity embedding algorithm is a mapping from some set of categorical values S to a space of n-dimensional vectors in Rn” ([Hancock and Khoshgoftaar, 2020, p. 5](zotero://select/library/items/G6I6BSG6)) ([pdf](zotero://open-pdf/library/items/KXEQH6PK?page=5&annotation=L8UP7ZX7))

“Let S be the set of distinct values of some variable that is characterized as categorical data. Then an entity embedding e is a mapping of the elements of S to vectors of real numbers. We define the range of an entity embedding as Rd to allow us the luxury of employing any theorems that hold for real numbers. In practice, the range of our embedding is a finite subset of vectors of rational numbers Qd because computers are currently only capable of storing rational approximations of real numbers. We refer to this as “entity embedding”, “embedding categorical data”, “embedding categorical values”, or simply as “embedding” when the context is clear.” ([Hancock and Khoshgoftaar, 2020, p. 5](zotero://select/library/items/G6I6BSG6)) ([pdf](zotero://open-pdf/library/items/KXEQH6PK?page=5&annotation=V57GC5XH))

“The example we work through here is similar to the functioning of a Keras embedding layer [20]. To compute the embedded value e of the input we compute e = W v, where, W is the edge weight matrix for nodes in the neural network’s embedding layer, and v is the input value. For embedding categorical variables, v must be some encoded value of a categorical variable. Typically, v is a One-hot encoded value. Please see "One-hot encoding" section for a definition of One-hot encoding. Equations 2, and 3 give an example of how one may compute the embedded value of a One-hot encoded categorical variable. Assuming the entry equal to 1 in the column vector on the right-hand side of Eq. 2 is the encoding vector v , and the value equal to 1 is on the jth row of v , the product on the right-hand side of Eq. 2 will be An easy way to think of the embedding process we illustrate in Eqs. 2 and 3 is that W is a look-up table for One-hot encoded values. A neural network’s optimizer applies some procedure such as Stochastic Gradient Descent (SGD ) to update its weight values, including those of the weight matrix W. Over time the optimizer finds a value of W that minimizes the value of some loss function, J. As the neural network changes the value of W, the components of the embedded vectors e change as well. Since the values of e are updated as a result of the neural network’s weight update function, we call the embedding automatic embedding.” ([Hancock and Khoshgoftaar, 2020, p. 8](zotero://select/library/items/G6I6BSG6)) ([pdf](zotero://open-pdf/library/items/KXEQH6PK?page=8&annotation=ETNNDHU9))