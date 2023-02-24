

**Problem formulation:**
- What type of data do we deal with?
- What is tabular data? What do we mean by categorical and numerical?
- What is probabilistic classification? Why do we do probabilistic classification?

**Criteria formulation:**
- What criteria should the models fulfil?
	- Interpretable
	- SOTA performance
	- Extendable for learning on unlabelled data
	- Suitable for classification
- Why do these criteria make sense?
- Why is it hard to assess the performance for tabular data? Can we be sure about SOTA?

**Discussion:**
- Which models does the discussion consider?
- What models do we try out finally?



**Notes:**
[[🍪Selection of supervised approaches notes]]

Due to the tabular nature of the data, with features arranged in a row-column fashion, the token embedding (see chapter [[🛌Token Embedding]]) is replaced for a *column embedding*. Also the notation needs to be adapted to the tabular domain. We denote the data set with $D:=\left\{\left(\mathbf{x}_k, y_k\right) \right\}_{k=1,\cdots N}$ identified with $\left[N_{\mathrm{D}}\right]:=\left\{1, \ldots, N_{\mathrm{D}}\right\}$.  Each tuple $(\boldsymbol{x}, y)$ represents a row in the data set, and consist of the binary classification target $y \in \mathbb{R}$ and the vector of features $\boldsymbol{x} = \left\{\boldsymbol{x}_{\text{cat}}, \boldsymbol{x}_{\text{cont}}\right\}$, where $x_{\text{cont}} \in \mathbb{R}^c$ denotes all $c$ numerical features and $\boldsymbol{x}_{\text{cat}}\in \mathbb{R}^{m}$ all $m$ categorical features. We denote the cardinality of the $j$-th feature with $j \in 1, \cdots m$ with $N_{C_j}$.


“We assume the reader is familiar with these concepts. For a complete reference see Hastie et al. [2009]. Let us just lay out our notation and say that in our framework we receive from an axiomatic data preparation stage an ordered set of multivariate observations W = (X , y). y is the outcome or target ordered set with individual elements y. Similarly, X and X are the feature-vector ordered set and element, respectively. Components of feature vectors are individual features, denoted x (ordered set) and x (element). Target and feature-vector elements y and X pertaining to the same element of W are said to be W-associated. The modeler’s goal is to infer the value of a target element, from its associated feature-vector element and from a separate group of observations, called the training examples Wtr. The solution to this problem is a model ˆ y = M(X , Wtr). We say that the model’s observational inputs for inferring ˆ y are X and Wtr, and this relation between the various entities in the framework is the base for our discussion.” ([[@kaufmanLeakageDataMining2012]], p. 158)