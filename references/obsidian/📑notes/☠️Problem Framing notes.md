TabTransformer uses Transformers to generate robust data representations — embeddings — for categorical variables, or variables that take on a finite set of discrete values, such as months of the year. Continuous variables (such as numerical values) are processed in a parallel stream. https://www.amazon.science/blog/bringing-the-power-of-deep-learning-to-data-in-tables

Assume a data distribution $\mathcal{D}$ on $\mathcal{X} \times \mathcal{Y}$, where $\mathcal{X} \subseteq \mathbb{R}^n$ denotes the feature space, and $\mathcal{Y}$  the target space. We previously defined $\mathcal{Y}$ to be $\{-1,1\}$, with $-1$ indicating sells and $1$ buys. The training is denoted as $\mathcal{D}_N=\left\{\left(x_i, y_i\right)\right\}_{i=1}^N$  with $N$ i.i.d. samples from $\mathcal{D}$.  

We denote our dataset with $N$ samples by $\mathcal{D} = \{(\boldsymbol{x}_n, y_n)\}_{n = 1}^{N}$ . Each tuple $(\boldsymbol{x}, y)$ represents a row in the data set, and consist of the binary classification target $y \in \mathbb{Y}$ with $\mathbb{Y}=\{-1,1\}$ and the vector of features $\boldsymbol{x} = \left\{\boldsymbol{x}_{\text{cat}}, \boldsymbol{x}_{\text{cont}}\right\}$, where $x_{\text{cont}} \in \mathbb{R}^c$ denotes all $c$ numerical features and $\boldsymbol{x}_{\text{cat}}\in \mathbb{R}^{m}$ all $m$ categorical features. We denote the cardinality of the $j$-th feature with $j \in 1, \cdots m$ with $N_{C_j}$.

Each data instance consists of a feature vector and target. The feature vector is given by $\boldsymbol{x} \in \mathcal{X}$ with $\mathcal{X} \subseteq \mathbb{R}^n$, described by a random variable $X$. Like before, the target or trade initiator is given by $y \in \mathcal{Y}$ and described by a random variable $Y$. Each data instance is sampled from the joint probability distribution $p^*(X, Y)$. The training is denoted as $\mathcal{D}_N=\left\{\left(x_i, y_i\right)\right\}_{i=1}^N$  with $N$ i.i.d. samples from $p^*$.  

$\mathcal{D}=\left\{\left(\mathbf{x}_n, y_n\right)\right\}_{n=1}^N$ : Dataset $\mathcal{D}$ with $N$ i.i.d (independant and identically distributed) samples from $p^*$

For the prediction $\hat{y} \in \mathbb{Y}$ with of classical trade classification rules we simply estimate the probability as 

For notation see: https://web.stanford.edu/~nanbhas/blog/some-unifying-notation/
https://ee104.stanford.edu/lectures/prob_classification.pdf

Thus, the classifier returns a probability for 

Classical trade classification rules perform a discrete classification. As such, we assign a probability of $1$ for the predicted class  $\hat{v} \in \mathcal{V}$ and zero otherwise.

if point classifier predicts $\hat{v} \in \mathcal{V}$, associated probabilistic classifier returns $\hat{p}$, with


i.e., the value in $\mathcal{V}$ that has highest probability
called a maximum likelihood classifier
extends to a list classifier, by giving values sorted by probability, largest to smallest

In order to maintain 

Due to the tabular nature of the data, with features arranged in a row-column fashion, the token embedding (see chapter [[🛌Token Embedding]]) is replaced for a *column embedding*. Also the notation needs to be adapted to the tabular domain. We denote the data set with $D:=\left\{\left(\mathbf{x}_k, y_k\right) \right\}_{k=1,\cdots N}$ identified with $\left[N_{\mathrm{D}}\right]:=\left\{1, \ldots, N_{\mathrm{D}}\right\}$.  Each tuple $(\boldsymbol{x}, y)$ represents a row in the data set, and consist of the binary classification target $y \in \mathbb{R}$ and the vector of features $\boldsymbol{x} = \left\{\boldsymbol{x}_{\text{cat}}, \boldsymbol{x}_{\text{cont}}\right\}$, where $x_{\text{cont}} \in \mathbb{R}^c$ denotes all $c$ numerical features and $\boldsymbol{x}_{\text{cat}}\in \mathbb{R}^{m}$ all $m$ categorical features. We denote the cardinality of the $j$-th feature with $j \in 1, \cdots m$ with $N_{C_j}$.

https://medium.com/@oded.kalev/comparing-classifiers-using-roc-9a9d8c9c819b

(See [[@easleyDiscerningInformationTrade2016]]). However, this is not the case for the algorithms working on a trade-per-trade basis. Still, one can derive probabilities

Among numerous classifiers, some are hard classifiers while some are soft ones. Soft classifiers explicitly estimate the class conditional probabilities and then perform classification based on estimated probabilities. In contrast, hard classifiers directly target on the classification decision boundary without producing the probability estimation. (from [[@liuHardSoftClassification2011]]).


**Why probabilistic classification:**
- Due to a unsatisfactory research situation, for trade classification (see chapter [[👪Related Work]]) we base
- Use classification methods (*probabilistic classifier*) that can return probabilities instead of class-only for better analysis. Using probabilistic trade classification rules might have been studied in [[@easleyDiscerningInformationTrade2016]]
- Why to formulate problem as probabilistic classification problem: https://www.youtube.com/watch?v=RXMu96RJj_s
- Could be supervised if all labels are known
- Could be semi-supervised if only some of the labels are known. Cover later.
- hard and soft classification in general [[@liuHardSoftClassification2011]] and neural networks [[@foodyHardSoftClassifications2002]] (do not cite?)


- It's not easy to decide between *hard* and *soft classification*. See some references in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3233196/

hard decision boundary / boolean decision.

An accuracy-interpretability trade-off [42]istrue for almost all machine learning methods. For example, deep learning networks, an advanced form of machine learning, typically combine the activities of several hundred or even thousands of neurons. Despite each neural unit’s relative simplicity, the network’s structure can be so intricate that it may not be fully understood, even by its designer.










