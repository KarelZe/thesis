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

Authors discuss an ideal Bayesian trade classification approach. Authors view the problem of trade classification similar to Bayesian statistican with priors on the unoverservable information (buy or sell indicator), who is trying to extract trading intentions from observable trade date. (found in [[@boweNewClassicalBayesian]] (do not cite but interesting to look at)) -> As this probabilistic view is similar to a probabilistic classifier it could be used to motivate my own work.

“A Bayesian statistician would start with a prior on the unobservable information, observe the data, and use a likelihood function to update his or her prior to form a posterior on the underlying information. This is not what a tick rule does. It classifies a trade as a buy if the previous price is below the current price, a sell, if it is above. The bulk volume approach, by contrast, can be thought of as assigning a posterior probability to a trade being a buy or sell, an approach closer conceptually to Bayes’ rule.” ([Easley et al., 2016, p. 270](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=2&annotation=8WU3R2SV)) “Tick: T ( ) = 1 if > 0 and T ( ) = 0 if < 0, and” ([Easley et al., 2016, p. 272](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=4&annotation=E8GXDD5Y))

“We consider three methodologies to assign a probability that the underlying trade type was a buy or a sell given the observation of a single draw of : Bayes’ rule, the tick rule, and BVC specialized to a single observation. The tick rule assigns probability one or zero to the trade having been a buy.” ([Easley et al., 2016, p. 272](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=4&annotation=E9GPBVPP))

“Using a statistical model, we investigate the errors that arise from a tick rule approach and the bulk volume approach, relative to a Bayesian approach. We show that when the noise in the data is low, tick rule errors can be relatively low, and over some regions the tick rule can perform better than the bulk volume approach. When noise is substantial, the bulk volume approach can outperform a tick rule and permit more accurate sorting of the data.” ([Easley et al., 2016, p. 270](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=2&annotation=VDMJDEGC))

“Much of market microstructure analysis is built on the concept that traders learn from market data. Some of this learning is prosaic, such as inferring buys and sells from trade execution. Other learning is more complex, such as inferring underlying new information from trade executions. In this paper, we investigate the general issue of how to discern underlying information from trading data. We examine the accuracy and efficacy of three methods for classifying trades: the tick rule, the aggregated tick rule, and the bulk volume classification methodology. Our results indicate that the tick rule is a reasonably good classifier of the aggressor side of trading, both for individual trades and in aggregate. Bulk volume is shown to also be reasonably accurate for classifying buy and sell trades, but, unlike the tick-based approaches, it can also provide insight into other proxies for underlying information.” ([Easley et al., 2016, p. 284](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=16&annotation=VC98DC2N))









