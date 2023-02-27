All classical trade classification rules from (cref [[🔢Basic rules]]) perform *discrete classification* and assign a class to the trade. Naturally, a more powerful insight is to not just obtain the most probable class, but also the associated class probabilities for a trade to be a buy or sell. This gives additional insights into the confidence of the prediction, but calls for a *probabilistic classifier*.  

Thus, we frame trade signing as a probabilistic classification problem. This is similar to work of ([[@easleyDiscerningInformationTrade2016]] 272), who retrieve probabilities from *bulked* versions of the tick rule and glsc-bvc algorithm, but on a *trade-per-trade* level. 

In order to maintain compability with 

For notation see: https://web.stanford.edu/~nanbhas/blog/some-unifying-notation/

 $\mathbb{Y} = \{-1,1\}$ 

We introduce some more notation, we will use throughout. We denote our dataset with $N$ samples by $\mathcal{D} = \{(\boldsymbol{x}_n, y_n)\}_{n = 1}^{N}$ . Each tuple $(\boldsymbol{x}, y)$ represents a row in the data set, and consist of the binary classification target $y \in \mathbb{Y}$ with $\mathbb{Y}=\{-1,1\}$ and the vector of features $\boldsymbol{x} = \left\{\boldsymbol{x}_{\text{cat}}, \boldsymbol{x}_{\text{cont}}\right\}$, where $x_{\text{cont}} \in \mathbb{R}^c$ denotes all $c$ numerical features and $\boldsymbol{x}_{\text{cat}}\in \mathbb{R}^{m}$ all $m$ categorical features. We denote the cardinality of the $j$-th feature with $j \in 1, \cdots m$ with $N_{C_j}$.

For the prediction $\hat{y} \in \mathbb{Y}$ with of classical trade classification rules we simply estimate the probability as 


Thus, the classifier returns a probability for 

Classical trade classification rules perform a discrete classification. As such, we assign a probability of $1$ for the predicted class  $\hat{v} \in \mathcal{V}$ and zero otherwise.

if point classifier predicts $\hat{v} \in \mathcal{V}$, associated probabilistic classifier returns $\hat{p}$, with

Thus, we assign 
i.e., we return a distribution that has $100 \%$ probability on our point guess $\hat{v}$, and $0 \%$ on others
$$
\hat{p}(v)= \begin{cases}1 & \text { if } v=\hat{v} \\ 0 & \text { else }\end{cases}
$$
Vice versa, for probability estimate $\hat{p}$  we can retrieve the most probable class in $\mathcal{V}$ as:
$$
\hat{v}=\arg\max_{v \in \mathcal{V}} \hat{p}(v).
$$
Cref-eq and cref-eq allow us to switch between a discrete and probabilistic formulation for  trade classification rules. Since, the class probability estimates are either $0$ or $1$, no insight on the confidence of the prediction is gained. Yet, it provides evaluate classical trade classification rules and probabilistic classifiers in machine learning. In the subsequent section we provide a short discussion to select state-of-the-art classifiers, which we consider for our empirical study.


i.e., the value in $\mathcal{V}$ that has highest probability
called a maximum likelihood classifier
extends to a list classifier, by giving values sorted by probability, largest to smallest

In order to maintain 

Due to the tabular nature of the data, with features arranged in a row-column fashion, the token embedding (see chapter [[🛌Token Embedding]]) is replaced for a *column embedding*. Also the notation needs to be adapted to the tabular domain. We denote the data set with $D:=\left\{\left(\mathbf{x}_k, y_k\right) \right\}_{k=1,\cdots N}$ identified with $\left[N_{\mathrm{D}}\right]:=\left\{1, \ldots, N_{\mathrm{D}}\right\}$.  Each tuple $(\boldsymbol{x}, y)$ represents a row in the data set, and consist of the binary classification target $y \in \mathbb{R}$ and the vector of features $\boldsymbol{x} = \left\{\boldsymbol{x}_{\text{cat}}, \boldsymbol{x}_{\text{cont}}\right\}$, where $x_{\text{cont}} \in \mathbb{R}^c$ denotes all $c$ numerical features and $\boldsymbol{x}_{\text{cat}}\in \mathbb{R}^{m}$ all $m$ categorical features. We denote the cardinality of the $j$-th feature with $j \in 1, \cdots m$ with $N_{C_j}$.


https://medium.com/@oded.kalev/comparing-classifiers-using-roc-9a9d8c9c819b


This provides additional insights on the uncertainty of the classifier.
This is particularily appealing for cases, if the assigned class is associated with uncertainty


(See [[@easleyDiscerningInformationTrade2016]]). However, this is not the case for the algorithms working on a trade-per-trade basis. Still, one can derive probabilities

Besides the predicted class, it would also 

A more powerful view 
We consider three methodologies to assign a probability that the underlying trade type was a buy or a sell given the observation of a single draw of :
In contrast  

meaning they directly assign a class to the trade / classify the trade to be buyer- or seller-initiated.  In contrast  

In a similar spirit,

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










