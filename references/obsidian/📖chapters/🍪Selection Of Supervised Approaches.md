

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

- Discuss why probabilities of gradient boosted trees might be missleading [[@arikTabNetAttentiveInterpretable2020]]
- Problems of tree-based approaches and neural networks in semi-supervised learning. See [[@huangTabTransformerTabularData2020]] or [[@arikTabNetAttentiveInterpretable2020]]and [[@tanhaSemisupervisedSelftrainingDecision2017]]


#gbm #transformer #supervised-learning #deep-learning 

**Notes:**
[[🍪Selection of supervised approaches notes]]

Formally, an "ordinary" classifier is some rule, or function, that assigns to a sample $x$ a class label $\hat{y}$
$$
\hat{y}=f(x)
$$
The samples come from some set $X$ (e.g., the set of all documents, or the set of all images), while the class labels form a finite set $Y$ defined prior to training.
Probabilistic classifiers generalize this notion of classifiers: instead of functions, they are conditional distributions $\operatorname{Pr}(Y \mid X)$, meaning that for a given $x \in X$, they assign probabilities to all $y \in Y$ (and these probabilities sum to one). "Hard" classification can then be done using the optimal decision rule [2]:39-40
$$
\hat{y}=\arg \max _y \operatorname{Pr}(Y=y \mid X)
$$
or, in English, the predicted class is that which has the highest probability.
Binary probabilistic classifiers are also called binary regression models in statistics. In econometrics, probabilistic classification in general is called discrete choice.

Some classification models, such as naive Bayes, logistic regression and multilayer perceptrons (when trained under an appropriate loss function) are naturally probabilistic. Other models such as support vector machines are not, but methods exist to tum them into probabilistic classifiers.


“We assume the reader is familiar with these concepts. For a complete reference see Hastie et al. [2009]. Let us just lay out our notation and say that in our framework we receive from an axiomatic data preparation stage an ordered set of multivariate observations W = (X , y). y is the outcome or target ordered set with individual elements y. Similarly, X and X are the feature-vector ordered set and element, respectively. Components of feature vectors are individual features, denoted x (ordered set) and x (element). Target and feature-vector elements y and X pertaining to the same element of W are said to be W-associated. The modeler’s goal is to infer the value of a target element, from its associated feature-vector element and from a separate group of observations, called the training examples Wtr. The solution to this problem is a model ˆ y = M(X , Wtr). We say that the model’s observational inputs for inferring ˆ y are X and Wtr, and this relation between the various entities in the framework is the base for our discussion.” ([[@kaufmanLeakageDataMining2012]], p. 158)

Formally, we aim to model a target variable $Y \in \mathbb{Y}$ given some feature vector $X \in \mathbb{X}$ based on training data $\left\{\left(x_i, y_i\right)\right\}_{i=1}^n$ that has been sampled according to the joint distribution of $X$ and $Y$. We focus on models in the form of a single-valued scoring function $f: \mathbb{X} \rightarrow \mathbb{R}$. For instance, in regression problems $(\mathbb{Y}=\mathbb{R}), f$ typically models the conditional expectation of the target, i.e., $f(x) \approx E(Y \mid X=x)$, whereas in binary classification problems $(\mathbb{Y}=\{-1,1\}), f$ ty (found here[[@boleyBetterShortGreedy2021]]; do not cite but like their presentation)

Assume that we are given a distribution $\mathcal{D}$ on $\mathcal{X} \times \mathcal{Y}$, where $\mathcal{X}$ is a feature space (typically $\mathbb{R}^k$ ) and $\mathcal{Y}$ is a target space (typically $\mathbb{R}$ for regression or $\{0,1\}$ for classification). ${ }^1$ We are also given a loss function $L(z, y): \mathcal{Z} \times \mathcal{Y} \rightarrow \mathbb{R}$, where $\mathcal{Z}$ is a space of predictions (typically $\mathbb{R}$ or $\{0,1\}$ ). Our goal is to minimize the expected loss $\mathcal{L}(f \mid \mathcal{D}):=\mathbb{E}_{\mathcal{D}} L(f(x), y)$
https://arxiv.org/pdf/2001.07248.pdf


$$
\begin{array}{cl}
x \in \mathcal{X} & \text { Features, typically from } \mathbb{R}^k \\
y \in \mathcal{Y} & \text { Target, typically from } \mathbb{R} \text { or }\{0,1\} \\
z \in \mathcal{Z} & \text { Prediction, typically from } \mathbb{R} \\
\mathcal{D} & \text { Data distribution over } \mathcal{X} \times \mathcal{Y} \\
\mathcal{D}_N=\left\{\left(x_i, y_i\right)\right\}_{i=1}^N & \text { I.i.d. samples from } \mathcal{D}
\end{array}
$$
https://arxiv.org/pdf/2001.07248.pdf

For a given data set with $n$ examples and $m$ features $\mathcal{D}=\left\{\left(\mathbf{x}_i, y_i\right)\right\}\left(|\mathcal{D}|=n, \mathbf{x}_i \in \mathbb{R}^m, y_i \in \mathbb{R}\right)$ to (from [[@chengWideDeepLearning2016]])

Notation
- $\mathbf{x} \in \mathbb{R}^d$ : feature inputs or covariates (usually high dimensional), described by the random variable $X$
- $y \in\{1,2, \ldots, C\}$ : classes (labels), described by the random variable $Y$
- Each instance of data is drawn from a join probability distribution $p^*(X, Y)$
- $p^*(\mathbf{x}, y)$ : True distribution of $\mathbf{x}$ and $y$. Also denoted by $p^*(X, Y)$ or $p^*(X=\mathbf{x}, Y=y)$

- $C$-class Classification problem setting:
- true distribution is assumed to be a discrete distribution over $C$ classes
- observed $y$ is a sample from conditional distribution $p^*(y \mid \mathbf{x})$ or $p^*(Y \mid X=\mathbf{x})$
- Neural networks (discriminative classifiers) try and estimate $p_\theta(y \mid \mathbf{x})$ by fitting $\theta$ using $\mathcal{D}$ (training dataset)
- During deployment, the $\mathrm{NN}$ is evaluated using a dataset $\mathcal{T}$, sampled from a distribution $q(\mathbf{x}, y)$ or $q(X, Y)$

Classical trade classification rule (or at least the ones shown) perform hard classification. Some bulked trade classification algorithms can perform soft classification (See [[@easleyDiscerningInformationTrade2016]]). However, this is not the case for the algorithms working on a trade-per-trade basis. Still, one can derive probabilities. (See [[@easleyDiscerningInformationTrade2016]] for tick rule)

Authors discuss an ideal Bayesian trade classification approach. Authors view the problem of trade classification similar to Bayesian statistican with priors on the unoverservable information (buy or sell indicator), who is trying to extract trading intentions from observable trade date. (found in [[@boweNewClassicalBayesian]] (do not cite but interesting to look at)) -> As this probabilistic view is similar to a probabilistic classifier it could be used to motivate my own work.

“A Bayesian statistician would start with a prior on the unobservable information, observe the data, and use a likelihood function to update his or her prior to form a posterior on the underlying information. This is not what a tick rule does. It classifies a trade as a buy if the previous price is below the current price, a sell, if it is above. The bulk volume approach, by contrast, can be thought of as assigning a posterior probability to a trade being a buy or sell, an approach closer conceptually to Bayes’ rule.” ([Easley et al., 2016, p. 270](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=2&annotation=8WU3R2SV)) “Tick: T ( ) = 1 if > 0 and T ( ) = 0 if < 0, and” ([Easley et al., 2016, p. 272](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=4&annotation=E8GXDD5Y))

“We consider three methodologies to assign a probability that the underlying trade type was a buy or a sell given the observation of a single draw of : Bayes’ rule, the tick rule, and BVC specialized to a single observation. The tick rule assigns probability one or zero to the trade having been a buy.” ([Easley et al., 2016, p. 272](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=4&annotation=E9GPBVPP))

“Using a statistical model, we investigate the errors that arise from a tick rule approach and the bulk volume approach, relative to a Bayesian approach. We show that when the noise in the data is low, tick rule errors can be relatively low, and over some regions the tick rule can perform better than the bulk volume approach. When noise is substantial, the bulk volume approach can outperform a tick rule and permit more accurate sorting of the data.” ([Easley et al., 2016, p. 270](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=2&annotation=VDMJDEGC))

“Much of market microstructure analysis is built on the concept that traders learn from market data. Some of this learning is prosaic, such as inferring buys and sells from trade execution. Other learning is more complex, such as inferring underlying new information from trade executions. In this paper, we investigate the general issue of how to discern underlying information from trading data. We examine the accuracy and efficacy of three methods for classifying trades: the tick rule, the aggregated tick rule, and the bulk volume classification methodology. Our results indicate that the tick rule is a reasonably good classifier of the aggressor side of trading, both for individual trades and in aggregate. Bulk volume is shown to also be reasonably accurate for classifying buy and sell trades, but, unlike the tick-based approaches, it can also provide insight into other proxies for underlying information.” ([Easley et al., 2016, p. 284](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=16&annotation=VC98DC2N))


“The paper closest to our work is Gorishniy et al. [2021], benchmarking novel algorithms, on 11 tabular datasets. We provide a more comprehensive benchmark, with 45 datasets, split across different settings (medium-sized / large-size, with/without categorical features), accounting for the hyperparameter tuning cost, to establish a standard benchmark.” ([Grinsztajn et al., 2022, p. 2](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=2&annotation=YXJLM6JN)) “FT_Transformer : a simple Transformer model combined with a module embedding categorical and numerical features, created in Gorishniy et al. [2021]. We choose this model because it was benchmarked in a convincing way against tree-based models and other tabular-specific models. It can thus be considered a “best case” for Deep learning models on tabular data.” ([Grinsztajn et al., 2022, p. 5](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=5&annotation=AHYUCL2P))

“MLP-like architectures are not robust to uninformative features In the two experiments shown in Fig. 4, we can see that removing uninformative features (4a) reduces the performance gap between MLPs (Resnet) and the other models (FT Transformers and tree-based models), while adding uninformative features widens the gap. This shows that MLPs are less robust to uninformative features, and, given the frequency of such features in tabular datasets, partly explain the results from Sec. 4.2.” ([Grinsztajn et al., 2022, p. 7](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=7&annotation=TQSG939L))

“Why are MLPs much more hindered by uninformative features, compared to other models? One answer is that this learner is rotationally invariant in the sense of Ng [2004]: the learning procedure which learns an MLP on a training set and evaluate it on a testing set is unchanged when applying a rotation (unitary matrix) to the features on both the training and testing set. On the contrary, tree-based models are not rotationally invariant, as they attend to each feature separately, and neither are FT Transformers, because of the initial FT Tokenizer, which implements a pointwise operation theoretical link between this concept and uninformative features is provided by Ng [2004], which shows that any rotationallly invariant learning procedure has a worst-case sample complexity that grows at least linearly in the number of irrelevant features. Intuitively, to remove uninformative features, a rotationaly invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost.” ([Grinsztajn et al., 2022, p. 8](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=8&annotation=W6LGGVAC))


**Criteria:** 💂‍♀️
- **performance** That is, approach must deliver state-of-the-art performance in similar problems.
- **interpretability** Classical approaches are transparent in a sense that we know how the decision was derived. In the best case try to aim for local and global interpretability. Think about how interpretability can be narrowed down? Note supervisor wants to see if her features are also important to the model. 

**Why tabular data is hard:**
- “Tabular data is a database that is structured in a tabular form. It arranges data elements in vertical columns (features) and horizontal rows (samples)” ([Yoon et al., 2020, p. 1](zotero://select/library/items/XSYUS7JZ)) ([pdf](zotero://open-pdf/library/items/78GQQ36U?page=1&annotation=8MAKL2B9))
- Challenges of learning of tabular data can be found in [[@borisovDeepNeuralNetworks2022]] e. g. both 

![[decision-process-supervised-semi.jpg]]


**Coarse grained selection:**
- Show that there is a general concensus, that gradient boosted trees and neural networks work best. Show that there is a great bandwith of opinions and its most promising to try both. Papers: [[@shwartz-zivTabularDataDeep2021]]
- selection is hard e. g., in deep learning, as there are no universal benchmarks and robust, battle tested approaches for tabular data compared to other data sources. (see [[@gorishniyRevisitingDeepLearning2021]])
- reasons why deep learning on tabular data is challenging [[@shavittRegularizationLearningNetworks2018]] (use more as background citation)
- Taxonomy of approaches can be found in [[@borisovDeepNeuralNetworks2022]] 
![[tabular-learning-architectures.png]]

- Perform a wide (ensemble) vs. deep (neural net) comparison. This is commonly done in literature. Possible papers include:
	- [[@gorishniyRevisitingDeepLearning2021]] compare DL models with Gradient Boosted Decision Trees and conclude that there is still no universally superior solution.
	- For "shallow" state-of-the-art are ensembles such as GBMs. (see [[@gorishniyRevisitingDeepLearning2021]])
	- Deep learning for tabular data could potentially yield a higher performance and allow to combine tbular data with non-tabular data such as images, audio or other data that can be easily processed with deep learning. [[@gorishniyRevisitingDeepLearning2021]]
	- Despite growing number of novel (neural net) architectures, there is still no simple, yet reliable solution that achieves stable performance across many tasks. 
	- [[@arikTabNetAttentiveInterpretable2020]] Discuss a number of reasons why decisiion tree esembles dominate neural networks for tabular data.
	- [[@huangTabTransformerTabularData2020]] argue that tree-based ensembles are the leading approach for tabular data. The base this on the prediction accuracy, the speed of training and the ability to interpret the models. However, they list sevre limitations. As such they are not suitabl efor streaming data, multi-modality with tabular data e. g. additional image date and do not support semi-supervised learning by default.
- Choose neural network architectures, that are tailored towards tabular data.



**Camparison:**
- large number of datapoints -> Transformers are data hungry (must be stated in the [[@vaswaniAttentionAllYou2017]] paper)
- Nice formulation and overview of the dominance of GBT and deep learning is given in [[@levinTransferLearningDeep2022]]
- for use of transformer-based models in finance see[[@zouStockMarketPrediction2022]]
- Non-parametric model of [[@kossenSelfAttentionDatapointsGoing2021]]

- Sophisticated neural network architectures might not be required, but rather a mix of regularization approaches to regularize MLPs [[@kadraWelltunedSimpleNets2021]].
- See [[@huangTabTransformerTabularData2020]] that point out common problems of comparsions between gbts and dl.

“An extensive line of work on tabular deep learning aims to challenge the dominance of GBDT models. Numerous tabular neural architectures have been introduced, based on the ideas of creating differentiable learner ensembles [55, 29, 77, 43, 8], incorporating attention mechanisms and transformer architectures [64, 26, 6, 34, 65, 44], as well as a variety of other approaches [70, 71, 10, 42, 23, 61]. However, recent systematic benchmarking of deep tabular models [26, 63] shows that while these models are competitive with GBDT on some tasks, there is still no universal best method. Gorishniy et al. [26] show that transformer-based models are the strongest alternative to GBDT and that ResNet and MLP models coupled with a strong hyperparameter tuning routine [2] offer competitive baselines. Similarly, Kadra et al. [40] find that carefully regularized MLPs are competitive. In a follow-up work, Gorishniy et al. [27] show that transformer architectures equipped with advanced embedding schemes for numerical features bridge the performance gap between deep tabular models and GBDT” (Levin et al., 2022, p. 3)

**GBM:** There are several established libraries such as catboost, XGBoost and LightGBM, (that differ in e. g., the growing policy of trees, handling missing values or the calculation of gradients. (see papers also see [[@josseConsistencySupervisedLearning2020]]))  Their performance however, doesn't differ much. (found in [[@gorishniyRevisitingDeepLearning2021]] and cited [[@prokhorenkovaCatBoostUnbiasedBoosting2018]])

**Regularization:** “Why are MLPs much more hindered by uninformative features, compared to other models? One answer is that this learner is rotationally invariant in the sense of Ng [2004]: the learning procedure which learns an MLP on a training set and evaluate it on a testing set is unchanged when applying a rotation (unitary matrix) to the features on both the training and testing set. On the contrary, tree-based models are not rotationally invariant, as they attend to each feature separately, and neither are FT Transformers, because of the initial FT Tokenizer, which implements a pointwise operation theoretical link between this concept and uninformative features is provided by Ng [2004], which shows that any rotationallly invariant learning procedure has a worst-case sample complexity that grows at least linearly in the number of irrelevant features. Intuitively, to remove uninformative features, a rotationaly invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost.” ([Grinsztajn et al., 2022, p. 8](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=8&annotation=W6LGGVAC))
“The paper closest to our work is Gorishniy et al. [2021], benchmarking novel algorithms, on 11 tabular datasets. We provide a more comprehensive benchmark, with 45 datasets, split across different settings (medium-sized / large-size, with/without categorical features), accounting for the hyperparameter tuning cost, to establish a standard benchmark.” ([Grinsztajn et al., 2022, p. 2](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=2&annotation=YXJLM6JN)) “FT_Transformer : a simple Transformer model combined with a module embedding categorical and numerical features, created in Gorishniy et al. [2021]. We choose this model because it was benchmarked in a convincing way against tree-based models and other tabular-specific models. It can thus be considered a “best case” for Deep learning models on tabular data.” ([Grinsztajn et al., 2022, p. 5](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=5&annotation=AHYUCL2P))

“MLP-like architectures are not robust to uninformative features In the two experiments shown in Fig. 4, we can see that removing uninformative features (4a) reduces the performance gap between MLPs (Resnet) and the other models (FT Transformers and tree-based models), while adding uninformative features widens the gap. This shows that MLPs are less robust to uninformative features, and, given the frequency of such features in tabular datasets, partly explain the results from Sec. 4.2.” ([Grinsztajn et al., 2022, p. 7](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=7&annotation=TQSG939L))

“Tuning hyperparameters does not make neural networks state-of-the-art Tree-based models are superior for every random search budget, and the performance gap stays wide even after a large number of random search iterations. This does not take into account that each random search iteration is generally slower for neural networks than for tree-based models (see A.2).” ([Grinsztajn et al., 2022, p. 6](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=6&annotation=K2FYJND8)) [[@grinsztajnWhyTreebasedModels2022]]

“Semi-supervised boosting methods have been studied extensively over the past two decades. The success achieved by supervised boosting methods, such as AdaBoost (Freund and Schapire 1997), gradient boosting, and XGBoost (Chen and Guestrin 2016), provides ample motivation for bringing boosting to the semi-supervised setting. Furthermore, the pseudo-labelling approach of self-training and co-training can be easily extended to boosting methods.” (Engelen and Hoos, 2020, p. 391) [[@vanengelenSurveySemisupervisedLearning2020]]


In the case of decision trees, where Pr(_y_|**x**) is the proportion of training samples with label y in the leaf where **x** ends up, these distortions come about because learning algorithms such as [C4.5](https://en.wikipedia.org/wiki/C4.5 "C4.5") or [CART](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees "Predictive analytics") explicitly aim to produce homogeneous leaves (giving probabilities close to zero or one, and thus high [bias](https://en.wikipedia.org/wiki/Bias_of_an_estimator "Bias of an estimator")) while using few samples to estimate the relevant proportion (high [variance](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff "Bias–variance tradeoff")).[[4]](https://en.wikipedia.org/wiki/Probabilistic_classification#cite_note-4)


“Our contributions are as follow: 1. We create a new benchmark for tabular data, with a precise methodology for choosing and preprocessing a large number of representative datasets. We share these datasets through OpenML [Vanschoren et al., 2014], which makes them easy to use. 2. We extensively compare deep learning models and tree-based models on generic tabular datasets in multiple settings, accounting for the cost of choosing hyperparameters.” (Grinsztajn et al., 2022, p. 2)