

![[results-supervised.png]]

Next we perform (coarse grained to finegrained)
- look into accuracy first. Provide detailled breakdown later.
- roc curves
- then move to confusion matrix and McNemar’s test then provide detailled breakdown of results in robustness chapter.
Employ McNemar test. See e. g., [[@raschkaIntroductionLatestTechniques2021]].
- unclear picture which classifiers performs best

- best model hyperparam search
- what is the baseline

Recall from 

- both models establish a new state-of-the-art
- larger feature sets improve performance in particular inclusion of size features as motivated by rule-based innovations. Look into this more detailledly later.
- further performance testing is required
- how do improvements compare to literature?
- emphasize that we used most rigit baseline
- document improvement over default choices e. g. quote rule, tick rule,
- point out advantage of achieving full coverage
- point out advantage that models are even stronger on cboe dataset dispite being learned on ise dataset. Cost of inference is low. Good practical use.

The results can be further improved by allowing for retraining / training on cboe. Document results in the appendix.

Table 1 Panel A (May sample) shows the success rate of our algorithm and other algorithms. Overall, our algorithm outperforms all three competing rules – the LR, tick, and EMO – with improvements of 2.10%, 0.72% and 1.12% over the LR, EMO and tick rules respectively.7 Panel B (May sample) of the same table indicates that our algorithm has a substantial improvement over other algorithms.


https://machinelearningmastery.com/mcnemars-test-for-machine-learning/



Employ Friedman test
To assess significance of the above results, we ran 3 statistical tests: the Friedman test [34, 35], the Nemenyi test [36], and the 1-sided Wilcoxon signed-rank test [37], all described in Demšar [38]. The Friedman test compares the mean ranks of several algorithms run on several datasets. The null hypothesis assumes that all algorithms are equivalent, i.e., their rank should be equal. Table A2 shows that the null hypothesis is rejected, with P-values much less than the 0.05 level for the sizes 2,500, 10,000, and 25,000. This indicates that ≥1 algorithm has significantly different performances from 1 other on these sizes.





**FT-Transformer (10 % of  Data / 10 Trials)**
![[FT-Transformer.png]]


Things get a bit more complicated when you want to use statistical tests to compare more than two models, since doing multiple pairwise tests is a bit like using the test set multiple times — it can lead to overly-optimistic interpretations of significance. Basically, each time you carry out a comparison between two models using a statistical test, there’s a probability that it will discover significant differences where there aren’t any. This is represented by the confidence level of the test, usually set at 95%: meaning that 1 in 20 times it will give you a false positive. For a single comparison, this may be a level of uncertainty you can live with. However, it accumulates. That is, if you do 20 pairwise tests with a confidence level of 95%, one of them is likely to give you the wrong answer. This is known as the multiplicity effect, and is an example of a broader issue in data science known as data dredging or p-hacking — see [Head et al., 2015]. To address this problem, you can apply a correction for multiple tests. The most common approach is the Bonferroni correction, a very simple method that lowers the significance threshold based on the number of tests that are being carried out — see [Salzberg, 1997] for a gentle introduction. However, there are numerous other approaches, and there is also some debate about when and where these corrections should be applied; for an 1 accessible overview, see [Streiner, 2015]. (from [[@lonesHowAvoidMachine2022]])


Broadly speaking, there are two categories of tests for comparing individual ML models. The first is used to compare individual model instances, e.g. two trained decision trees. For example, McNemar’s test is a fairly common choice for comparing two classifiers, and works by comparing the classifiers’ output labels for each sample in the test set (so do remember to record these). The second category of tests are used to compare two models more generally, e.g. whether a decision tree or a neural network is a better fit for the data. These require multiple evaluations of each model, which you can get by using cross-validation or repeated resampling (or, if your training algorithm is stochastic, multiple repeats using the same data). The test then compares the two resulting distributions. Student’s T test is a common choice for this kind of comparison, but it’s only reliable when the distributions are normally distributed, which is often not the case. A safer bet is Mann-Whitney’s U test, since this does not assume that the distributions are normal. For more information, see [Raschka, 2020] and [Carrasco et al., 2020]. Also see Do correct for multiple comparisons and Do be careful when reporting statistical significance. (from [[@lonesHowAvoidMachine2022]])

“One way to achieve better rigour when evaluating and comparing models is to use multiple data sets. This helps to overcome any deficiencies associated with individual data sets (see Don’t always believe results from community benchmarks) and allows you to present a more complete picture of your model’s performance. It’s also good practise to report multiple metrics for each data set, since different metrics can present different perspectives on the results, and increase the transparency of your work. For example, if you use accuracy, it’s also a good idea to include metrics that are less sensitive to class imbalances (see Don’t use accuracy with imbalanced data sets). If you use a partial metric like precision, recall, sensitivity or specificity, also include a metric that gives a more complete picture of your model’s error rates. And make sure it’s clear which metrics you are using. For instance, if you report F-scores, be clear whether this is F1, or some other balance between precision and recall. If you report AUC, indicate whether this is the area under the ROC curve or the PR curve. For a broader discussion, see [Blagec et al., 2020].” (Lones, 2022, p. 13) [[@lonesHowAvoidMachine2022]]

![[visualise_results.png]]
(found in [[@jurkatisInferringTradeDirections2022]] )


look into [[@lonesHowAvoidMachine2022]]

[[@gorishniyRevisitingDeepLearning2021]] vary the random seed of the best configuration (see their NIPS talk https://slideslive.com/38968794/revisiting-deep-learning-models-for-tabular-data?ref=recommended)

For visualising across images and tables, one could adapt the following logic:
![[viz_of_results.png]]


Investigate the confidence of predictions. See intuition here: https://www.youtube.com/watch?v=RXMu96RJj_s

Calculate average rank


![[performance-degradations.png]]


- 
- What are the findings? Find appropriate visualisation (e. g., tables, charts)
-  For each tuned configuration, we run 15 experiments with different random seeds and report the performance on the test set. For some algorithms, we also report the performance of default configurations without hyperparameter tuning. [[@gorishniyRevisitingDeepLearning2021]]
- divide sample into zero ticks and non-zero ticks and see how the accuracy behaves. This was e. g. done in [[@finucaneDirectTestMethods2000]]. See also this paper for reasoning on zero tick and non-zero tick trades.
- perform friedman test to compare algorithms. (see [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- See [[@odders-whiteOccurrenceConsequencesInaccurate2000]] she differentiates between a systematic and non-systematic error and studies the impact on the results in other studies. She uses the terms bias and noise. She also performs several robustness cheques to see if the results can be maintained at different trade sizes etc.
- [[@huyenDesigningMachineLearning]] suggest to tet for fairness, calibration, robustness etc. through:
	- perturbation: change data slightly, add noise etc.
	- invariance: keep features the same, but change some sensitive information
	- Directional expectation tests. e. g. does a change in the feature has a logical impact on the prediction e. g. very high bid (**could be interesting!**)
- adhere to http://www.sigplan.org/Resources/EmpiricalEvaluation/
- Visualise learnt embeddings for categorical data as done in [[@huangTabTransformerTabularData2020]]. 



Before proceeding to a presentation of the hypotheses to be tested and the test results, our primary test for goodness-of-fit is the chi-square test, $\chi^2$. We also use the $G$-test, which is also known as a (log-) likelihood ratio test, as an alternative test since the chi-square test is simply an approximation to the $G$-test for convenient manual computation and the $G$-test is based on the multinomial distribution without using the normal distribution approximation. The $\chi^2$ and $G$-test statistics are computed as: ${ }^{16}$
$$
\chi_{(r-1)(c-1)}^2=\sum_{i, j} \frac{\left(O_{i j}-E_{i j}\right)^2}{E_{i j}}, \quad \text { and } G=2 \sum_{i, j} o_{i j} \cdot \ln \left(\frac{O_{i j}}{E_{i j}}\right) \text {, }
$$
where $O_{i j}$ and $E_{i j}$ are the observed and expected frequencies for cell $i, j$, respectively, in the contingency table; In is the natural logarithm; and the sum is taken over all non-empty cells. (from [[@aktasTradeClassificationAccuracy2014]]. Not sure why they use it.)


Interesting adversarial examples: https://arxiv.org/pdf/1705.07263.pdf


- Unknown's are these where bid (ex) or ask (ex) is NaN. Grauer et al don't report these separately. They must be included somewhere else.
- Makes sense that unknowns are close to 50 % for e. g., quote rule (ex). 
- Stacking adds robustness. It looks suspicious that combinations e. g., quote + quote, GSU reach highest classification accuracies.

% TODO: These proxies have in common that they factor in the order book imbalance the relative depth quoted at the best bid and ask prices. If traders care about transaction costs, the relatively wide ask-side spread deters buyers, whereas the tight bid-side spread may attract sellers. There are then more traders submitting market orders at the bid side, and the true effective spread is, on average, smaller than the average midpoint effective spread.

% TODO: Derive in greater detail why orderbook imbalance makes sense! See my notes from Hagströmer

For example, if the 50,000 transactions misclassi"ed by the Lee and Ready method constitute a representative cross-section of the entire sample, then the misclassi"cation will simply add noise to the data. In this case, the 85% accuracy rate is quite good. If, on the other hand, the Lee and Ready method systematically misclassi"es certain types of transactions, a bias could result.

We report the accurac

- Perform an error analysis. For which classes does CatBoost do so poorly? See some ideas here. https://elitedatascience.com/feature-engineering-best-practises

Akin to selecting the machine learning classifiers, we determine our classical baselines on the gls-ISE validation set. This guarantees a challenging baselines, while maintaining consistency between both paradigms. For the same reason, baselines are kept constant in the transfer setting on the gls-CBOE sample. Solely for reference, we also report accuracies of the gls-tick, gls-quote, gls-lr, due to their widespread adoption in finance.

(insert table here)

Table-x reports the accuracies of common trade classification rules over the entire validation set and broken down by the trade price's location relative to the quotes. The tick test applied to trade prices at the trading venue, performs worst with an accuracy below a random guess. Against this backdrop, we estimate all hybrid rules involving tick rule, over all exchanges ($\operatorname{tick}_{\text{all}}$). From all classical rules, a combination of the quote rule ($\operatorname{quote}_{\text{nbbo}} \to \operatorname{quote}_{\text{ex}}$), where the quote rule is first applied to the gls-NBBO and then to quotes of the gls-ISE quotes, performs best. The rule can be estimated using features from FS1, which qualifies it as a benchmark. Also, it is commonly studied in literature, as previously by ([[@muravyevOptionsTradingCosts2020]]).

By extension, we also estimate rules combinations involving overrides from the tradesize rule ($\operatorname{tsize}$) and the depth rule ($\operatorname{depth}$) on the top-performing baselines of FS1. Consistent with the recommendation of ([[@grauerOptionTradeClassification2022]]14), we find that a deep combination of the $\operatorname{tsize}_{\text{ex}} \to \operatorname{quote}_{\text{nbbo}} \to \operatorname{quote}_{\text{ex}} \to \operatorname{depth}_{\text{nbbo}} \to \operatorname{depth}_{\text{ex}} \to \operatorname{rtick}_{\text{all}}$ achieves the highest validation. For brevity, we refer to this hybrid as the gls-GSU method. Much of the performance improvements is owed to the trade size and depth rules, which reduce the dependence on the reverse tick test as a last resort and provide overrides for trades at the quotes, improving validation accuracy to percent-68.8359. 

In absence of other suitable baselines, we also the GSU method for FS3, even if it doesn't utilise option-specific features.
