Next we test the performance of our supervised models. We take the best configurations from cref-[[💡Hyperparameter Tuning]], trained and tuned on the gls-ISE trade data, and evaluate their performance on the gls-ise and and gls-cboe test sets. Cref-tab-accuracies-supervised summarizes our results and compares the so-obtained results with state-of-the art algorithms from the literature.

![[results-supervised.png]]
Accuracies of Supervised Approaches On \glsentryshort{CBOE} and \glsentryshort{ISE}. This table reports the accuracy of glspl-gbrt and Transformers for the three different feature sets on the gls-ise and gls-cboe dataset. The improvement is estimated as the absolute change in accuracy between the classifier and the benchmark. For feature set classical, $\operatorname{gsu}_{\mathrm{small}}$ is the benchmark and otherwise $\operatorname{gsu}_{\mathrm{large}}$. Models are trained on the gls-ISE training set. The best classifier per dataset is in **bold**. 

**Finding 3: Supervised Learning outperforms Rule-Based Classifier**

Both model architectures consistently outperform their respective benchmarks on the \gls{ISE} and \gls{CBOE} datasets, achieving state-of the art performance in option trade classification assuming equal data requirements. Thereby, Transformers dominate the gls-ise sample when trained on quotes and trade prices reaching percentage-63.78 and percentage 66.18 on the gls-cboe sample outperforming previous approaches by percentage-3.73 and percentage-5.44.  Additional trade size features push the accuracy to percentage-72.85 for the gls-ise sample and percentage-72.15 for the gls-cboe sample. Gradient boosting outperforms all other approaches when trained on additional option features. While absolute improvements in accuracy are modest on the smallest feature set over $\operatorname{gsu}_{\mathrm{small}}$, improvements are more substantial for larger feature sets ranging between percentage-4.73  to percentage-7.86 over $\operatorname{gsu}_{\mathrm{large}}$. Specifically, the addition of trade size-related features positively contribute to the performance. The results can be further improved by allowing for retraining on the validation set. Results are documented in the appendix. Relative to related works performing trade classification using machine learning, the improvements are strong, as a direct comparison with appendix-table reveals.
  
Visually, the performance differences between gradient boosting and transformers on the same feature sets are minor, consistent with previous studies ([[@grinsztajnWhyTreebasedModels2022]]) and ([@gorishniyEmbeddingsNumericalFeatures2022]]). These studies conclude, generally for tabular modelling, that neither Transformers nor gls-gbrt are universally superior. To formally test, whether differences between both classifiers are significant, we construct contingency tables and pair-wise compare predictions using McNemar's test ([[@mcnemarNoteSamplingError1947]]153--157). We formulate the null hypothesis that both classifiers disagree by the same amount.
The procedure is different to ([[@odders-whiteOccurrenceConsequencesInaccurate2000]] 267), who uses contingency tables of rule-based methods and true labels. Here, contingency tables are used to pair-wise compare the predictions of gls-gbrt against Transformers. We study the distribution against the true label as part of our robustness checks.

**Finding 4: None of the supervised classifiers performs better?**

Based on table-contingency, we can conclude.

table-with contingency tables and p values statistic
footnote-what is link to accuracy 

todo-Discuss results
Null hypothesis: A and B have the same error rate.

![[mcnemar-ise.png]]

---

https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html

## Comparing Two Models with the McNemar Test [#](https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html#comparing-two-models-with-the-mcnemar-test) [](https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html#comparing-two-models-with-the-mcnemar-test)

So, instead of using the “difference of proportions” test, Dietterich (Dietterich, 1998) found that the McNemar test is to be preferred. The McNemar test, introduced by Quinn McNemar in 1947 (McNemar 1947), is a non-parametric statistical test for paired comparisons that can be applied to compare the performance of two machine learning classifiers.

Often, McNemar’s test is also referred to as “within-subjects chi-squared test,” and it is applied to paired nominal data based on a version of 2x2 confusion matrix (sometimes also referred to as _2x2 contingency table_) that compares the predictions of two models to each other (not be confused with the typical confusion matrices encountered in machine learning, which are listing false positive, true positive, false negative, and true negative counts of a single model). The layout of the 2x2 confusion matrix suitable for McNemar’s test is shown in the following figure:

![McNemar Table Layout](https://sebastianraschka.com/images/blog/2018/model-evaluation-selection-part4/mcnemar-table-layout.png)

Given such a 2x2 confusion matrix as shown in the previous figure, we can compute the accuracy of a _Model 1_ via (A+B)/(A+B+C+D)(�+�)/(�+�+�+�), where A+B+C+D�+�+�+� is the total number of test examples n�. Similarly, we can compute the accuracy of Model 2 as (A+C)/n(�+�)/�. The most interesting numbers in this table are in cells B and C, though, as A and D merely count the number of samples where both _Model 1_ and _Model 2_ made correct or wrong predictions, respectively. Cells B and C (the off-diagonal entries), however, tell us how the models differ. To illustrate this point, let us take a look at the following example:

![McNemar Table Layout](https://sebastianraschka.com/images/blog/2018/model-evaluation-selection-part4/mcnemar-table-example1.png)

In both subpanels, A and B, the accuracy of _Model 1_ and _Model 2_ are 99.6% and 99.7%, respectively.

- Model 1 accuracy subpanel A: (9959+11)/10000×100%=99.7%(9959+11)/10000×100%=99.7%
- Model 1 accuracy subpanel B: (9945+25)/10000×100%=99.7%(9945+25)/10000×100%=99.7%
- Model 2 accuracy subpanel A: (9959+1)/10000×100%=99.6%(9959+1)/10000×100%=99.6%
- Model 2 accuracy subpanel B: (9945+15)/10000×100%=99.6%(9945+15)/10000×100%=99.6%

Now, in subpanel A, we can see that _Model 1_ got 11 predictions right that _Model 2_ got wrong. Vice versa, _Model 2_ got one prediction right that _Model 1_ got wrong. Thus, based on this 11:1 ratio, we may conclude, based on our intuition, that _Model 1_ performs substantially better than _Model 2_. However, in subpanel B, the _Model 1_:_Model 2_ ratio is 25:15, which is less conclusive about which model is the better one to choose. This is a good example where McNemar’s test can come in handy.

In McNemar’s Test, we formulate the null hypothesis that the probabilities p(B)�(�) and p(C)�(�) – where B� and C� refer to the confusion matrix cells introduced in an earlier figure – are the same, or in simplified terms: None of the two models performs better than the other. Thus, we might consider the alternative hypothesis that the performances of the two models are not equal.

The McNemar test statistic (“chi-squared”) can be computed as follows:

χ2=(B−C)2B+C.�2=(�−�)2�+�.

After setting a significance threshold, for example, α=0.05�=0.05, we can compute a p-value – assuming that the null hypothesis is true, the p-value is the probability of observing the given empirical (or a larger) χ2�2-squared value. If the p-value is lower than our chosen significance level, we can reject the null hypothesis that the two models’ performances are equal.

Since the McNemar test statistic, χ2�2, follows a χ2�2 distribution with one degree of freedom (assuming the null hypothesis and relatively large numbers in cells B and C, say > 25), we can now use our favorite software package to “look up” the (1-tail) probability via the χ2�2 probability distribution with one degree of freedom.

If we did this for scenario B in the previous figure (χ2=2.5�2=2.5), we would obtain a p-value of 0.1138, which is larger than our significance threshold, and thus, we cannot reject the null hypothesis. Now, if we computed the p-value for scenario A (χ2=8.3�2=8.3), we would obtain a p-value of 0.0039, which is below the set significance threshold (α=0.05�=0.05) and leads to the rejection of the null hypothesis; we can conclude that the models’ performances are different (for instance, _Model 1_ performs better than _Model 2_).

Approximately one year after Quinn McNemar published the McNemar Test (McNemar 1947), Allen L. Edwards (Edwards 1948) proposed a continuity corrected version, which is the more commonly used variant today:

χ2=(|B−C|−1)2B+C.�2=(|�−�|−1)2�+�.

In particular, Edwards wrote:

> This correction will have the apparent result of reducing the absolute value of the difference, [B - C], by unity.

According to Edwards, this continuity correction increases the usefulness and accuracy of McNemar’s test if we are dealing with discrete frequencies and the data is evaluated regarding the chi-squared distribution.

A function for using McNemar’s test is implemented in MLxtend (Raschka, 2018): [http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/](http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/).






Counter-intuitively, performance improvements are highest for the gls-cboe dataset, despite the models being trained on gls-ise data. Part of this is due to a weaker benchmark performance, but also due to a considerably stronger accuracy of classifiers on the smallest and mid-sized feature set. This result is counter-intuitive, as one would expect a degradation between sets, assuming exchange-specific trading patterns and require exploration in greater detail.

**Finding 4: Fee-Structures Affect Classifier Performance**
tbd



Next, we estimate 

After performing the test and finding a significant result, it may be useful to report an effect statistical measure in order to quantify the finding. For example, a natural choice would be to report the odds ratios, or the contingency table itself, although both of these assume a sophisticated reader.

Similar to ([[@odders-whiteOccurrenceConsequencesInaccurate2000]] 267) we further break down the results by calculating confusion matrices as visualized cref-ise-confusion and cref-cboe-confusion. Based on the 

and estimate McNemar's

This allows more detailed analysis than simply observing the proportion of correct classifications (accuracy). Accuracy will yield misleading results if the data set is unbalanced; that is, when the numbers of observations in different classes vary greatly.

![[confusion-matrix-ise.png]]
(ise)
![[confusion-matrix-cboe.png]]
(cboe)


- The recommendation of the McNemar’s test for models that are expensive to train, which suits large deep learning models.
- How to transform prediction results from two classifiers into a contingency table and how the table is used to calculate the statistic in the McNemar’s test.
- How to calculate the McNemar’s test in Python and interpret and report the result.

Specifically, Dietterich’s study was concerned with the evaluation of different statistical hypothesis tests, some operating upon the results from resampling methods. The concern of the study was low [Type I error](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors), that is, the statistical test reporting an effect when in fact no effect was present (false positive).

Statistical tests that can compare models based on a single test set is an important consideration for modern machine learning, specifically in the field of deep learning.

The default assumption, or null hypothesis, of the test is that the two cases disagree to the same amount. If the null hypothesis is rejected, it suggests that there is evidence to suggest that the cases disagree in different ways, that the disagreements are skewed.

Given the selection of a significance level, the p-value calculated by the test can be interpreted as follows:

- **p > alpha**: fail to reject H0, no difference in the disagreement (e.g. treatment had no effect).
- **p <= alpha**: reject H0, significant difference in the disagreement (e.g. treatment had an effect).

We can summarize this as follows:

- **Fail to Reject Null Hypothesis**: Classifiers have a similar proportion of errors on the test set.
- **Reject Null Hypothesis**: Classifiers have a different proportion of errors on the test set.

The confusion


Additianlly roc-curves and comparing classifiers with the McNemar test. 
In pair-wise comparisions,

Table 2 contains a comparison of the true classi"cation (buy or sell) with the classi"cation from each of the three algorithms. Based purely on the percentage FINMAR=38=KGM=VVC 266 E.R. Odders-White / Journal of Financial Markets 3 (2000) 259}28

Table 2 Performance of the algorithms The table below contains a comparison of the true classi"cation (buy or sell) to the classi"cation from the quote (Panel A), the tick (Panel B), and the Lee and Ready algorithms (Panel C). A description of these methods is contained in Section 2 of the text. Each entry contains the number and percentage of transactions in the sample that fall into the respective category. Analyses are based only on transactions for which the true initiator can be determined. Method and classi"cation True buy True sell Number Percent Number Percent Panel A: Quote method vs. true classixcation Quote method: Buy 127,827 40.15 14,997 4.71 Quote method: Sell 13,893 4.36 110,870 34.82 Quote method: Unclassi"ed 26,308 8.26 24,469 7.69 Panel B: Tick method vs. true classixcation Tick method: Buy 134,649 42.29 34,662 10.89 Tick method: Sell 33,379 10.48 115,674 36.33 Panel C: Lee and Ready method vs. true classixcation Lee and Ready method: Buy 144,348 45.34 24,183 7.60 Lee and Ready method: Sell 23,680 7.44 126,153 39.63 of transactions classi"ed correctly, the Lee and Ready method (Panel C) is the most accurate.

Our results document a strong performance of supervised classifiers for the task of option trade classification, but leave open whether the performance is consistent across sub samples. Following common track in literature we employ robustness checks for all models in cref-[[🏅Robustness]]. 

Next, we further break down the results by calculating 

The configuration of this model is listed in the bottom line of Table 3
To put these results in perspective, our best model using additional trade size and option features improves over the frequently employed tick rule, quote rule, and gls-lr algorithm by more than (74.12 - 57.10) on the gls-ISE sample. 

Expectedly,



The results can be further improved by allowing for retraining / training on cboe. Document results in the appendix.

In summary, our supervised methods establish a new state-of-the-art in option trade classification. Our approach achieves full coverage and outperforms all previously reported classification rules in terms of accuracy. We perform additional robustness checks in cref-robustness to verify performance is not biased towards specific sub-samples. 


Following 

In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.

We test the subspace update strategies on our real-world data sets and calculate the baseline using GMD with M = 1000. The size of the sliding window depends on the data set and is given in the respective figure. To calculate the contrast, we run M MC simulations. For the evaluation of the Random Strategy, we set p = 1 D . The expected number of updated subspaces is then E(|Rt |) = 1 and thus comparable to the other strategies. Setting p to a larger value is computationally more expensive because more subspaces are updated, giving the Random Strategy an advantage. For the -Greedy Strategy, we set  = 0.5. For the SBTS Strategy, we set  = 0.5 and γ = 0.99. Appendix 6.2.2 and Appendix 6.2.3 show more parameter settings. Appendix 6.2.4 shows the results of the CBTS Strategy, which we omit here because of its poor performance.

Next we perform (coarse grained to finegrained)
- similar to odders-white calculate confusion matrix
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



https://machinelearningmastery.com/mcnemars-test-for-machine-learning/



Employ Friedman test
To assess significance of the above results, we ran 3 statistical tests: the Friedman test [34, 35], the Nemenyi test [36], and the 1-sided Wilcoxon signed-rank test [37], all described in Demšar [38]. The Friedman test compares the mean ranks of several algorithms run on several datasets. The null hypothesis assumes that all algorithms are equivalent, i.e., their rank should be equal. Table A2 shows that the null hypothesis is rejected, with P-values much less than the 0.05 level for the sizes 2,500, 10,000, and 25,000. This indicates that ≥1 algorithm has significantly different performances from 1 other on these sizes.




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
