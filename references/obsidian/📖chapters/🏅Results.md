
- Perform an error analysis. For which classes does CatBoost do so poorly? See some ideas here. https://elitedatascience.com/feature-engineering-best-practises



Akin to selecting the machine learning classifiers, we determine our classical baselines on the gls-ISE validation set. This guarantees a challenging baselines, while maintaining consistency between both paradigms. For the same reason, baselines are kept constant in the transfer setting on the gls-CBOE sample. Solely for reference, we also report accuracies of the gls-tick, gls-quote, gls-lr, due to their widespread adoption in finance.

(insert table here)

Table-x reports the accuracies of common trade classification rules over the entire validation set and broken down by the trade price's location relative to the quotes. The tick test applied to trade prices at the trading venue, performs worst with an accuracy below a random guess. Against this backdrop, we estimate all hybrid rules involving tick rule, over all exchanges ($\operatorname{tick}_{\text{all}}$). From all classical rules, a combination of the quote rule ($\operatorname{quote}_{\text{nbbo}} \to \operatorname{quote}_{\text{ex}}$), where the quote rule is first applied to the gls-NBBO and then to quotes of the gls-ISE quotes, performs best. The rule can be estimated using features from FS1, which qualifies it as a benchmark. Also, it is commonly studied in literature, as previously by ([[@muravyevOptionsTradingCosts2020]]).

By extension, we also estimate rules combinations involving overrides from the tradesize rule ($\operatorname{tsize}$) and the depth rule ($\operatorname{depth}$) on the top-performing baselines of FS1. Consistent with the recommendation of ([[@grauerOptionTradeClassification2022]]14), we find that a deep combination of the $\operatorname{tsize}_{\text{ex}} \to \operatorname{quote}_{\text{nbbo}} \to \operatorname{quote}_{\text{ex}} \to \operatorname{depth}_{\text{nbbo}} \to \operatorname{depth}_{\text{ex}} \to \operatorname{rtick}_{\text{all}}$ achieves the highest validation. For brevity, we refer to this hybrid as the gls-GSU method. Much of the performance improvements is owed to the trade size and depth rules, which reduce the dependence on the reverse tick test as a last resort and provide overrides for trades at the quotes, improving validation accuracy to percent-68.8359. 

In absence of other suitable baselines, we also the GSU method for FS3, even if it doesn't utilise option-specific features.

Calculate average rank


![[performance-degradations.png]]


- Think about using ensembles
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


Visualisation of tables with columns: https://tex.stackexchange.com/questions/174876/formatting-table-with-siunitx-problem-with-parentheses-and-signs

callibrated probas shouldnt be much of a problem, as we optimise for probabilities directly https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf