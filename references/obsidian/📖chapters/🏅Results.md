
- Think about using ensembles
- 
- What are the findings? Find appropriate visualization (e. g., tables, charts)
-  For each tuned configuration, we run 15 experiments with different random seeds and report the performance on the test set. For some algorithms, we also report the performance of default configurations without hyperparameter tuning. [[@gorishniyRevisitingDeepLearning2021]]
- divide sample into zero ticks and non-zero ticks and see how the accuracy behaves. This was e. g. done in [[@finucaneDirectTestMethods2000]]. See also this paper for reasoning on zero tick and non-zero tick trades.
- Think about stuying the economic impact of false classification trough portfolio construction as done in [[@jurkatisInferringTradeDirections2022]]
- perform friedman test to compare algorithms. (see [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- See [[@odders-whiteOccurrenceConsequencesInaccurate2000]] she differentiates between a systematic and non-systematic error and studies the impact on the results in other studies. She uses the terms bias and noise. She also performs several robustness checks to see if the results can be maintained at different trade sizes etc.
- [[@huyenDesigningMachineLearning]] suggest to tet for fairness, calibration, robustness etc. through:
	- perturbation: change data slightly, add noise etc.
	- invariance: keep features the same, but change some sensitive information
	- Directional expectation tests. e. g. does a change in the feature has a logical impact on the prediction e. g. very high bid (**could be interesting!**)
- adhere to http://www.sigplan.org/Resources/EmpiricalEvaluation/
- Visualize learned embeddings for categorical data as done in [[@huangTabTransformerTabularData2020]]. 



Before proceeding to a presentation of the hypotheses to be tested and the test results, our primary test for goodness-of-fit is the chi-square test, $\chi^2$. We also use the $G$-test, which is also known as a (log-) likelihood ratio test, as an alternative test since the chi-square test is simply an approximation to the $G$-test for convenient manual computation and the $G$-test is based on the multinomial distribution without using the normal distribution approximation. The $\chi^2$ and $G$-test statistics are computed as: ${ }^{16}$
$$
\chi_{(r-1)(c-1)}^2=\sum_{i, j} \frac{\left(O_{i j}-E_{i j}\right)^2}{E_{i j}}, \quad \text { and } G=2 \sum_{i, j} o_{i j} \cdot \ln \left(\frac{O_{i j}}{E_{i j}}\right) \text {, }
$$
where $O_{i j}$ and $E_{i j}$ are the observed and expected frequencies for cell $i, j$, respectively, in the contingency table; In is the natural logarithm; and the sum is taken over all non-empty cells. (from [[@aktasTradeClassificationAccuracy2014]]. Not sure why they use it.)