## Trade Classification Rules

We now estimate the accuracy of classical trade classification rules on the gls-ise and gls-cboe sample. We estimate the performance of the tick and quote rule, as well as the gls-LR algorithm, gls-EMO rule and gls-CLNV method in their classical and reversed formulation. Additionally, we consider two stacked combinations of ([[@grauerOptionTradeClassification2022]]) due to their state-of-the-art-performance on the validation set, as derived in cref-[[💡Hyperparameter Tuning]]. Namely, $\operatorname{quote}_{\mathrm{nbbo}} \to \operatorname{quote}_{\mathrm{ex}} \to \operatorname{rtick}_{\mathrm{all}}$ and $\operatorname{tsize}_{\mathrm{ex}} \to \operatorname{quote}_{\mathrm{nbbo}} \to \operatorname{quote}_{\mathrm{ex}} \to \operatorname{depth}_{\mathrm{nbbo}} \to \operatorname{depth}_{\mathrm{ex}} \to \operatorname{rtick}_{\mathrm{all}}$ or in short $\operatorname{gsu}_{\mathrm{small}}$ and $\operatorname{gsu}_{\mathrm{large}}$. 

We report in cref-table-ise accuracies for the entire data set and separate subsets spanning the periods of train, validation, and test set. Doing so, allows us to compare against previous works, but also provide meaningful estimates on the test set relevant for benchmarking purposes. 

Our results are approximately similar to ([[@grauerOptionTradeClassification2022]]29-33). Minor deviations exist, which can linked to differences in handling of unclassified trades and non-positive spreads, as well divergent implementations of the depth rule.-footnote(Correspondence with the author.)

From all rules, tick-based algorithms perform worst when applied to trade prices at the trading venue with accuracies of a random guess, percentage-49.67 or percentage-51.47.  For comparison, a simple majority vote would achieve percentage-51.40 accuracy. The application to trade prices at the inter-exchange level marginally improves over a random / dummy classification, achieving accuracies of percentage-55.25 for the reversed tick test. Due to the poor performance, of tick-based algorithms at the exchange level, we estimate all hybrids with $\operatorname{tick}_{\mathrm{all}}$ or $\operatorname{rtick}_{\mathrm{all}}$.

Quote-based algorithms, outperform tick-based algorithms delivering accuracy up to percentage-63.71, when estimated on the gls-NBBO. The superiority of quote-based algorithms in option trade classification has previously been documented in ([[@savickasInferringDirectionOption2003]]) and ([[@grauerOptionTradeClassification2022]]). 

The performance of hybrids, such as the gls-LR algorithm, hinges with the reliance on the tick test. Thus, the gls-emo rules and to a lesser extent the gls-clnv rules perform worst, achieving accuracies between percentage-55.42 and percentage-57.57. In turn, variants of the gls-LR, which uses the quote rule for most trades, is among the best performing algorithms. By extension, $\operatorname{gsu}_{\mathrm{small}}$ further reduces the dependence on tick-based methods through the successive applications of quote rules, here $\operatorname{quote}_{\mathrm{nbbo}} \to \operatorname{quote}_{\mathrm{ex}}$.

Notably, the combination of ([[@grauerOptionTradeClassification2022]]33) including overrides from the trade size and depth rules performs best, achieving percentage-67.20 accuracy on the gls-ise test set and percentage-75.03 on the entire dataset. Yet, the performance deteriorates most sharply between sets.

We also document the accuracies on the gls-cboe dataset in cref-table. Identical to 

Aside from these high-level observations, we focus three findings in greater detail. 

**Finding 1: Accuracy of tick-based algorithms is downward-biased by missingness**
- grauer et al trace back low accuracy of tick-based algorithms to illiquidity in option markets. 
- We do not have the time to previous trades. One would expect higher performance for more frequently traded options. Results of Grauer doe not indicate such a behaviour.
- Theoretical coverage reported in Grauer matches. Practically, coverage is much smaller due to minimal filter set. 
- Simple experiment filter only for trades that can be classified by all trade classification rules
- Thus, we conclude tick-based algorithms are downward-biased
- Practically, coverage is much smaller e. g., negative / zero spreads, missingness of quotes etc. 
- missingness as key driver to performance
- Plot missing trade prices and quotes over time.
- contemplate the results
- quote nbbo is far smaller for cboe then for ise. could this be the reason why the order reverses?

**Finding 2: Accuracy comes from depth**
- building on finding 1, depth enables strong models, as classification is not performed using fallback criterion
- visualize which layer was used in classification
- inverse experiment. What happens if we work on filtered trades only? How does this affect the accuracy of hybrids? 

**Finding 3: Fee structures affect accuracy over time**
- performance fluctuates / diminishes over time and is affected by fee structure (see argument in [[@grauerOptionTradeClassification2022]])
- track down fee structure changes with patterns in time series.

![[Pasted image 20230606072531.png]]
![[Pasted image 20230606072617.png]]


![[accuracies_classical.png]]

For example, if the 50,000 transactions misclassi"ed by the Lee and Ready method constitute a representative cross-section of the entire sample, then the misclassi"cation will simply add noise to the data. In this case, the 85% accuracy rate is quite good. If, on the other hand, the Lee and Ready method systematically misclassi"es certain types of transactions, a bias could result.


![[summarized-results.png]]


Our remaining analysis is focused on the test set.


## Sub-samples
![[sub-samples.png]]

visualize classical rules over time 



- Unknown's are these where bid (ex) or ask (ex) is NaN. Grauer et al don't report these separately. They must be included somewhere else.
- Makes sense that unknowns are close to 50 % for e. g., quote rule (ex). 
- Stacking adds robustness. It looks suspicious that combinations e. g., quote + quote, GSU reach highest classification accuracies.

% TODO: These proxies have in common that they factor in the order book imbalance the relative depth quoted at the best bid and ask prices. If traders care about transaction costs, the relatively wide ask-side spread deters buyers, whereas the tight bid-side spread may attract sellers. There are then more traders submitting market orders at the bid side, and the true effective spread is, on average, smaller than the average midpoint effective spread.

% TODO: Derive in greater detail why orderbook imbalance makes sense! See my notes from Hagströmer

- Perform an error analysis. For which classes does CatBoost do so poorly? See some ideas here. https://elitedatascience.com/feature-engineering-best-practises

Akin to selecting the machine learning classifiers, we determine our classical baselines on the gls-ISE validation set. This guarantees a challenging baselines, while maintaining consistency between both paradigms. For the same reason, baselines are kept constant in the transfer setting on the gls-CBOE sample. Solely for reference, we also report accuracies of the gls-tick, gls-quote, gls-lr, due to their widespread adoption in finance.

(insert table here)

Table-x reports the accuracies of common trade classification rules over the entire validation set and broken down by the trade price's location relative to the quotes. The tick test applied to trade prices at the trading venue, performs worst with an accuracy below a random guess. Against this backdrop, we estimate all hybrid rules involving tick rule, over all exchanges ($\operatorname{tick}_{\text{all}}$). From all classical rules, a combination of the quote rule ($\operatorname{quote}_{\text{nbbo}} \to \operatorname{quote}_{\text{ex}}$), where the quote rule is first applied to the gls-NBBO and then to quotes of the gls-ISE quotes, performs best. The rule can be estimated using features from FS1, which qualifies it as a benchmark. Also, it is commonly studied in literature, as previously by ([[@muravyevOptionsTradingCosts2020]]).

By extension, we also estimate rules combinations involving overrides from the tradesize rule ($\operatorname{tsize}$) and the depth rule ($\operatorname{depth}$) on the top-performing baselines of FS1. Consistent with the recommendation of ([[@grauerOptionTradeClassification2022]]14), we find that a deep combination of the $\operatorname{tsize}_{\text{ex}} \to \operatorname{quote}_{\text{nbbo}} \to \operatorname{quote}_{\text{ex}} \to \operatorname{depth}_{\text{nbbo}} \to \operatorname{depth}_{\text{ex}} \to \operatorname{rtick}_{\text{all}}$ achieves the highest validation. For brevity, we refer to this hybrid as the gls-GSU method. Much of the performance improvements is owed to the trade size and depth rules, which reduce the dependence on the reverse tick test as a last resort and provide overrides for trades at the quotes, improving validation accuracy to percent-68.8359. 

In absence of other suitable baselines, we also the GSU method for FS3, even if it doesn't utilise option-specific features.

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



