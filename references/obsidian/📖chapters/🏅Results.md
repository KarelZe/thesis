## Summary
- start coarse-grained then report fine-grained results
- explain why results differ compared to Grauer et al
- think about reporting classical rules on the entire dataset

![[summarized-results.png]]


## Sub-samples
![[sub-samples.png]]

visualize classical rules over time 




**Classical ISE:**
| | | tick(all)         | quote(ex)    | lr(ex)    | emo(ex)   | clnv(ex)  | quote(best)->quote(ex) | trade_size(ex)->quote(best)->quote(ex)->depth(best)->depth(ex)->rev_tick(all) |           |           |
|-------------------|--------------|-----------|-----------|-----------|------------------------|-------------------------------------------------------------------------------|-----------|-----------|
| Option Type       | C            | 53.566483 | 56.338657 | 56.416520 | 53.500982              | 54.338435                                                                     | 58.881009 | 66.995000 |
|                   | P            | 53.086524 | 57.748410 | 57.797980 | 54.118193              | 55.227019                                                                     | 60.772429 | 67.442413 |
| Security Type     | Index option | 51.543849 | 53.728014 | 53.788502 | 51.280161              | 51.725311                                                                     | 57.797688 | 58.524483 |
|                   | Others       | 53.255024 | 62.402034 | 62.494080 | 57.510196              | 59.066922                                                                     | 65.242783 | 70.137410 |
|                   | Stock option | 53.405332 | 54.870101 | 54.923804 | 52.328385              | 53.061654                                                                     | 57.588565 | 66.150581 |
| Trade Size        | (0,1]        | 52.850289 | 55.288035 | 55.418330 | 51.930230              | 52.957108                                                                     | 58.219535 | 68.780466 |
|                   | (1,3]        | 52.825320 | 55.312322 | 55.398067 | 51.948962              | 52.864432                                                                     | 58.172238 | 68.845596 |
|                   | (3,5]        | 52.401723 | 55.723949 | 55.768923 | 52.821732              | 53.604825                                                                     | 58.488028 | 68.958994 |
|                   | (5,11]       | 53.636487 | 59.893257 | 59.899878 | 57.168629              | 57.977015                                                                     | 62.292800 | 63.295412 |
|                   | >11          | 55.555200 | 60.714690 | 60.695406 | 57.231189              | 58.461494                                                                     | 63.427558 | 64.547913 |
| Year              | 2015         | 52.783955 | 54.902913 | 54.892890 | 52.851584              | 53.315622                                                                     | 56.143249 | 63.349227 |
|                   | 2016         | 53.284415 | 57.571543 | 57.635410 | 54.270965              | 55.222778                                                                     | 59.896339 | 67.117888 |
|                   | 2017         | 53.665234 | 56.367496 | 56.459133 | 52.986116              | 54.147313                                                                     | 60.676194 | 68.701844 |
| Time to Maturity  | <= 1         | 53.073395 | 57.301220 | 57.362873 | 54.417911              | 55.223769                                                                     | 60.579671 | 67.111616 |
|                   | (1-2]        | 53.404881 | 57.744957 | 57.762882 | 53.547187              | 54.776254                                                                     | 60.546956 | 68.117541 |
|                   | (2-3]        | 53.697592 | 57.086186 | 57.179021 | 52.890993              | 54.140812                                                                     | 59.715752 | 67.886100 |
|                   | (3-6]        | 53.868526 | 56.015521 | 56.110247 | 52.001021              | 53.348644                                                                     | 57.877722 | 67.452554 |
|                   | (6-12]       | 54.269323 | 56.418807 | 56.472172 | 51.953456              | 53.581516                                                                     | 57.676192 | 67.470964 |
|                   | > 12         | 54.883085 | 52.696504 | 52.850764 | 51.198376              | 52.073364                                                                     | 50.941699 | 64.706435 |
| Moneyness         | <= 0.7       | 54.502433 | 60.341653 | 60.421858 | 58.257767              | 58.981319                                                                     | 61.450529 | 64.113542 |
|                   | (0.7-0.9]    | 55.491002 | 60.382149 | 60.573694 | 57.566221              | 58.607257                                                                     | 63.633651 | 67.868621 |
|                   | (0.9-1.1]    | 52.950755 | 57.083316 | 57.112069 | 53.144508              | 54.335962                                                                     | 60.238270 | 68.152214 |
|                   | (1.1-1.3]    | 51.621348 | 49.939075 | 49.998204 | 49.857566              | 49.670232                                                                     | 50.021966 | 61.661661 |
|                   | > 1.3        | 52.038059 | 48.723613 | 48.811119 | 50.046064              | 48.785237                                                                     | 48.722381 | 59.973687 |
| Location to Quote | at mid       | 51.034217 | 49.998387 | 50.611316 | 51.028068              | 51.029882                                                                     | 55.950353 | 65.902387 |
|                   | inside       | 52.408920 | 57.153707 | 57.153707 | 52.411885              | 53.793190                                                                     | 60.205206 | 64.242773 |
|                   | at quotes    | 57.728367 | 59.971382 | 59.971382 | 59.970566              | 59.971127                                                                     | 59.969340 | 78.213062 |
|                   | outside      | 62.363283 | 66.080655 | 65.745152 | 62.316312              | 62.235791                                                                     | 66.610750 | 64.892975 |
|                   | unknown      | 52.459802 | 50.133425 | 52.500855 | 52.398221              | 52.480328                                                                     | 76.346220 | 76.442012 |
| All trades        | all          | 53.342427 | 56.996762 | 57.061417 | 53.789110              | 54.753246                                                                     | 59.763967 | 67.203863 |

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



