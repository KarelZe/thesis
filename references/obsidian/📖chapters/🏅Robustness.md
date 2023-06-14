Following the recommendations of ([[@odders-whiteOccurrenceConsequencesInaccurate2000]], 2000, p. 280) To assess the robustness of our algorithms, we partition the test sets into sub-samples along seven dimensions: option type, security type, trade size, year, time to maturity, moneyness, as well as proximity to the quotes. Comparable robustness checks have been previously conducted in ([[@grauerOptionTradeClassification2022]]47) as well as  ([[@savickasInferringDirectionOption2003]]890--892), enhancing comparability between works.

Our results are tabulated cref-tab-14-17, separately for gls-gbrt and Transformers as well as exchanges. 

Performance improvement of glspl-gbrt are consistent for calls and puts across all feature sets and exchanges. Conditional on the security type of the underlying, gsl-gbrt achieves the largest improvements for index options in the gls-cboe sample, but perform slightly worse than rule-based approaches on the gls-ise set. On both datasets, accuracies are lowest for index options, which corroborates with literature on rule-based classification.

The performance is stable for different trade sizes and over time. Similarly, accuracy improvements are comparable for different maturities and moneyness ratios. Aligning with rule-based approaches, accuracies are lowest for option trades with long maturities and deep in-the-money options, as reported in ([[@grauerOptionTradeClassification2022]]22). 

glspl-GBRT achieve particularly strong results for trades at the quotes or if quote data at the exchange level is incomplete. In these subsets, improvements reach up to percentage-16.01 in the gls-cboe and thus tighten the gap between trades inside or at the quotes. Consistent across all feature sets and exchanges, glspl-GBRT fail to improve upon classical rules for trades outside the spread, underperforming the benchmark by percent--0.89 to percent--5.43. We identify the relatively stronger performance of quote-based classification on these trades as a reason that poses major challenges. 

In line with ([[@grauerOptionTradeClassification2022]]41--44) we observe a strong performance of the benchmark outside the quotes, This. While we also observe a In our test samples, also the benchmarks deteriorate for trades outside the spread, whereas for We attribute the discrepancies to ()... (piece in)

Opposing to

([[@savickasInferringDirectionOption2003]]894) document a high missclassification error for rule-based approaches on trades outside the quotes. Controversly, ()

In summary, the results do not indicate a misclassification bias

**Transformer**
Performance results of Transformers are robust across all tested dimensions. The accuracy is approximately equal for calls and puts. We observe, that benchmark performance of puts is consistently higher in our sub samples, which is contrasts the finding of ([[@grauerOptionTradeClassification2022]]22). 

Similar to glspl-gbrt, the FT-Transformer slightly underperforms the benchmark for index options in the gls-ISE sample. Even though the effect reverses on the gls-cboe set, accuracies for index options are considerably lower than for any other underlying. Thus, we can contemplate / extend the finding of ([[@grauerOptionTradeClassification2022]]22) and ([[@savickasInferringDirectionOption2003]]9) that index options are notoriously difficult to classify to machine learning-based approaches. 

Classification is more accurate for options with near-maturity or deep-in-the-money options. In this sense, our finding contradicts the observation of  ([[@savickasInferringDirectionOption2003]]891) made for rule-based classification. Again, we observe that the addition of option-specific features, e. g. maturity or moneyness smooths out differences between maturity and moneyness levels.

Finally, the FT-Transformers perform best on trades at the quotes, but fails to meet benchmark performance for trades outside the spread. Notably, the FT-Transformer trained on ise data achieves the substantial improvements on trades at the quotes, despite that the some of the benchmarks contain explicit overrides from the trade size rule.

In summary, our tests show that the strong results are stable across multiple dimensions and between exchanges, which makes supervised classifiers a superior choice for trade classification. 




Classification is generally more accurate outside the spread than at or inside the quote; this differs from Ellis et al.’s (2000) and Peterson and Sirri’s (2003) finding lower accuracy outside the spread. As mentioned previously, this difference may be due to omitting negotiated trades.



Specifically, higher for put this contradicts grauer


Performance is stable across time. 

Performance diminishes for deep-in-the-money options and options with long maturity for feature set classical and classical-size. The addition of option-specific features, however, leads to largest improvements for these options,

Perforamance improvements are particularilly strong strong for midspread trades and trades at the quotes

This property is particularily appealing

A sample split by the option types shows no misclassification bias for Transformers.

The Transformer results are robust for calls and puts. 

Improvements in accuracy are balanced between calls and puts for Transformers. 

This reverses

- results are smoother?



While performance improvements for trades at the quotes are particularly strong, Transformers do not consistently outperform their benchmarks for trades outside the spread. Overall, we observe smoothing

Similar
Controversely to grauer



Again, our results corroborate with the empirical literature. Third and finally, we find a substantial underperformance of the tick rule for zero tick trades, which is 9.33 percentage points lower than for non-zero ticks, compared to only 3.67 percentage points for the quote based rules.

Finally, we observe, tah
Gradient-boosting performs best, , while

The performance of the tick test is only slightly worse than the performance of LR's method. Table 4's final column lends additional support to this result. When the sample is limited to trades occurring on zero ticks, but trades on quote changes, mid-spread trades, and crosses are excluded, the tick test and LR's method both correctly identify at least 95% of the trades.


The results are displayed in Table 5. First, we confirm the asymmetry in buyer and sellerinitiated trades found by Aitkin and Frino (1996) and Omrane and Welsh (2016), with sellerinitiated trades performing remarkably better than buyer-initiated trades (the average difference is 7.46% for all TCR compared to 9.49% in Omrane and Welsh 2016). Second, for all quote-based rules we find lower accuracy for trades inside the quotes. Again, our results corroborate with the empirical literature. Third and finally, we find a substantial underperformance of the tick rule for zero tick trades, which is 9.33 percentage points lower than for non-zero ticks, compared to only 3.67 percentage points for the quote based rules.


Notably, Transformers 


Improvements 

Similarily, for Transformers we observe that impro





TODO: When the analysis was repeated using only the Lee and Radhakrishna sub-
sample, the results were equally as strong or stronger, with two exceptions. Using
their subsample, time between transactions is no longer a statistically signi”cant
determinant of misclassi”cation and large trades are misclassi”ed slightly more fre-
quently than small trades (odderswhite) TODO: Our focus is on ... rules. TODO:
Improvements are particularily high for trades that are notourisly hard to classify
by classical trade classification algorithms.

2.3.6 How to Write the Results  Content of the Results’ Section  Presentation and description (interpretation) of the data (only the new, own results)  Use of Past tense  Representative data, not repetitive data  How to handle data  One or only little data = text  Repetitive determinations = tables or graphs 14  Strive for clarity  Short, clear, simple  Avoid redundancy  no repetition in words, if results are apparent in figures and tables.  Our Recommendations: This chapter “Results” can be written concisely and simply if the data are presented by tables and graphs. One dataset has to be presented either by a table or a graph, not a table and a graph! If specific values need to be presented you should use the table form; if e.g. different variants should be compared, the reader often gets a better overview by looking at figures. Figures could also be helpful, if a large amount of data should be summarized. As far as “How to design effective graphs/figures and tables?” is concerned, look in journals specific to your topic or follow the advice given by Day and Gastel (2012) in the respective chapter in their book.


## Results over time

![[results-over-time-ise.png]]
![[accuracies-over-time-ise.png]]
(time series look rather similar)


- Do full analysis first, then look into sub samples
- “In light of this evidence, I recommend that researchers partition their transaction samples along the dimensions investigated in the paper and examine the impact on the results of their studies. If the "ndings are consistent across partitions, then researchers can be reasonably con"dent that their results are robust to misclassi"cation bias. On the other hand, if the results change along these dimensions without any clear explanation given the focus of the research, this suggests that misclassi"cation may be a problem. In this case, at a minimum, di!erences across partitions should be discussed along with the overall results.” ([[@odders-whiteOccurrenceConsequencesInaccurate2000]], 2000, p. 280)
- “Finally we examine the accuracy of the rules conditional on the characteristics of the trades, i.e. separately for the buy and sell side ([[@aitkenIntradayAnalysisProbability1995]], Omrane and Welsh 2016), for trades inside the quote (Ellis et al. 2000)3, and for zero versus non-zero ticks ([[@aitkenIntradayAnalysisProbability1995]], 1996; Theissen, 2001; Omrane and Welch, 2016)” (Frömmel et al., 2021, p. 7)
- “Furthermore, the most important biases encountered in the literature have been confirmed in this study: Seller-initiated trades perform remarkably better than buyer-initiated trades. The EMO rule, and especially the MEMO rule, offer substantial improvements over LR as they have far more power for classifying trades that occurred inside the quotes. The biggest disadvantage of the TR is its poor performance for zero ticks.” ([[@frommelAccuracyTradeClassification2021]], p. 9)

- Study results over time like in [[@olbrysEvaluatingTradeSide2018]]

- analyse the classifier performance based on the $|\cdot|$ proximity of the quotes?

- LR-algorithm (see [[#^370c50]]) require an offset between the trade and quote. How does the offset affect the results? Do I even have the metric at different offsets?
]]
- Are probabilities a good indicator reliability e. g., do high probablities lead to high accuracy.
- Are there certain types of options that perform esspecially poor?
- Confusion matrix
- create kde plots to investigate misclassified samples further
- ![[kde-plot-results.png]]
- What is called robustnesss cheques is also refered as **slice-based evaluation**. The data is separated into subsets and your model's performance on each subset is evaluated. A reason why slice-based evaluation is crucial is Simpson's paradox. A trend can exist in several subgroups, but disappear or reverse when the groups are combined. Slicing could happen based on heuristics, errors or a slice finder (See [[@huyenDesigningMachineLearning]])
![[rankwise-correlations.png]]
(found in `@hansenApplicationsMachineLearning`, but also other papers)

“Finucane (2000) finds that a large proportion of incorrectly classified trades are trades with zeroticks.” ([Chakrabarty et al., 2007, p. 3814](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=9&annotation=6YW8JBQ6))

“Trade size may also affect the accuracy of trade classification rules. Odders-White (2000) finds that the success rate is higher for large trades than for small trades while Ellis et al. (2000) find that large trades are more frequently misclassified than small trades” ([Chakrabarty et al., 2007, p. 3814](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=9&annotation=RNDU5P5Z))

- “Furthermore, the most important biases encountered in the literature have been confirmed in this study: Seller-initiated trades perform remarkably better than buyer-initiated trades. The EMO rule, and especially the MEMO rule, offer substantial improvements over LR as they have far more power for classifying trades that occurred inside the quotes. The biggest disadvantage of the TR is its poor performance for zero ticks.” (Frömmel et al., 2021, p. 9) -> How are things in [[@savickasInferringDirectionOption2003]] and [[@grauerOptionTradeClassification2022]]


- “Overall, there is a monotonic relationship: better classification for smaller trades. For example, trades of less than 200 shares are correctly classified 81.73% of the time compared with 77.85% for trades over 10,000 shares. This is in contrast to Odders-White's (2000) finding that accuracy is lower for smaller NYSE trades.” ([Ellis et al., 2000, p. 536](zotero://select/library/items/54BPHWMV)) ([pdf](zotero://open-pdf/library/items/TTB4YUW6?page=9&annotation=SDMJDLDI))
- “Conditioning on the location of the trade, we find trade classification accuracy increases for larger trades, but overall accuracy is lower.” ([Ellis et al., 2000, p. 536](zotero://select/library/items/54BPHWMV)) ([pdf](zotero://open-pdf/library/items/TTB4YUW6?page=9&annotation=E56HF6MG))
- “The regression shows trade size, firm size, trading speed, and quoting speed are each less significant in determining the probability of correct classification than is proximity to the quotes. The probability of correct classification increases with trade size, decreases with firm size, increases with the time between trades (less rapid trading), and increases with the time between a quote update and a trade.” ([Ellis et al., 2000, p. 539](zotero://select/library/items/54BPHWMV)) ([pdf](zotero://open-pdf/library/items/TTB4YUW6?page=12&annotation=96FSFA7I))


![[viz-robustness.png]]
(from [[@carionEndtoEndObjectDetection2020]]; 12)

![[rules-across-underlyings.png]]


“Trade size may also affect the accuracy of trade classification rules. Odders-White (2000) finds that the success rate is higher for large trades than for small trades while Ellis et al. (2000) find that large trades are more frequently misclassified than small trades” (Chakrabarty et al., 2007, p. 3814)

To further challenge our hypothesis and to validate the improvement by our new trade size rule, we conduct two subsample analyses. First, we evaluate the performance of our trade size rule separately for different locations of trade prices relative to the bid and ask quotes at the ISE. The results in Panel B of Table 5 show that the new rule works best for trades occurring at the ask or bid quote and improves the success to classify at-quote trades by up to 21%. Contrary, the trade size rule even deteriorates the performance to correctly classify outside-quote trades compared to the traditional trade classification approaches by up to 4.5%. These results are also in line with our hypothesis that market makers fill limit orders from customers at the limit price, set by the customer. Contrary, if the trade price is outside the bid-ask spread, this is an indication that a customer wanted to trade against a standing limit order of a market maker, but the size of the market maker’s quote was not sufficient, leading to a further price deterioration. As a second subsample analysis, we evaluate the performance separately for various trade size categories. Figure 1 shows average success rates for the different specifications of the quote, tick, LR, reverse LR, EMO, and depth rules after the trade size rule has been applied for different trade 8OptionMetrics recently started offering a product (“IvyDB Signed Volume”) that provides buying and selling volume information. Their classification is also based on the LR algorithm (see OptionMetrics (2020)). 15 Electronic copy available at: https://ssrn.com/abstract=409847

size bins.9 The cutoffs for the bins are calculated as quintiles and are measured in number of contracts. We show the overall success rates of the classification algorithms using our trade size rule and also calculate the change in the success rates compared to the same algorithms not using the trade size rule. The results show that our new rule works best for small to medium-sized trades and even leads to a slight deterioration of the performance for the largest trade sizes. This finding is in line with the hypothesis that limit orders placed by customers are more likely to be smaller trades. In contrast, large trades for which the trade size is equal to the quote size are more likely to be market orders in which customers want to trade the full depth of the market maker’s bid or ask quote. Given the results of the two subsample analyses, it would be possible to further improve the methodology of applying the trade size rule. In additional results, which are not tabulated to conserve space, we find that not applying the new rule for very large trades and for trades outside the bid-ask spread leads to small additional improvements of up to 0.4%.10 Due to the additional complexity, which is prone to a potential over-fitting regarding the cutoff between small and large trades, and also due to the results from the out-of-sample tests that show mixed results for these additional refinements, we recommend to apply our two new rules to all trades.