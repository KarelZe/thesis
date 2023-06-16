## Trade Classification Rules

We now estimate the accuracy of classical trade classification rules on the gls-ise and gls-cboe sample. We estimate the performance of the tick and quote rule, as well as the gls-LR algorithm, gls-EMO rule and gls-CLNV method in their classical and reversed formulation. Additionally, we consider two stacked combinations of ([[@grauerOptionTradeClassification2022]]) due to their state-of-the-art-performance on the validation set, as derived in cref-[[💡Hyperparameter Tuning]]. Namely, $\operatorname{quote}_{\mathrm{nbbo}} \to \operatorname{quote}_{\mathrm{ex}} \to \operatorname{rtick}_{\mathrm{all}}$ and $\operatorname{tsize}_{\mathrm{ex}} \to \operatorname{quote}_{\mathrm{nbbo}} \to \operatorname{quote}_{\mathrm{ex}} \to \operatorname{depth}_{\mathrm{nbbo}} \to \operatorname{depth}_{\mathrm{ex}} \to \operatorname{rtick}_{\mathrm{all}}$ or in short $\operatorname{gsu}_{\mathrm{small}}$ and $\operatorname{gsu}_{\mathrm{large}}$. 

We report in cref-table-ise accuracies for the entire data set and separate subsets spanning the periods of train, validation, and test set. Doing so, allows us to compare against previous works, but also provide meaningful estimates on the test set relevant for benchmarking purposes. 

Our results are approximately similar to ([[@grauerOptionTradeClassification2022]]29-33). Minor deviations exist, which can linked to differences in handling of unclassified trades and non-positive spreads, as well divergent implementations of the depth rule.-footnote(Correspondence with the author.)

From all rules, tick-based algorithms perform worst when applied to trade prices at the trading venue with accuracies of a random guess, percentage-49.67 or percentage-51.47.  For comparison, a simple majority vote would achieve percentage-51.40 accuracy. The application to trade prices at the inter-exchange level marginally improves over a random / dummy classification, achieving accuracies of percentage-55.25 for the reversed tick test. Due to the poor performance, of tick-based algorithms at the exchange level, we estimate all hybrids with $\operatorname{tick}_{\mathrm{all}}$ or $\operatorname{rtick}_{\mathrm{all}}$.

![[classical-ise-results.png]]

Quote-based algorithms, outperform tick-based algorithms delivering accuracy up to percentage-63.71, when estimated on the gls-NBBO. The superiority of quote-based algorithms in option trade classification has previously been documented in ([[@savickasInferringDirectionOption2003]]) and ([[@grauerOptionTradeClassification2022]]). 

The performance of hybrids, such as the gls-LR algorithm, hinges with the reliance on the tick test. Thus, the gls-emo rules and to a lesser extent the gls-clnv rules perform worst, achieving accuracies between percentage-55.42 and percentage-57.57. In turn, variants of the gls-LR, which uses the quote rule for most trades, is among the best performing algorithms. By extension, $\operatorname{gsu}_{\mathrm{small}}$ further reduces the dependence on tick-based methods through the successive applications of quote rules, here $\operatorname{quote}_{\mathrm{nbbo}} \to \operatorname{quote}_{\mathrm{ex}}$.

Notably, the combination of ([[@grauerOptionTradeClassification2022]]33) including overrides from the trade size and depth rules performs best, achieving percentage-67.20 accuracy on the gls-ise test set and percentage-75.03 on the entire dataset. Yet, the performance deteriorates most sharply between sets.

![[accuracies_classical.png]]

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

We report accuracies for train, validation set and test set and entire set


Aside from these high-level observations, we focus three findings in greater detail. 

We repeat the analysis on the gls-cboe dataset in cref-table-cboe and observe a similar ranking to cref-table-ise. Overall, the performance of classical trade classification rules further diminishes strengthening the need for alternative classifiers. Tick-based rules trail the performance of quote-based approaches, and the accuracy of hybrids varies with the dependence on the tick test. Different from the gls-ise sample, the quote rule estimated on the gls-NBBO, $\operatorname{quote}_{\mathrm{nbbo}}$, leads to a lower performance than the quote rule applied to gls-CBOE quotes. Parts of this is due to the fact, that  $\operatorname{quote}_{\mathrm{nbbo}}$ achieves a considerably lower coverage of percentage-94.77 compared to percentage-99.89 in the gls-ise sample, with fewer trades classified by the fallback criterion. In a filtered common sample, where trades are classified by both rules, performance is approximately similar. Again, $\operatorname{gsu}_{\mathrm{small}}$ and $\operatorname{gsu}_{\mathrm{large}}$ perform best. footnote-(Performance on gls-cboe, can be improved, if the order of quote rules is reversed. For full combinatoric coverage see ([[@grauerOptionTradeClassification2022]]33).  To avoid overfitting the test set by classical rules, we keep the baseline constant following our reasoning from cref-[[💡Hyperparameter Tuning]].) On the test subsample, performance improvements from the trade size and depth rule are considerably smaller than in the gls-ISE dataset. 

For example, if the 50,000 transactions misclassi"ed by the Lee and Ready method constitute a representative cross-section of the entire sample, then the misclassi"cation will simply add noise to the data. In this case, the 85% accuracy rate is quite good. If, on the other hand, the Lee and Ready method systematically misclassi"es certain types of transactions, a bias could result.


![[summarized-results.png]]

Our remaining analysis is focused on the test set.

By extension, we also estimate rules combinations involving overrides from the tradesize rule ($\operatorname{tsize}$) and the depth rule ($\operatorname{depth}$) on the top-performing baselines of FS1. Consistent with the recommendation of ([[@grauerOptionTradeClassification2022]]14), we find that a deep combination of the $\operatorname{tsize}_{\text{ex}} \to \operatorname{quote}_{\text{nbbo}} \to \operatorname{quote}_{\text{ex}} \to \operatorname{depth}_{\text{nbbo}} \to \operatorname{depth}_{\text{ex}} \to \operatorname{rtick}_{\text{all}}$ achieves the highest validation. For brevity, we refer to this hybrid as the gls-GSU method. Much of the performance improvements is owed to the trade size and depth rules, which reduce the dependence on the reverse tick test as a last resort and provide overrides for trades at the quotes, improving validation accuracy to percent-68.8359. 

In absence of other suitable baselines, we also the GSU method for FS3, even if it doesn't utilise option-specific features.


**Tsize (alone) (ISE)**
![[Pasted image 20230616153144.png]]



Tsize (alone) (CBOE)

![[Pasted image 20230616153642.png]]


Accuracy Depth Ex (ISE)

![[Pasted image 20230616153235.png]]

Accuracy Depth EX (CBOE)
![[Pasted image 20230616153750.png]]