We compare the performance of pre-trained Transformers and self-trained gradient-boosting on the gls-ise and gls-cboe test set. Results are reported in cref-tab-semi-supervised-results. 
![[Pasted image 20230701154037.png]]
(supervised)

![[Pasted image 20230701153951.png]]
(semi-supervised)

Identical to the supervised case, our models consistently outperform their respective benchmarks. Gradient boosting with self-training surpasses $\operatorname{gsu}_{\mathrm{small}}$ by percentage-3.35 on gls-ise and percentage-5.44 on gls-cboe in accuracy. Improvements for larger feature sets over $\operatorname{gsu}_{\mathrm{large}}$ are marginally lower to the supervised model and range between percentage-4.55 and percentage-7.44.

Pre-training is beneficial for the performance of Transformers on \gls{ISE} trades, improving over Transformer with random initialisation by up to \SI{0.87000}{\percent}. Hence, the performance improvement from pre-training observed in cref-[[💡Hyperparameter Tuning]] on the validation set carries over the test set. On the \gls{CBOE} dataset, pre-training hurts performance.

As no previous work performed semi-supervised classification, we discuss where the performance difference between pre-training and self-training hails from.

Pre-training is beneficial for transfor

![[Pasted image 20230701153902.png]]
As evident from \cref{tab:contigency-semi-supervised-classifiers}, a vast majority of trades are classified by both classifiers correctly. For the \gls{ISE}, performance improvements in larger feature sets are driven by trades that are distinctly classified by both classifiers. In turn, at the \gls{CBOE}, the share of common classifications continues to grow. Performance differences between classifiers are significant.    


The results do not support the hypothesis, that incorporating unlabelled trades into the training corpus improves the performance of the classifier. We explore this finding in detail.

\todo{Why is pre-training successful? Why is self-training not successful?}

\todo{How would a linear model do?}


Rethinking pre-training and universal feature representations. One of the grandest goals of computer vision is to develop universal feature representations that can solve many tasks. Our experiments show the limitation of learning universal representations from both classification and self-supervised tasks, demonstrated by the performance differences in self-training and pre-training. Our intuition for the weak performance of pre-training is that pre-training is not aware of the task of interest and can fail to adapt. Such adaptation is often needed when switching tasks because, for example, good features for ImageNet may discard positional information which is needed for COCO. We argue that jointly training the self-training objective with supervised learning is more adaptive to the task of interest. We suspect that this leads self-training to be more generally beneficial.

To summarise, unrewarded for higher training costs, semi-supervised variants of \glspl{GBRT} do not provide better generalisation performance than supervised approaches. Pre-training of Transformers improves performance on the \gls{ISE} sample but slightly deteriorates performance on the \gls{CBOE} set. We subsequently evaluate if semi-supervised learning improves robustness if not performance.


**Finding 5: Unlabelled Trades Provide Poor Guidance**
todo()

To summarize, despite the significantly higher training costs, semi-supervised variants do not provide better generalisation performance than supervised approaches. We subsequently evaluate if semi-supervised learning improves robustness, if not performance.