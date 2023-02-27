All classical trade classification rules from (cref [[🔢Basic rules]]) perform *discrete classification* and assign a class to the trade. Naturally, a more powerful insight is to not just obtain the most probable class, but also the associated class probabilities for a trade to be a buy or sell. This gives additional insights into the confidence of the prediction, but calls for a *probabilistic classifier*.  

Thus, we frame trade signing as a probabilistic classification problem. This is similar to work of a ([[@easleyDiscerningInformationTrade2016]] 272), who retrieve probabilities from *bulked* versions of the tick rule and glsc-bvc algorithm, but on a *trade-per-trade* level. 

As our goal is to maintain 


In order to maintain 

Probabilistic classification in the context of trade classification rules, has been previously explored in ([[@easleyDiscerningInformationTrade2016]] 272). However, their scope is limited to *bulked* trade classification algorithms. Thus our work, 

We introduce some more notation, we will use throughout.

https://medium.com/@oded.kalev/comparing-classifiers-using-roc-9a9d8c9c819b


This provides additional insights on the uncertainty of the classifier.
This is particularily appealing for cases, if the assigned class is associated with uncertainty


(See [[@easleyDiscerningInformationTrade2016]]). However, this is not the case for the algorithms working on a trade-per-trade basis. Still, one can derive probabilities

Besides the predicted class, it would also 

A more powerful view 
We consider three methodologies to assign a probability that the underlying trade type was a buy or a sell given the observation of a single draw of :
In contrast  

meaning they directly assign a class to the trade / classify the trade to be buyer- or seller-initiated.  In contrast  

In a similar spirit,

Among numerous classifiers, some are hard classifiers while some are soft ones. Soft classifiers explicitly estimate the class conditional probabilities and then perform classification based on estimated probabilities. In contrast, hard classifiers directly target on the classification decision boundary without producing the probability estimation. (from [[@liuHardSoftClassification2011]]).



**Why probabilistic classification:**
- Due to a unsatisfactory research situation, for trade classification (see chapter [[👪Related Work]]) we base
- Use classification methods (*probabilistic classifier*) that can return probabilities instead of class-only for better analysis. Using probabilistic trade classification rules might have been studied in [[@easleyDiscerningInformationTrade2016]]
- Why to formulate problem as probabilistic classification problem: https://www.youtube.com/watch?v=RXMu96RJj_s
- Could be supervised if all labels are known
- Could be semi-supervised if only some of the labels are known. Cover later.
- hard and soft classification in general [[@liuHardSoftClassification2011]] and neural networks [[@foodyHardSoftClassifications2002]] (do not cite?)


- It's not easy to decide between *hard* and *soft classification*. See some references in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3233196/

hard decision boundary / boolean decision.










