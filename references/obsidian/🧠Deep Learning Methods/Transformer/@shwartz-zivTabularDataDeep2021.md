
*title:** Tabular Data: Deep Learning is Not All You Need
*authors:** Ravid Shwartz-Ziv, Amitai Armon
*year:** 2021
*tags:* #deep-learning #gradient_boosting #supervised-learning #gbm 
*status:* #📥
*related:* 
- [[@borisovDeepNeuralNetworks2022]]
## Notes Sebastian Raschka
-   This paper compares XGBoost and deep learning architectures for tabular data; no new method is proposed here.
-  The results show that XGBoost performs better than most deep learning methods across all datasets; however, while no deep learning dataset performs well across _all_ datasets, a deep learning method usually performs better than XGBoost (except on one dataset). The takeaway is that across different tasks, XGBoost performs most consistently well.
- Another takeaway is that XGBoost requires substantially less hyperparameter tuning to perform well, which is a significant benefit in many real-life scenarios.
-   The experiments with various ensembles are worth highlighting: The best results are achieved when deep neural networks are combined with XGBoost.  
-   No code examples are available, so everything in this paper must be taken with a large grain of salt
