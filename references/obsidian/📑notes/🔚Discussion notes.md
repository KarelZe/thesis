
- moneyness / time-to-maturity / how do both relate with trade classification / motives
- low accuracy for index options
	- Study sources of missclassification. See e. g., [[@savickasInferringDirectionOption2003]]
	- The extent to which inaccurate trade classification biases empirical research dependes on whether misclassifications occur randomly or systematically [[@theissenTestAccuracyLee2000]]. This document also contains ideas how to study the impact of wrong classifications in stock markets. Might different in option markets.
- low accuracy for trades outside the quotes
	- see also [[@ellisAccuracyTradeClassification2000]] for trades inside and outside the spread
- high gains for options for otm options and options with long maturity
	- Accuracy is not the sole criterion. Depends on whether error is systematic or not. Thus, we do application study. See reasoning in ([[@theissenTestAccuracyLee2000]])
- performance gap in classical rules
- strong performance of neural networks / tree-based ensembles
	- We identify missingess in data to be down-ward biasing the results of classical estimators. ML predictors are robust to this missingness, as they can handle missing values and potentially substitute.
- methodology
	- our study puts special emphasises on thoughtful tuning, data pre-processing.
- the elephant in the room: 
	- labelled data and cmputational data. 
	- Finetune. Low cost of inference
- which algorithm is no preferable? Do Friedman rank test


## Algorithm
2.3.7 How to Write the Discussion  Assessment of the results  Comparison of your own results with the results of other studies = Citation of already published literature!  Components  Principles, relationships, generalizations shown by the results = Discussion, not recapitulation of the results  Exceptions, lack of correlation, open points  Referring to published work: = Results and interpretations in agreement with or in contrast to your results  Our Recommendations: The writing of the chapter “Discussion” is the most difficult one. Compare your own data/results with the results from other already published papers (and cite them!). Outline the discussion part in a similar way to that in the Results section = consistency. Evaluate whether your results are in agreement with or in contrast to existing knowledge to date. You can describe why or where the differences occur, e.g. in methods, in sites, in special conditions, etc. Sometimes it is difficult to discuss results without repetition from the chapter “Results”. Then, there is the possibility to combine the “Results” and “Discussion” sections into one chapter. However, in your presentation you have to classify clearly which are your own results and which are taken from other studies. For beginners, it is often easier to separate these sections.


## Other
- calculate $z$-scores / $z$-statistic of classification accuracies to assess if the results are significant. (see e. g., [[@theissenTestAccuracyLee2000]])
- provide $p$-values. Compare twitter / linkedin posting of S. Raschka on deep learning paper.
- When ranking algorithms think about using the onesided Wilcoxon signed-rank test and the Friedman test. (see e. g. , code or practical application in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- To test these hypotheses it would be best if we had the precise motivation behind the trades. While such analysis is not feasible here, using trade classification algorithms, we are able to assign stock and option volume as buyer or seller initiated. Easley et al. (1998) show how this directional volume is more informative than raw volume, because signed volume provides important information about the motivation of the trade (bullish or bearish). ([[@caoInformationalContentOption2005]])
- [https://doi.org/10.1287/mnsc.2019.3398](https://doi.org/10.1287/mnsc.2019.3398)
- https://pdf.sciencedirectassets.com/271671/1-s2.0-S0304405X20X00067/1-s2.0-S0304405X19302831/am.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIDixxfTiKliJIuzoOXxlII71RLwniTDskEPKeGqAyItEAiEA4%2Fytxevo9ZXJNkxW1jrTnKzaaobWySQgbq68siGmgwQquwUI7f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDLKVDK%2Bh4Dg%2B7qRvKCqPBUP%2BVpVVJhJWgxEXvneeMcgDHwz3Q%2BFj4BacIV3D2cLphWRHZirMPOW1Scz6VIzOfzGnUYgdZXRNb0yT8KQur9GvN%2B1TwgULtgOLUlII49PNpfnhgo%2B5TFji2%2BRpB4Bs7BoBu6JZH6x2vrhjfqFGSsl19%2Bsyxe3zfS%2BYZzLkEUBwXTVS0Omt3AWowOaltN5qRbzjH0M16ijT3HTA3BTtQJLZe%2BNqqKsohziXZJ2GIC0I%2BswnrB9qpx8TplWGO62ITP0I4Xa4F2GhzByCl2nrGKeHUdJ03VUa3dYpyw4ml8n3E7ADheEZh4yhh8W3GS%2Btc2AkrpJkl9JpInWeTwijmC5rsQVtRZfYLCNFXdSZkPtWFWOBYM0WVIiRHMx8urSTYs%2FQ5XiP61nmWn%2BlIdyeLDYgg8uYcBCwciMCBdfBKu86mAK42snqIIJC8fHQ6RjZ0HkTxXK3ecfWG9ZD5LYrwOig7B30VufNzSvG%2FnJ8UxeUOPfXcX9Ob8OEUuaWzvTCSU3%2BIsw8vx4%2BpScmof5EwYvWDb4ndAD02RdDsAps9DjoZT0Fo6ezxpGZYsSzJ%2FRvzXoxrKIhTVugO2%2BDJubQ9sHIex7HBmGf1dM3j56ypwqghzdFmDohh4bPT3oYbBkQkIeojKcuifG6RPtAROKxHdSv0Htm9LZrdehnkehKyeFESJ8pcZ9IrTP5sejH9%2BHrVo8m7gjUaYTw6vWdsQxw1dCZ96jSuoANt2O8QvxR5S%2BG078zV0yhx76Y1nuhz3Dzgk%2FJCwLUwklQcsGDNZzXKhuXoZkpyE1sHDgqeSwDU7xYdJEZnQy60exHcdnjh5qQt1cY3ZCb4EMH8Y4yUtDwJgf5YOlxHJo8rElAkn5T3e0w2YDWpAY6sQGhOGrkIFN%2FaJBDi7pCA1DemSn4hZ5FnHX1%2Fh492NDWmIx5ojFpVIHW5WftmRayIG6R4hzuIank8S0YgtSwnLUQpKMXKQEBeTmLwzHqVg7sHjO8dIJ0dtTywPmoP%2Fz7Op4wRbImbrCavUEUdf9bIt8ywHhHPJowJ9OnPNLByKti3DmdKLe%2FjO1iMpJGKTCwRu4WN9EDOsW3Jxo1xPUe3l2EUBbOB5oy6z3j8m1qo4bMbDE%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230623T124336Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYUUDJI6EP%2F20230623%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f0e2119f981b01f0c6f371ba9815e791e0ff23c077563ba5b6e75152e2e77385&hash=97254ce55dfdd1c8959a1ac605bad32a3be5f6fff1c70209a668432c2a4bbeff&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0304405X19302831&tid=pdf-81ea8b48-352c-43fe-bfe6-12fecbdb988f&sid=d1961e3d9967514b7918a74-b56e3090c4eagxrqb&type=client
- https://doi.org/10.1287/mnsc.2019.3529
- https://www.dropbox.com/s/1i4zxc23qm00bv9/OptionMarketMakers.032623.pdf?dl=0
- https://dmurav.com/CV_Dmitry_Muravyev_202305.pdf
- for index options see [[@chordiaIndexOptionTrading2021]]
- To test these hypotheses it would be best if we had the precise motivation behind the trades. While such analysis is not feasible here, using trade classification algorithms, we are able to assign stock and option volume as buyer or seller initiated. Easley et al. (1998) show how this directional volume is more informative than raw volume, because signed volume provides important information about the motivation of the trade (bullish or bearish). ([[@caoInformationalContentOption2005]])