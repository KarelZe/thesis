Classical trade signing algorithms, such as the tick test, are also impacted by missing values. In theses cases, we defer to a random classification or a subsequent rule, if rules can not be computed. Details are provided in section [[💡Training of models (supervised)]].


## Transformer
[[🤖Training of the Transformer]]


Look into grooking: https://arxiv.org/pdf/2201.02177.pdf
![[grocking.png]]

- What optimizer is chosen? Why? Could try out Adam or Adan?

[[@somepalliSAINTImprovedNeural2021]] use logistic regression. I really like the fact they also compare a simple logistic regression to these models, because if you’re not able to perform notably better relative to the simplest model one could do, then why would we care? The fact that logistic regression is at times competitive and even beats boosting/SAINT methods occasionally gives me pause though. Perhaps some of these data are not sufficiently complex to be useful in distinguishing these methods? It is realistic though. While it’s best not to assume as such, sometimes a linear model is appropriate given the features and target at hand.


Many practical implementations of boosting like XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017), and CatBoost (Prokhorenkova et al., 2018) use constant learning rate in their default settings as in practice it outperforms dynamically decreasing ones. However, existing works on the convergence of boosting algorithms assume decreasing learning rates (Zhang & Yu, 2005; Zhou & Hooker, 2018), thus leaving an open question: if we assume constant learning rate  > 0, can convergence be guaranteed? https://arxiv.org/pdf/2001.07248.pdf

## Categoricals
- The problem of high number of categories is called a high cardinality problem of categoricals see e. g., [[@huangTabTransformerTabularData2020]]
- To inform our models which features are categorical, we pass the index the index of categorical features and the their cardinality to the models.
- Discuss cardinality of categoricals.
- strict assumption as we have out-of-vocabulary tokens e. g., unseen symbols like "TSLA".  (see done differently here https://keras.io/examples/structured_data/tabtransformer/)
- Idea: Instead of assign an unknown token it could help assign to map the token to random vector. https://stackoverflow.com/questions/45495190/initializing-out-of-vocabulary-oov-tokens
- Idea: reduce the least frequent root symbols.
- Apply an idea similar to sentence piece. Here, the number of words in vocabulary is fixed https://github.com/google/sentencepiece. See repo for paper / algorithm.
- For explosion in parameters also see [[@tunstallNaturalLanguageProcessing2022]]. Could apply their reasoning (calculate no. of parameters) for my work. 
- KISS. Dimensionality is probably not so high, that it can not be handled. It's much smaller than common corpi sizes. Mapping to 'UKNWN' character. -> Think how this can be done using the current `sklearn` implementation.
- **Solutions:** 
	- Use a linear projection: https://www.kaggle.com/code/limerobot/dsb2019-v77-tr-dt-aug0-5-3tta/notebook
	- https://en.wikipedia.org/wiki/Additive_smoothing



Studies adressing high cardinality
<mark style="background: #ADCCFFA6;">“Study how both tree-based models and neural networks cope with specific challenges such as missing data or high-cardinality categorical features, thus extending to neural networks prior empirical work [Cerda et al., 2018, Cerda and Varoquaux, 2020, Perez-Lebel et al., 2022].” ([Grinsztajn et al., 2022, p. 9](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=9&annotation=PCA3SDUE))</mark>














