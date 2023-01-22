
## Transformer
[[🤖Training of the Transformer]]

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















