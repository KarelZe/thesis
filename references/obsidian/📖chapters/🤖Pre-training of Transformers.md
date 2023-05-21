
Code github.com/LevinRoman/tabular-transfer-learning.

For deep models with transfer learning, we tune the hyperparameters on the full upstream data using the available large upstream validation set with the goal to obtain the best performing feature extractor for the pre-training multi-target task. We then fine-tune this feature extractor with a small learning rate on the downstream data. As this strategy offers considerable performance gains over default hyperparameters, we highlight the importance of tuning the feature extractor and present the comparison with default hyperparameters in Appendix B as well as the details on hyperparameter search spaces for each model.




**RTD for Textual Data**
gls-rtd is a pre-training objective proposed by ([[@radfordImprovingLanguageUnderstanding]]2--3) for the use in language models. The core idea is to randomly replace tokens with plausible alternatives and learn a binary classifier to distinguish between original and replaced tokens. Intuitionally, the random replacement forces the model to learn generalisable representations of the input, rather than memorising the co-occurrence of certain tokens. Additionally, surprising the model with random tokens strengthens its ability to incorporate contextual information.

The approach uses two neural networks, namely the generator and the discriminator, typically implemented as Transformers.  The generator, is responsible for generating replacement tokens, receives an input sequence, i. e., a sentence, that has been intentionally masked out. It learns to predict the original token of the now-masked token through tokens in the bidirectional context (cp. [[🅰️Attention]]). For masking, an additional $\mathtt{[MASK]}$ token is introduced, that extends the vocabulary (cp. [[💤Embeddings For Tabular Data]]). Separately for each token, the final hidden state of the masked token is fed through a softmax activation to obtain the predicted probability distribution of the masked token and the cross entropy loss is used to compare against the true distribution. By replacing the masked token with a token from the generator distribution, convincing replacements now take place for some of the original inputs ([[@radfordImprovingLanguageUnderstanding]]2--3).  

The discriminator then receives the corrupted input sequence and is trained to distinguish between original and replaced tokens originating from the generator. The output is a binary mask which can be compared against the mask initially used for masking tokens in the generator ([[@radfordImprovingLanguageUnderstanding]]2--3).

![[architecture-rtd.png]]
([[@radfordImprovingLanguageUnderstanding]]2--3)

**Random Token Replacement for Tabular Data**
Applied to tabular datasets, gls-rtd transfers to randomly replacing feature values in $\mathbf{x}_{i}$ instead of sequences. The objective is now to predict a binary mask $\mathbf{m}_{i}\in \{0,1\}^{M}$ corresponding to $\mathbf{x}_{i}$, indicating which features, or entries in $\mathbf{x}_{i}$, have been replaced. Previous adaptions for tabular data, e.g., ([[@huangTabTransformerTabularData2020]]3), simplify the replacement strategy by sampling replacement values directly from the feature, which alleviates the need for a generator network and requires less compute. Since, the replacement is done on a feature-per-feature basis, the replaced token is *per se* harder to detect.

For tabular the random replacement of feature values also strengthens the model's ability to incorporate a combination of features, rather than single or few features based on their absolute value, which would facilitate overfitting.

In summary, gls-rtd can help to learn expressive representations of the input, even if the true label is unavailable. Given previous research, we expect pre-training gls-rtd to improve the performance of the FT-Transformer and match or exceed the one of the gls-gbm.

**Notes:**
[[🤖Pretraining notes]]


