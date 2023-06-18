
- results for classical rules demonstrate that classical choices for option trade classification.
- We identify missingess in data to be down-ward biasing the results of classical estimators. ML predictors are robust to this missingness, as they can handle missing values and potentially substitute.

- our study puts special emphasise on thoughtful tuning, data pre-processing.
- 

- the elephant in the room, labelled data and cmputational data. Finetune. Low cost of inference
- our results contradict ronen et al. neural networks can achieve sota performance if well tuned.


es it mean? Point out limitations and e. g., managerial implications or future impact.
- How do wide models compare to deep models
- Study sources of missclassification. See e. g., [[@savickasInferringDirectionOption2003]]
- Would assembeling help here? As discussed in [[@huangSnapshotEnsemblesTrain2017]] ensembles can only improve the model, if individual models have a low test error and if models do not overlap in the samples they missclassify.
- The extent to which inaccurate trade classification biases empirical research dependes on whether misclassifications occur randomly or systematically [[@theissenTestAccuracyLee2000]]. This document also contains ideas how to study the impact of wrong classifications in stock markets. Might different in option markets.
- Ceveat is that we don't know the true labels, but rather subsets. Could be biased?

“The established methods, most notably the algorithms of Lee and Ready (1991) (LR), Ellis et al. (2000) (EMO), and Chakrabarty et al. (2007) (CLNV), classify trades based on the proximity of the transaction price to the quotes in effect at the time of the trade. This is problematic due to the increased frequency of order submission and cancellation. With several quote changes taking place at the time of the trade, it is not clear which quotes to select for the decision rule of the algorithm.” (Jurkatis, 2022, p. 6)

To put these results in perspective, our best model using additional trade size and option features improves over the frequently employed tick rule, quote rule, and gls-lr algorithm by more than (74.12 - 57.10) on the gls-ISE sample. 

Cost of inference is low. Good practical use.



We have proposed a series of analysis methods for understanding the attention mechanisms of models and applied them to BERT. While most recent work on model analysis for NLP has focused on probing vector representations or model outputs, we have shown that a substantial amount of linguistic knowledge can be found not only in the hidden states, but also in the attention maps. We think probing attention maps complements these other model analysis techniques, and should be part of the toolkit used by researchers to understand what neural networks learn about language.

In this work, we have systematically evaluated typical pretraining objectives for tabular deep learning. We have revealed several important recipes for optimal pretraining performance that can be universally beneficial across various problems and models. Our findings confirm that pretraining can significantly improve the performance of tabular deep models and provide additional evidence that tabular DL can become a strong alternative to GBDT.


Discussion AlphaDev discovers new, state-of-the-art sorting algorithms from scratch that have been incorporated into the LLVM C++ library, used by millions of developers and applications around the world23–25. Both AlphaDev and stochastic search are powerful algorithms. An interesting direction for future research is to investigate combining these algorithms together to realize the complementary advantages of both approaches. It is important to note that AlphaDev can, in theory, generalize to functions that do not require exhaustive verification of test cases. For example, hashing functions48 as well as cryptographic hashing functions49 define function correctness by the number of hashing collisions. Therefore, in this case, AlphaDev can optimize for minimizing collisions as well as latency. AlphaDev can also, in theory, optimize complicated logic components within the body of large, impressive functions. We hope that AlphaDev can provide interesting insights and inspire new approaches in both the artificial intelligence and program synthesis communities.

In this paper, we presented a series of language models that are released openly, and competitive with state-of-the-art foundation models. Most notably, LLaMA-13B outperforms GPT-3 while being more than 10× smaller, and LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B. Unlike previous studies, we show that it is possible to achieve state-of-the-art performance by training exclusively on publicly available data, without resorting to proprietary datasets. We hope that releasing these models to the research community will accelerate the development of large language models, and help efforts to improve their robustness and mitigate known issues such as toxicity and bias. Additionally, we observed like Chung et al. (2022) that finetuning these models on instructions lead to promising results, and we plan to further investigate this in future work. Finally, we plan to release larger models trained on larger pretraining corpora in the future, since we have seen a constant improvement in performance as we were scaling.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

The challenges encountered throughout this work can be revisited in future work. T

n this thesis, we proposed different strategies to estimate the contrast of subspaces on a data stream in real-time. We introduced a new index structure allowing efficient insert- and delete-operations to improve the runtime of the contrast estimator. Building on this, we successfully extended GMD to the streaming setting. Our empirical results show that the contrast estimated by the Exponential Weighting Strategy is close to the contrast reported by the baseline algorithm while running three times faster. Our subspace search algorithm runs up to 30 times faster than the baseline algorithm and produces equally-good or better results than competitors when applied to downstream data analysis tasks, such as outlier detection. The results of our comprehensive experiments on real-world data sets indicate that outlier detection results in a streaming setting profit from combining the results obtained in individual high-contrast subspaces. In future work, we will investigate the effect of different update schemes running the subspace search only at every k-th time-step, which reduces the runtime further.

In this work, we have systematically evaluated typical pretraining objectives for tabular deep learning. We have revealed several important recipes for optimal pretraining performance that can be universally beneficial across various problems and models. Our findings confirm that pretraining can significantly improve the performance of tabular deep models and provide additional evidence that tabular DL can become a strong alternative to GBDT.


2.3.8 How to Write the Conclusion  Answers to the questions formulated in the chapter Introduction  Same order as the questions  Recommendations  Theoretical implications  Possible practical applications = Including the significance of the work = Defining scientific truth


The challenges encountered throughout this work can be revisited in future work. This includes the Kafka configuration, Jaeger code instrumentation as well as the reproduction of the tracing data. To what extent different distributed tracing tools create different or the same results is another point for further investigation. The focus of the architecture extraction script introduced in section 6.2.1 are the span ids in order to make all communication relations visible. Other approaches taking single traces into account or considering the internal communications inside the microservices are possible avenues for future work. The use case project used in this work is written in Java and the MOM used is Kafka. The architecture extraction is based on JSON data as output from the distributed tracing tool and can be reused for other projects written in other programming languages. To which extent the instrumentation of the source code is suitable in other programming languages is also an open issue. The usage of other distributed tracing tools or projects utilizing a different type of MOM is another aspect that can be considered in future work.


2.3.7 How to Write the Discussion  Assessment of the results  Comparison of your own results with the results of other studies = Citation of already published literature!  Components  Principles, relationships, generalizations shown by the results = Discussion, not recapitulation of the results  Exceptions, lack of correlation, open points  Referring to published work: = Results and interpretations in agreement with or in contrast to your results  Our Recommendations: The writing of the chapter “Discussion” is the most difficult one. Compare your own data/results with the results from other already published papers (and cite them!). Outline the discussion part in a similar way to that in the Results section = consistency. Evaluate whether your results are in agreement with or in contrast to existing knowledge to date. You can describe why or where the differences occur, e.g. in methods, in sites, in special conditions, etc. Sometimes it is difficult to discuss results without repetition from the chapter “Results”. Then, there is the possibility to combine the “Results” and “Discussion” sections into one chapter. However, in your presentation you have to classify clearly which are your own results and which are taken from other studies. For beginners, it is often easier to separate these sections.

