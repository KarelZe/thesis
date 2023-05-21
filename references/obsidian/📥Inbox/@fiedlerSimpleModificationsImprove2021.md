*title:* Simple Modifications to Improve Tabular Neural Networks
*authors:* James Fiedler
*year:* 2021
*tags:* #gbm #transformer #regularisation
*status:* #📥
*related:*
- [[@arikTabnetAttentiveInterpretable2020]]
- [[@huangTabTransformerTabularData2020]]
- [[@kadraWelltunedSimpleNets2021]](similar idea)
## Notes 

## Annotations


“Gradient boosted decision trees (GBDTs) are very good general-purpose models, and in fact are frequently used by tabular deep learning models as both inspiration and the standard by which to measure performance.” ([Fiedler, 2021, p. 1](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=1&annotation=N94MMVIN))

“Possibly the biggest advantage is that neural networks have the potential to be end-to-end learners, removing the need for manual categorical encoding or other feature engineering. NN models also allow more training options than GBDTs. For example, they allow continual training on streaming data, and more generally allow adjustment of learnt NN parameters by training on new data. Neural network models are also better suited to unsupervised pre-training; for example, see (Arik and Pfister 2020) and (Huang et al. 2020)” ([Fiedler, 2021, p. 1](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=1&annotation=PQEZJP4Z))

“Several simple modifications that make MLP, PNN, and AutoInt models perform on par with, or better than, recent general-purpose tabular NNs and GBDTs 2. Model comparisons across a broad range of datasets showing the effectiveness of the proposed modifications 3. A demonstration of how one modification in particular contributes to model interpretability” ([Fiedler, 2021, p. 2](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=2&annotation=5WET9SMG))

“GBN allows the use of large batch sizes, but with batch norm parameters calculated on smaller sub-batches. One big motivation for using GBN here is to speed up training, but (Hoffer, Hubara, and Soudry 2018) also showed that GBN improves generalisation when using large batch sizes.” ([Fiedler, 2021, p. 2](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=2&annotation=Y4XFHB8D))

“Leaky Gates will also be used in all of the models. These are a combination of two simple elements, an element-wise linear transformation followed by a LeakyReLU activation.” ([Fiedler, 2021, p. 2](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=2&annotation=9T8WMGI6))

“The results for MLP+, PNN, and AutoInt are compared against • Logistic regression (LR) • Gradient boosted decision trees, specifically, LightGBM (Ke et al. 2017) • A simple MLP model, created by removing layers from the TabTransformer model (see (Huang et al. 2020), §3.1, paragraph 1) • A sparse MLP, based on (Morcos et al. 2019) • TabTransformer (Huang et al. 2020) • TabNet (Arik and Pfister 2020) • Variational Information Bottleneck (VIB) (Alemi et al. 2017)” ([Fiedler, 2021, p. 5](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=5&annotation=LI7KCAVJ))

“In particular, AutoInt, PNN, and MLP+ seem to outperform the recently-introduced TabTransformer and TabNet models.” ([Fiedler, 2021, p. 5](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=5&annotation=4PKLN9FQ))

“The datasets and experiment setup were taken from (Huang et al. 2020) specifically to avoid choices that potentially favoured the new models over, e.g., TabTransformer or TabNet.” ([Fiedler, 2021, p. 5](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=5&annotation=BGIT7Z6F))

“A very recent article (Kadra et al. 2021) showed that MLPs with a “cocktail” of regularisation strategies can obtain excellent performance. In that paper the MLP architecture was fixed and optimisation focused on selecting regularisation techniques from five categories: 1. Weight decay, e.g., \`1, \`2 regularisation 2. Data augmentation, e.g., Cut-Out (DeVries and Taylor 2017) and Mix-Up (Zhang et al. 2018) 3. Model averaging, e.g., dropout and explicit average of models 4. Structural and linearization, e.g., skip layers 5. Implicit, e.g., batch normalisation The approach was tested on a large collection of datasets and obtained better overall performance than GBDTs from XGBoost (Chen and Guestrin 2016) and models such as TabNet, Neural Oblivious Decision Ensembles (NODE), (Popov, Morozov, and Babenko 2019), and DNF-Net (Abutbul et al. 2020). There are a lot of similarities between those results and the results here. The MLP+, PNN, and AutoInt models have techniques from 3 of the 5 regularisation categories: dropout, skip layers, batch normalisation (via Ghost Batch Norm)7, and explicit averaging of sub-components. The results here showed that the modified MLP+ outperformed GBDTs (but using LightGBM instead of XGBoost). The results here go one step further and show that the modifications used for MLP+ can also improve other tabular neural network models” ([Fiedler, 2021, p. 8](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=8&annotation=DVSPMB8T))

“The results show that GBDTs generally outperform any single recent neural network. Another interesting finding is that the recent neural networks generally perform much better on datasets from their own papers. In other words, the recent tabular neural networks that they look at do not seem to generalise well. One lesson to take away is that the modified models introduced here should be tested on a variety of additional datasets.” ([Fiedler, 2021, p. 9](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=9&annotation=KRDEM6EN))

“Another recent paper, (Gorishniy et al. 2021), compares GBDTs and tabular neural networks, and argues that GBDTs and neural networks perform well on different problems. It uses GBDTs from XGBoost and CatBoost (Prokhorenkova et al. 2018), and TabNet, NODE, AutoInt (not the modified version used in the current paper) and other neural network models. The neural network models performed better when data was “homogeneous”, when the concepts measured in the data were the same or very similar from feature to feature. For example, images with each pixel location as a different field would be homogeneous. GBDTs performed better when features were “heterogeneous” ([Fiedler, 2021, p. 9](zotero://select/library/items/2WDR69I9)) ([pdf](zotero://open-pdf/library/items/LTL6RA8D?page=9&annotation=ZZ5CYMWU))