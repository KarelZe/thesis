We present our research framework for trade classification, summarized in cref-fig-research-framework. (Centered around two ideas: gls-gbrt + transformer (1) and TCR are just a classifier (2)) At its core, we utilize gls-gbrt and Transformers for trade classification, chosen in cref-selection-supervised for their expected performance, scalability, and extensibility. Classical trade classification rules are realised as a generic classifier exploiting the stacking principle, ensuring a coherent evaluation. The data preparation process, outlined in cref-sec-data-and-data-preparation, covers all necessary steps to obtain data for consumption by the classifiers. Model enhancements, training setups, and tuning procedures are detailed in cref-sec-models. The predictions of the classifiers are consistently evaluated in terms of accuracy as part of cref-sec-evaluation. With the model-agnostic importance measure SAGE, we are able to attribute predictions to features and even cross-compare the feature importances between classical trade classification rules and machine learning predictors. Additionally, the model-specific attention maps provide further insights into Transformers. Lastly, in cref-section-application-study, we test all classifiers in the problem of effective spread calculation to demonstrate the effectiveness of our approach.

![[Research Framework.png]]

In this Section, we demonstrate the effectiveness of gradient boosting and transformer-based approaches for trade classification. We benchmark against the LR algorithm, EMO algorithm, CLNV algorithm, as well as hybrids involving the trade size rule and depth rule on two datasets of option trades recorded at the ISE and CBOE.

Experiments were conducted on nodes of the *bwHPC cluster* running Ubuntu 20.04, with x processor x.xGHz, x GB RAM,  and 4 Nvidia Tesla V100. For reproducibility the implementation and experiment tracking is publicly available. -footnote(wandb + github)

The subsequent section provides further details on the dataset.

**Notes:**
[[🌏Environment notes]]