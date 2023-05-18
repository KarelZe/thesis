We present our research framework for trade classification, summarized in cref-fig-research-framework. Our approach revolves around two key ideas. First, we utilize gls-gbrt and Transformers for trade classification, chosen in cref-selection-supervised for their expected performance, scalability, and extensibility. Distinctions are made between models trained on labelled and unlabelled trades simultaneously.  Second, classical trade classification rules, such as the gls-LR algorithm, are realised as a generic classifier leveraging the stacking principle, thereby enabling a coherent evaluation and model interpretation. 

![[Research Framework.png]]

The data preparation process, outlined in cref-sec-data-and-data-preparation, encompasses all steps necessary to obtain features to be processed by the classifiers. Model enhancements, training setups, and tuning procedures are detailed in cref-sec-models. The predictions of the classifiers are consistently evaluated in terms of accuracy as part of cref-sec-evaluation. With the model-agnostic interpretability method SAGE, we attribute predictions to features and cross-compare the feature importances of classical trade classification rules and machine learning predictors. In turn, for Transformers attention maps provide additional insights into the model. Lastly, cref-section-application-study tests all classifiers in the problem of effective spread calculation to demonstrate the effectiveness of our approach.

The subsequent section provides further details on the dataset.

**Notes:**
[[🌏Environment notes]]