- [nature-summary-paragraph.pdf](https://www.nature.com/documents/nature-summary-paragraph.pdf)
- Guide on visualizations https://www.nature.com/articles/s41467-020-19160-7
- Guide on storytelling with data https://www.practicedataviz.com/pdv-evd-mvp#PDV-EVD-mvp-g
- For visualization see: https://www.data-to-viz.com
- see  `writing-a-good-introduction.pdf`
- Prepare final pdf https://github.com/google-research/arxiv-latex-cleaner
- Adhere to best practices http://www.sigplan.org/Resources/EmpiricalEvaluation/

# Title
Forget About the Rules: Improving Trade Side Classification With Machine Learning

# Introduction

[[introduction]]

# 👨‍👩‍👧‍👦 Related Work
[[related_work]]
# 🔗Rule-Based Approaches

The following section introduces common rules for signing option trades. We start by introducing the prevailing quote and tick rule and continue with the recently introduced depth and trade size rule. In section [[#^a043d0]] we combine hybrids thereoff. We draw a connection to ensemble learning.

## Basic Rules

[[basic_rules]]
### Quote Rule
[[quote_rule]]
### Tick Test
[[tick_test]]
### Depth Rule 🟢
![[depth_rule]]

### Trade Size Rule
[[tradesize_rule]]
## Hybrid Rules

^a043d0

^ce4ff0
[[hybrid_rules]]
### Lee and Ready Algorithm

^370c50
[[lr_algorithm]]

### Ellis-Michaely-O’Hara Rule
[[emo_rule]]
### Chakrabarty-Li-Nguyen-Van-Ness Method
[[clnv_method]]

# 🧠 Supervised Approaches
## Selection of Approaches

^d8f019
[[selection_of_approaches]]

## Gradient Boosted Trees  🟢

In our attempt to compare shallow with deep architectures for trade classification, we first introduce gradient boosting trees, an ensemble of decision trees.

### Decision Tree 🟡

^5db625
![[🎄decison_trees]]
### Gradient Boosting Procedure 🟡
![[🐈gradient-boosting]]
## Transformer Networks

[[transformer_networks]]
### Network Architecture
[[network_architecture]]
### Attention
[[attention]]
### Positional Encoding
### Embeddings
[[embeddings]]
### Extensions in TabNet
[[extensions-to-tabnet]]

### Extensions in TabTransformer
[[extensions-to-tabtransformer]]
### Extensions in FTTransformer
[[🤖pretraining-FTTransformer]]

# 👽 Semi-Supervised Approaches

## Selection of Approaches

^c77130
[[selection-of-approaches]]
## Extensions to Gradient Boosted Trees
[[extensions-to-gradient-boosting]]
## Extensions to TabNet
[[extensions-to-tabnet]]
## Extensions to TabTransformer
[[extensions-to-tabtransformer]]
## Extension to FTTransformer
# 👒 Empirical Study
- In the subsequent sections we apply methods from (...) in an empirical setting.
## Environment 🟡
[[environment]]

## Data and Data Preparation 🟡

- present data sets for this study
- describe applied pre-processing
- describe and reason about applied feature engineering
- describe and reason about test and training split
### ISE Data Set 🟡
[[ise-dataset]]

### CBOE Data Set 🟡
describe if data set is actually used. Write similarily to 

### Exploratory Data Analysis
[[🚏exploratory data analysis]]

### Data Pre-Processing 🟡
[[preprocessing]]

### Feature Engineering 🟡
![[🪛feature engineering]]

### Train-Test Split 🟡

^d50f5d
[[train-test-split]]
## Training and Tuning
[[training_and_tuning]]

### Training of Supervised Models
[[training-of-supervised-models]]
### Training of Semi-Supervised Models
[[tuning_of_semisupervised]]
### Hyperparameter Tuning
[[hyperparametertuning]]

## Evaluation
### Feature Importance Measure
[[feature_importance_measure]]
### Evaluation Metric
[[evaluation_metric]]

# 🏁 Results
[[results]]
## Results of Supervised Models
[[results_of_supervised]]
## Results of Semi-Supervised Models
[[results_of_semisupervised]]

## Robustness of Results
[[robustness]]

## Feature Importance
[[feature_importance]]

![[🧭Attention map]]

## Ablation Study of Models

# 🍕 Application in Transaction Cost Estimation
[[🍕Application study]]
## Simulation Setup
## Simulation Results

# 💣Discussion
[[discussion]]
# Conclusion
- Repeat the problem and its relevance, as well as the contribution (plus quantitative results).
# 🌄Outlook
- Provide an outlook for further research steps.