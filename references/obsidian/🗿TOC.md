- [nature-summary-paragraph.pdf](https://www.nature.com/documents/nature-summary-paragraph.pdf)
- Guide on visualizations https://www.nature.com/articles/s41467-020-19160-7
- Guide on storytelling with data https://www.practicedataviz.com/pdv-evd-mvp#PDV-EVD-mvp-g
- For visualization see: https://www.data-to-viz.com
- see  `writing-a-good-introduction.pdf`
- Prepare final pdf https://github.com/google-research/arxiv-latex-cleaner
- Adhere to best practices http://www.sigplan.org/Resources/EmpiricalEvaluation/
- use colors in formulas🎨. See [[@patrignaniWhyShouldAnyone2021]]
- https://www.molecularecologist.com/2020/04/23/simple-tools-for-mastering-color-in-scientific-figures/
- look into [[@lonesHowAvoidMachine2022]]

# Title
Forget About the Rules: Improving Trade Side Classification With Machine Learning

# Introduction

[[👶Introduction]]

# 👨‍👩‍👧‍👦 Related Work 🟢
[[👪Related Work]]
# 🔗Rule-Based Approaches

The following section introduces common rules for signing option trades. We start by introducing the prevailing quote and tick rule and continue with the recently introduced depth and trade size rule. In section [[#^a043d0]] we combine hybrids thereoff. We draw a connection to ensemble learning.

## Basic Rules

[[🔢Basic rules]]
### Quote Rule
[[🔢Quote rule]]
### Tick Test
[[🔢Tick test]]
### Depth Rule 🟢
[[🔢Depth rule]]

### Trade Size Rule
[[🔢Tradesize Rule]]
## Hybrid Rules

^a043d0

^ce4ff0
[[🔢Hybrid rules]]
### Lee and Ready Algorithm

^370c50
[[🔢LR algorithm notes]]

### Ellis-Michaely-O’Hara Rule
[[🔢EMO rule]]
### Chakrabarty-Li-Nguyen-Van-Ness Method
[[🔢CLNV method]]

# 🧠 Supervised Approaches
## Selection of Approaches

^d8f019
[[🥠Selection of approaches (supervised)]]

## Gradient Boosted Trees  🟢

In our attempt to compare shallow with deep architectures for trade classification, we first introduce gradient boosting trees, an ensemble of decision trees.

### Decision Tree 🟡

^5db625
[[🎄Decison_trees]]
### Gradient Boosting Procedure 🟡
[[🐈Gradient Boosting]]
## Transformer

### Architectural Overview

### Token Embedding

### Positional Encoding

### Residual Connections

### Layer Norm



### Position-wise FFN 🟢
[[🎱Position-wise FFN]]
### Attention
[[🅰️Attention]]
### Positional Encoding 🟢
[[🧵Positional Embedding]]
### Embeddings 🟢
[[🛌Token Embedding]]

### Extensions in TabTransformer🟢
[[🤖TabTransformer]]
### Extensions in FTTransformer🟢
[[🤖FTTransformer]]

# 👽 Semi-Supervised Approaches

## Selection of Approaches

^c77130
[[🥠Selection of approaches (semi-supervised)]]
## Extensions to Gradient Boosted Trees
[[🐈Extension to Gradient Boosting]]

## Extensions to TabTransformer 

## Extension to FTTransformer
# 👒 Empirical Study
- In the subsequent sections we apply methods from (...) in an empirical setting.
## Environment 🟡
[[🌏Environment]]

## Data and Data Preparation 🟡

- present data sets for this study
- describe applied pre-processing
- describe and reason about applied feature engineering
- describe and reason about test and training split
### ISE Data Set 🟡
[[🌏ISE dataset]]

### CBOE Data Set 🟡
describe if data set is actually used. Write similarily to 

### Exploratory Data Analysis
[[🚏Exploratory Data Analysis]]

### Data Pre-Processing 🟡
[[👨‍🍳Pre-Processing]]

### Feature Engineering 🟡
[[🪄Feature Engineering]]

### Train-Test Split 🟡

^d50f5d
[[👨‍🍳Tain-Test-split]]
## Training and Tuning
[[💡Training and tuning]]

### Training of Supervised Models
[[💡Training of models (supervised)]]
### Training of Semi-Supervised Models
[[💡Tuning of models (semi-supervised)]]
### Hyperparameter Tuning
[[💡Hyperparameter tuning]]

## Evaluation
### Feature Importance Measure
[[🏅Feature importance measure]]
### Evaluation Metric
[[🧭Evaluation metric]]
[[🧭Attention map]]

# 🏁 Results
[[🏅Results]]
## Results of Supervised Models
[[🏅Results of supervised]]
## Results of Semi-Supervised Models
[[🏅Results of semi-supervised]]

## Robustness of Results
[[🏅Robustness]]

## Feature Importance
[[🏅Feature importance results]]

## Ablation Study of Models

# 🍕 Application in Transaction Cost Estimation
[[🍕Application study]]
## Simulation Setup
## Simulation Results

# 💣Discussion
[[🧓Discussion]]
# Conclusion
- Repeat the problem and its relevance, as well as the contribution (plus quantitative results).
# 🌄Outlook
- Provide an outlook for further research steps.