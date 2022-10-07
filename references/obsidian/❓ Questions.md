
## ❓Questions
**Organizational:**
- What are the expectations I have to meet in order to reach 1.3 or better?
- What should be the focus? Primary focus on improving prediction quality / bridging the gap? What would be the focus then e. g., beat [[@grauerOptionTradeClassification2022]] or the [[@savickasInferringDirectionOption2003]] in terms of accuracy?  Do a fully-fledged comparison of ML methods? Come up with something creative? (see [[❓ Questions#^f9bb84]])
- What were the greatest challenges in writing the draft of the paper?
- Who would co-supervise / grade a thesis? What is his / her special focus e. g., economical inference/interpretability?

**Data set:**
- Could I obtain access to the previously used data? What kind of features like volumes at certain exchanges would be available?
- How many features does the data set contain? `8 categories for buy and sell volumes  x 4 trader types by option series x trading day`. 
- What should be primary target e. g., classification on `NBBO`? -> all exchanges
- Should the data set remain static or include new samples since publication? Hence, 2017 onwards for the ISE set. -> Focus on ISE 
- merge data from ise + cboe -> needs features from stock exchanges

## 💥Ideas

^f9bb84

### 📜Classical rules
- Enhance the mid-spread rule. Consider the distance from the mid-spread and bid-ask spread as a probability. The current formulation can't classify trades at mid-spread. A different formulation assigns a low probability to classifications close to the mean.
- Current hybrid approaches use stacking ([[@grauerOptionTradeClassification2022]] p. 11). Also, due to technical limitations. Why not try out the majority vote/voting classifier with a final estimator?

### 🧠 Machine Learning-based approaches

#### 🦺 Data pre-processing
- Perform EDA e. g., [AutoViML/AutoViz: Automatically Visualize any dataset, any size with a single line of code. Created by Ram Seshadri. Collaborators Welcome. Permission Granted upon Request. (github.com)](https://github.com/AutoViML/AutoViz) and [lmcinnes/umap: Uniform Manifold Approximation and Projection (github.com)](https://github.com/lmcinnes/umap)
- The approach of [[@grauerOptionTradeClassification2022]] matches the LiveVol data set, only if there is a matching volume on buyer or seller side. Results in 40 % reconstruction rate [[@grauerOptionTradeClassification2022]](p. 9). One could obtain more **training samples** by:
- **pseudo-labelling:** e. g., [How To Build an Efficient NLP Model – Weights & Biases (wandb.ai)](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx) and [[@leePseudolabelSimpleEfficient]]. Requires solving the issue of obtaining soft probablities.
- **synthetic data:** Use (unconditional) Generative Adversial Networks to generate tabular data, that is not part of the data set itself but could be used for training. See e. g., [nbsynthetic/vgan.py at master · NextBrain-ml/nbsynthetic (github.com)](https://github.com/NextBrain-ml/nbsynthetic/blob/master/src/nbsynthetic/vgan.py) or [Synthetic Tabular Data Generation | by Javier Marin | Sep, 2022 | Towards Data Science](https://towardsdatascience.com/synthetic-tabular-data-generation-34eb94a992ed)
- **fuzzy matching:** e. g., match volumes, even if there are small deviations in the volumes e. g. 5 contracts. Similar technique used for time stamps in [[@savickasInferringDirectionOption2003]].

#### 🏗️ Modelling
- Implement reproducible models with reproducible data sets early on. (see [[❓ Questions#^bd4973]])
- Current rules like the tick test, consider previous prices. Frame problem as time series classification problem, if previous prices/orders are available. Mind the gaps in options data though. Would somewhat contract the poor experiences with the tick rule.
- Start with something simple e. g., Logistic Regression or Gradient Boosted Trees, due to being well suited for tabular data. Implement robustness checks (as in [[@grauerOptionTradeClassification2022]]) early on.
- If using neural networks, try out feed forward networks or Transformers in the form of TabTransformer. Look into [snapshot ensembles](https://arxiv.org/pdf/1704.00109.pdf).
- Use classification methods (*probabilistic classifier*) that can return probabilities instead of class-only for better analysis.

## 🔔 Tasks
^bd4973
Migrated to [Issues · KarelZe/thesis (github.com)](https://github.com/KarelZe/thesis/issues)


