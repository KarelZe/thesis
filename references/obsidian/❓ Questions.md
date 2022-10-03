

## ❓Questions
**Organizational:**
- What were the weakest points in the seminar paper? Where do I have to improve most?
- What are the expectations I have to meet?
- What should be the focus? Primary focus on improving prediction quality / bridging the gap? What would be the focus then e. g., beat [[@grauerOptionTradeClassification2022]] or the [[@savickasInferringDirectionOption2003]] in terms of accuracy?  Do a fully-fledged comparison of ML methods? Come up with something creative? (see [[❓ Questions#^f9bb84]])
- What were the greatest challenges in writing the draft of the paper?
- How were quote and tick rules implemented e. g., library/custom implementation in python? How were the tables generated?
- Who would co-supervise / grade a thesis? What is his / her special focus e. g., economical inference/interpretability?
- Is it ok to start with full thrust at beginning of the semester and register for the thesis before Christmas?
- Are there any criteria/expectations I have to meet? Are there certain ideas from ongoing research that I have to try out? Do I have to formally apply e. g., with my CV?
**Data set:**
- Could I obtain access to the previously used data? What kind of features like volumes at certain exchanges would be available?
- How many features does the data set contain? `8 categories for buy and sell volumes  x 4 trader types by option series x trading day`. 
- What should be primary target e. g., classification on `NBBO`? -> all exchanges
- Should the data set remain static or include new samples since publication? Hence, 2017 onwards for the ISE set. -> Focus on ISE 
- Are there certain options to put special emphasis on e. g., index options? Hard to classify trades e. g., large trade sizes or close to mid-spread? (see [[@grauerOptionTradeClassification2022]]) -> all
- merge data from ise + cboe -> needs features from stock exchanges

**Models:**
- Are there certain models that must be considered?

**Other:**
- What is the idea behind the two step process of publishing on Open Science Framework first? Do you want to "mark the field"? Get early feedback? [osf.io](https://osf.io/kj86r/ ?view_only=388a89b23254425a8271402e2b11fc4e.)

## 💥Ideas

^f9bb84

### 📜Classical rules
- Enhance the mid-spread rule. Consider the distance from the mid-spread and bid-ask spread as a probability. The current formulation can't classify trades at mid-spread. A different formulation assigns a low probability to classifications close to the mean.
- Current hybrid approaches use stacking ([[@grauerOptionTradeClassification2022]] p. 11). Also, due to technical limitations. Why not try out the majority vote/voting classifier with a final estimator?

### 🧠 Machine Learning-based approaches

#### 🦺 Data pre-processing
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

## 🏫 Organizational
- Define micro-tasks
- Sketch a rough timeline. Finalize scope.
- Schedule meetings
- Apply for SCC computing resources
- Collect checklists of errors to check for e. g., inconsistent capitalization etc.
- Investigate printing bug from seminar paper with complex diagrams. Try at copyshop at uni. Try compression.

### 👨‍🚀 Technical

^bd4973

- Make research reproducible. Use [`modeldb`]([modeldb/client at main · VertaAI/modeldb (github.com)](https://github.com/VertaAI/modeldb/tree/main/client)) to persist model runs / make research reproducable. Save results to [Cloud Storage](https://cloud.google.com/storage?hl=de) and map as drive as described in [How to import data from Google Cloud Storage to Google Colab - Stack Overflow](https://stackoverflow.com/questions/51715268/how-to-import-data-from-google-cloud-storage-to-google-colab).
- Implement experiment tracking with [`weights and bias`]([Weights & Biases – Developer tools for ML (wandb.ai)](https://wandb.ai/site)) or [Overview - Verta](https://docs.verta.ai/verta/). 
- Set up `mypy` and [beartype/beartype: Unbearably fast O(1) runtime type-checking in pure Python. (github.com)](https://github.com/beartype/beartype) type to avoid errors through incorrect typing. 
- Set up consistent plotting early on e. g. style. E. g., see [garrettj403/SciencePlots: Matplotlib styles for scientific plotting (github.com)](https://github.com/garrettj403/SciencePlots).
- Replace `pandas` with [`modin`](https://github.com/modin-project/modin)
- If using neural networks in `pytorch`, develop a deeper understanding of the profiler. See e. g., [OPT-175B: Open Pretrained Transformer | ML Coding Series - YouTube](https://www.youtube.com/watch?v=5RUOrXl3nag). 
- If using neural networks try out `flax`, [google/flax: Flax is a neural network library for JAX that is designed for flexibility. (github.com)](https://github.com/google/flax)
- Improve `tikz` skills for diagrams.
- ~~Set up pre-commit hooks early on.~~
- Set up a `obisidian`-`pandoc` workflow.
- Set up GitHub actions for pdf generation.
- Write scripts to detect weasel words, fill words, improper title casing etc.

### ✍️ Content-wise
- Ask for feedback on previous work. Where are the greatest weaknesses/strengths?
- Research / understand why this research gap hasn't been addressed earlier.
- Research works on options trade classification since the draft was published.
- How do stock direction classification and stock trade classification relate?
- The results of [[@savickasInferringDirectionOption2003]] seem to be to good to be true. Could the data set of [[@savickasInferringDirectionOption2003]] be reconstructed one-by-one (e. g., sample selection) first? Where could possible errors be?
- Fully understand why option trade classification matters.
- Research works on stock trade classification with ML
	- What feature engineering is commonly applied? Why?
	- What approaches deliver the best results?
	- Search `kaggle` for interesting ideas from (stock) classification challenges
	- Search `google colab` for similar problems
	- Try out research rabbit [ResearchRabbit](https://www.researchrabbit.ai/) 💫
	- Investigate robustness checks
	- Research approaches so I can make sure I don't fish for noise.
	- Investigate how results behave over time. Do they deterioriate?
- Write after Gopen rules (see [Microsoft Word - Science of Scientific Writing.doc (fu-berlin.de)](http://www.inf.fu-berlin.de/lehre/pmo/eng/ScientificWriting.pdf))
- Implement a baseline e. g., rules from paper. Do I get the same results?
- Update readme.
- Create expose -> problem, empirisch, ml algorithmus
- Think about leakage between sets when doing splits.

- feedback
	- Ergebnisse ökonomisch einordnet
	- Zusammenhang zwischen der Arbeiten
	- Schachtelsätze
	- Asset-Pricing-Formula -> Intuition 
	- Am Anfang die Ergebnisse stellen / Hauptlearning am Ende wiederholen