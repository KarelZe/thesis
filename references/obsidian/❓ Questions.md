

## ❓Questions
**Data set:**
- Could I obtain access to the previously used data? What kind of features like volumes at certain exchanges would be available?
- How many features does the data set contain?
- Should the data set remain static or include new samples since publication? Hence, 2017 onwards for the ISE set.
- Are there certain options to put special emphasis on e. g. index options?

**Models:**
- Are there certain models that must be considered?

**Organizational:**
- What were the weakest points in the seminar paper? Where do I have to improve most?
- What are the expectations I have to meet?
- What should be the focus? Primary focus on improving prediction quality / bridging the gap? Do a fully-fledged comparison of ML methods? Come up with something creative? (see [[❓ Questions#^f9bb84]])
- What were the greatest challenges in writing the draft of the paper?
- How were quote and tick rules implemented e. g., library/custom implementation in python? How were the tables generated?
- Who would co-supervise / grade a thesis? What is his / her special focus e. g., economical inference/interpretability?
- Is it ok to start with full thrust at beginning of the semester and register for the thesis before Christmas?

## 💥Ideas

^f9bb84

### 📜Classical rules
- Enhance the mid-spread rule. Consider the distance from the mid-spread and bid-ask spread as a probability. The current formulation can't classify trades at mid-spread. A different formulation assigns a low probability to classifications close to the mean.
- Current hybrid approaches use stacking (p. 11). Also, due to technical limitations. Why not try out the majority vote/voting classifier with final estimator.

### 🧠 Machine Learning-based approaches

#### Data pre-processing
- The approach of Grauer matches the LiveVol data set, only if there is a matching volume on buyer or seller side. Results in 40 % reconstruction rate (p. 9). One could obtain more **training samples** by:
- **pseudo-labelling:** e. g., [How To Build an Efficient NLP Model – Weights & Biases (wandb.ai)](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx) and [[@leePseudolabelSimpleEfficient]]. Requires solving the issue of obtaining soft probablities.
- **synthetic data:** Use (unconditional) Generative Adversial Networks to generate tabular data, that is not part of the data set itself but could be used for testing. See e. g., [nbsynthetic/vgan.py at master · NextBrain-ml/nbsynthetic (github.com)](https://github.com/NextBrain-ml/nbsynthetic/blob/master/src/nbsynthetic/vgan.py) or [Synthetic Tabular Data Generation | by Javier Marin | Sep, 2022 | Towards Data Science](https://towardsdatascience.com/synthetic-tabular-data-generation-34eb94a992ed)
- **fuzzy matching:** e. g., match even if there are small deviations in the volumes e. g. 5 contracts.

#### Modelling
- Implement reproducible models with reproducible data sets early on. (see [[❓ Questions#^bd4973]])
- Current rules like the tick test, consider previous prices. Frame problem as time series classification problem, if previous prices/orders are available. Mind the gaps in options data though.
- Start with something simple e. g., Logistic Regression or Gradient Boosted Trees, due to being well suited for tabular data.
- Use classification methods (*probabilistic classifier*) that can return probabilities instead of class-only for better analysis.


## Tasks

## 🏫 Organizational
- Define micro-tasks
- Schedule meetings
- Apply for SCC computing resources
- Collect checklists of errors to check for e. g., inconsistent capitalization etc.

### 👨‍🚀 Technical

^bd4973

- Make research reproducible. Use [`modeldb`]([modeldb/client at main · VertaAI/modeldb (github.com)](https://github.com/VertaAI/modeldb/tree/main/client)) to persist model runs / make research reproducable. Save results to [Cloud Storage](https://cloud.google.com/storage?hl=de) and map as drive as described in [How to import data from Google Cloud Storage to Google Colab - Stack Overflow](https://stackoverflow.com/questions/51715268/how-to-import-data-from-google-cloud-storage-to-google-colab)
- Implement experiment tracking with [`weights and bias`]([Weights & Biases – Developer tools for ML (wandb.ai)](https://wandb.ai/site)) or [Overview - Verta](https://docs.verta.ai/verta/). 
- Set up `mypy` and [beartype/beartype: Unbearably fast O(1) runtime type-checking in pure Python. (github.com)](https://github.com/beartype/beartype) type to avoid errors through incorrect typing. 
- Set up consistent plotting early on e. g. style. E. g., see [garrettj403/SciencePlots: Matplotlib styles for scientific plotting (github.com)](https://github.com/garrettj403/SciencePlots).
- If using neural networks in `pytorch`, develop a deeper understanding of the profiler. See e. g., [OPT-175B: Open Pretrained Transformer | ML Coding Series - YouTube](https://www.youtube.com/watch?v=5RUOrXl3nag). 
- If using neural networks try out `flax`, [google/flax: Flax is a neural network library for JAX that is designed for flexibility. (github.com)](https://github.com/google/flax)
- Improve `tikz` skills for diagrams.
- ~~Set up pre-commit hooks early on.~~
- Set up a obisidian-pandoc workflow.
- Set up GitHub action for pdf generation.
- Write scripts to detect weasel words, fill words, improper title casing etc.

### ✍️ Content-wise
- Ask for feedback on previous work. Where are the greatest weaknesses/strengths?
- Research / understand why this research gap hasn't been addressed earlier.
- Research works on options trade classification since the draft was published.
- How do stock direction classification and stock trade classification relate?
- Fully understand why option trade classification matters.
- Research works on stock trade classification with ML
	- What feature engineering is commonly applied? Why?
	- What approaches deliver the best results?
	- Search kaggle for interesting ideas from (stock) classification challenges
	- Search google colab for similar problems
	- Try out research rabbit [ResearchRabbit](https://www.researchrabbit.ai/) 💫
	- Investigate robustness checks
	- Research approaches so I can make sure I don't fish for noise.
- Write after Gopen rules (see [Microsoft Word - Science of Scientific Writing.doc (fu-berlin.de)](http://www.inf.fu-berlin.de/lehre/pmo/eng/ScientificWriting.pdf))
- Implement a baseline e. g., rules from paper. Do I get the same results?
- Update readme.
- Sketch a rough timeline. Outline what techniques to try.
- Create expose.