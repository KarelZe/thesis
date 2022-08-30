

## Questions
- Why is the 2-step process relevant?
- Could I obtain access to data?
- Should the data set remain static or include new samples since publication?
- How many features does the data set contain? Which ones does it ok?
- How were quote and tick rules implemented e. g. library / custom implementation in python? How are the tables generated?
- What were the greatest challenges writing the draft of the paper?
- Who would supervise / grade a thesis? What is his / her special focus?
- Is it ok to start with full thrust at beginning of semester?
- What should be the focus one? Primary focus on improving prediction quality / bridging gap?
- Are there certain options to put special emphasize on e. g. index options?
- Are there certain models that must be considered?

## Tasks

### Technical
- Make research reproducable. Use [`modeldb`]([modeldb/client at main · VertaAI/modeldb (github.com)](https://github.com/VertaAI/modeldb/tree/main/client)) to persist model runs / make research reproducable. Save results to [Cloud Storage](https://cloud.google.com/storage?hl=de) and map as drive as described in [How to import data from Google Cloud Storage to Google Colab - Stack Overflow](https://stackoverflow.com/questions/51715268/how-to-import-data-from-google-cloud-storage-to-google-colab)
- Implement experiment tracking with [`weights and bias`]([Weights & Biases – Developer tools for ML (wandb.ai)](https://wandb.ai/site)) or [Overview - Verta](https://docs.verta.ai/verta/). 
- Set up mypy and [beartype/beartype: Unbearably fast O(1) runtime type-checking in pure Python. (github.com)](https://github.com/beartype/beartype) type to avoid errors through incorrect typing. 
- Set up consistent plotting early on e. g. style.
- Improve tikz skills for diagrams.
- Set up pre-commit hooks early on.
- Set up a obisidian-pandoc workflow.
- Write scripts to detect weasel words, fill words, improper title casing etc.

### Content-wise
- Ask for feedback on previous work. Where are the greatest weaknesses / strengths?
- Research / understand why this research gap hasn't been adressed earlier.
- Research works on option trade classification since draft was published.
- How do stock direction classification and stock trade classification relate?
- Fully understand why option trade classification matters.
- Research works on stock trade classification with ML
	- What feature engineering is commonly applied? Why?
	- What approaches deliver the best results?
	- Search kaggle for interesting ideas from (stock) classification challenges
	- Search google colab for similar problems
	- Try out reseach rabbit [ResearchRabbit](https://www.researchrabbit.ai/) 💫
- Write after Gopen rules [Microsoft Word - Science of Writing.rtf (ucsd.edu)](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- Implement a baseline e. g., rules from paper. Do I get the same results?
- Update readme.
- Sketch rough timeline. Outline what techniques to try when?
- Create expose.

## Ideas
- Try out pseudo-labelling e. g., [How To Build an Efficient NLP Model – Weights & Biases (wandb.ai)](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx) and [[@leePseudolabelSimpleEfficient]]. Requires solving the issue of obtaining soft probablities.
- Implement re-producable models.
- Start of with Gradient Boosted Trees, due to being well suited for tabular data.

## Problems
- How can the disadvantages of generating the true labels be mitigated e. g., matching trade size, not just buys / sells?
- Investigate if this is yet another low signal problem?
- Investigate if other robustness checks are available?