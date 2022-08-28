

## Questions
- Why is the 2-tep process relevant?
- Could I obtain access to data?
- How many features does the data set contain? Which ones does it ok?
- How were quote and tick rules implemented e. g. library / custom implementation in python? How are the tables generated?
- What were the greatest challenges writing the draft of the paper?
- Ist it ok to stark with full thrust at beginning of semester?
- What should be the focus one? Primary focus on improving prediction quality / bridging gap?

## Tasks
- Ask for feedback on previous work. Where are the greatest weaknesses / strengthes?
- Research works on option trade classification since draft was published
- How do stock direction classification and stock trade classification relate?
- Fully understand why option trade classification matters
- Research works on stock trade classification with ML
	- What feature engineering is commonly applied? Why?
	- What approaches deliver the best results?
	- search kaggle for interesting ideas from (stock) classification challenges
	- Try out reseach rabbit [ResearchRabbit](https://www.researchrabbit.ai/) 💫
- Implement a baseline e. g., rules from paper
- Update readme
- Sketch rough timeline
- Create expose

## Technical Todos

- Make research reproducable. Use [`modeldb`]([modeldb/client at main · VertaAI/modeldb (github.com)](https://github.com/VertaAI/modeldb/tree/main/client)) to persist model runs / make research reproducable. Save results to [Cloud Storage](https://cloud.google.com/storage?hl=de) and map as drive as described in [How to import data from Google Cloud Storage to Google Colab - Stack Overflow](https://stackoverflow.com/questions/51715268/how-to-import-data-from-google-cloud-storage-to-google-colab)
- Implement experiment tracking with [`weights and bias`]([Weights & Biases – Developer tools for ML (wandb.ai)](https://wandb.ai/site)) or [Overview - Verta](https://docs.verta.ai/verta/). 
- Set up mypy and [beartype/beartype: Unbearably fast O(1) runtime type-checking in pure Python. (github.com)](https://github.com/beartype/beartype) type to avoid errors through incorrect typing. 
- Set up consisten plotting early on.
- Set up precommit hooks early on.
- Set up a obisidian-pandas workflow.
- Try out pseudo-labelling e. g., [How To Build an Efficient NLP Model – Weights & Biases (wandb.ai)](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx) and [[@leePseudolabelSimpleEfficient]]. Requires solving the issue of obtaining soft probablities.
- Write after Gopen rules [Microsoft Word - Science of Writing.rtf (ucsd.edu)](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- Write scripts to detect weasel words, fill words, improper titlecasing etc.

## Problems
- How can the disadvantages of generating the true labels be mitigated e. g., matching trade size, not just buys / sells?
- Investigate if yet another low signal problem?
- Investigate if other robustness checks are available?