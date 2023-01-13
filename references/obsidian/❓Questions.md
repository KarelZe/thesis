## Open
- When bid is zero at ISE exchange. Does it impact calculation of absolute spread?
- Ask what the `day_vol` feature is. Found no meta data for it. 
- Discuss correlated features in feature importance calculation.
- What should I do with the CBOE data set?

## Closed
- Ask for feedback regarding the toc. Indicate where I deviate from my initial expose (i. e., swap TabNet for FTTransformer due to slow training and no implementation of Rosenthal's rule due to low importance. Added ablation study, simulation, and list of algorithms). -> ok, but might be to fine-grained. Use bold text instead of chapters.
- I marked TabNet as optional in my toc, due to the very slow training / convergence. Architecture is hard to optimize, as e. g., some gradients are hand-crafted and no approximations available. Similar expected performance. -> Change is ok.
- Was able to improve the test accuracy of gradient-boosting approach to 72.84 % (ca. 6 % above SOTA) on the test set. Would this be sufficient for the thesis? It's hard to squeeze out more accuracy from these few features. -> Figures are ok.
- Discuss the idea of describing all rules as algorithms for preciseness. -> Ok, but not just. Add text as well.
- Request final dataset e. g., CBOE data for comparison and unlabeled dataset for implementing and testing pre-training routines. -> Will provide. Might take some time.

- How is the "time from the previous trade" calculated in table 9? Are there any restrictions regarding the option or underlying? -> time to the previous trade of the same option, as used in the tick rule.
- Are there other master's students at the chair to share ideas with? There is no other student with similar topic.
- Minor differences in accuracy for classical rules between the reported figures from the paper and my implementation. Differences are usually $\leq 1.3~\%$  (see [here.](https://github.com/KarelZe/thesis/blob/main/notebooks/4.0a-mb-classical_rules.ipynb)). I suspect the differences to come from `NaN` values, if e. g., `price_ex_lag` is missing, I would not classify using tick rule and assign a random class using `np.random.choice([-1, 1]`. In Grauer et al. for the tick rule the percentage of unclassified trades is $0~\%$ (see table 3). -> Note, there were some minor typos.
- Do you have a preference regarding eda? I would do it on the training set only, then check if engineered features work on validation set. I currently use the whole data set, but plan to switch (see [here](https://github.com/KarelZe/thesis/blob/feature-engineering/notebooks/3.0a-mb-data_preprocessing_explanatory_data_analysis.ipynb)).  Different views possible (see e. g., [here](https://stats.stackexchange.com/questions/424263/should-exploratory-data-analysis-include-validation-set)). -> Training set only.

- How did you define "others" in table 9? Using special codes? -> From data on underlying. Will receive the additional feature.
- Could I please get the current stock price for moneyness to integrate them in my robustness checks? -> Yes, will receive the additional feature.
- Symbol / root is somewhat problematic, as some are only in the train set or test set. Could still use root and rely on embedding or use special codes as features. Might be wise to use a more generic feature like sector instead.
- What are the expectations I have to meet in order to reach $\geq 1.3$?
- Any feedback to toc / expose / first results? -> *It's ok.*
- Opinions on weekly release info 📧 (e. g., every sunday) with closed issues and completed tasks + short bi-weekly meeting. -> *Meeting scheduled. Release notes sent.*
- Who would co-supervise / grade the thesis? What is his / her special focus e. g., economical inference/interpretability? -> *Prof. Dr. Uhrig-Homburg (1), Prof. Dr. Ruckes (2). Prof. Uhrig-Homburg is very open to new ideas.* 
- Discuss what to do with low-quality papers e. g., [[@ronenMachineLearningTrade2022]] or [[@blazejewskiLocalNonparametricModel2005]]? Cite, but be critical? Is it ok to also leave some poor papers out? -> *Ok, to leave out or point out what is problematic.*
- What to do with [[@hansenApplicationsMachineLearning]]? Thesis does something similar but was not published. Mostly different techniques / different data set / focus on EDA.
- Discuss citations of pre-prints? Some important concepts in ML have only been published on [archive.org](www.archive.org). -> *Citing preprints is ok.*