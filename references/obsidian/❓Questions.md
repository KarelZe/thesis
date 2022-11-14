- Differences in accuracy for classical rules between the implementations exist. Do you assign classes randomly for trades that can not be classified with the classification rule? (`0` or `np.random.choice([-1, 1]`) How did you define "success rate"?
- How did you define "others" in table 9? Using special codes?
- Could I please get the current stock price for moneyness to integrate them in my robustness checks?
- Symbol / root is somewhat problematic, as some are only in the train set or test set. Could still use root and rely on embedding or use special codes as features. Might be wise to use a more generic feature like sector instead.
- Which offset was used when matching transactions and quotes? None?

## Closed
- What are the expectations I have to meet in order to reach $\geq 1.3$?
- Any feedback to toc / expose / first results? -> *It's ok.*
- Opinions on weekly release info 📧 (e. g., every sunday) with closed issues and completed tasks + short bi-weekly meeting. -> *Meeting scheduled. Release notes sent.*
- Who would co-supervise / grade the thesis? What is his / her special focus e. g., economical inference/interpretability? -> *Prof. Dr. Uhrig-Homburg (1), Prof. Dr. Ruckes (2). Prof. Uhrig-Homburg is very open to new ideas.* 
- Discuss what to do with low-quality papers e. g., [[@ronenMachineLearningTrade2022]] or [[@blazejewskiLocalNonparametricModel2005]]? Cite, but be critical? Is it ok to also leave some poor papers out? -> *Ok, to leave out or point out what is problematic.*
- What to do with [[@hansenApplicationsMachineLearning]]? Thesis does something similar but was not published. Mostly different techniques / different data set / focus on EDA.
- Discuss citations of pre-prints? Some important concepts in ML have only been published on [archive.org](www.archive.org). -> *Citing preprints is ok.*