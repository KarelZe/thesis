
title: Bayesian Optimisation is Superior to Random Search for Machine Learning Hyperparameter Tuning: Analysis of the Black-Box Optimisation Challenge 2020
authors: Ryan Turner, David Eriksson, Michael McCourt, Juha Kiili, Eero Laaksonen, Zhen Xu, Isabelle Guyon
year: 2021
tags :  #bayesian-search #grid-search #hyperparameter-tuning #randomised-search
status : #📦 
related: 
- [[@banachewiczKaggleBookData2022]]
- [[@akibaOptunaNextgenerationHyperparameter2019]]
- [[@gorishniyRevisitingDeepLearning2021]] (found paper here)
- [[@mockusApplicationBayesianApproach1994]] (discusses Bayesian search in finding hyperparams)

## Notes📍

- Authors organised the first black-box optimisation challenge. Only (local) practise were visible to the participants, whereas the feedback and final problems were not. There was a feedback session similar to a public leaderboard, where participants could learn from similar entries. The authors refer to this as warm starting.
- Two baselines (TuRBO and Randomised Search) were provided. 61 entries outperformed random search and 23 outperformed TuRBO.
- Bayesian optimisation can use only one surrogate model and one acquisition function or combine several methods through ensembling.
- **Conclusion:**
	- Surrogat-assisted optimisation is very effective. All participants ranking in the top 20 used surrogate-assisted optimisation.
	- While Randomised Search could outperform the [[Bayesian optimisation]] solutions, it requires much more function evaluations to achieve the same.
	- **The top submissions showed over 100x sample effiency gains compared to randomised search.**
	- Warm starting (learning on similar problems) yielded large performance gains.
	- Ensembling can further improve performance. All of the top-ranking participants used ensembles. Some are conceptionally easy and simple to implement.

## Annotations 📖

“In black-box optimisation we aim to solve the problem minx∈Ω f (x), where f is a computationally expensive black-box function and the domain Ω is commonly a hyper-rectangle.” ([Turner et al., 2021, p. 1](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=1&annotation=R64GP3FK))

“Using this surrogate model, an acquisition function is used to determine the most promising point to evaluate next, where popular options include expected improvement (EI) [35], knowledge gradient (KG) [18], and entropy search (ES) [31]. There are also other surrogate optimisation methods that rely on deterministic surrogate models such as radial basis functions [16, 66], see Forrester et al. [17] for an overview.” ([Turner et al., 2021, p. 1](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=1&annotation=X9KGT3SL))

“This is a new competition as there have been no ML-oriented black-box optimisation competitions in the past.3 Th” ([Turner et al., 2021, p. 3](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=3&annotation=JTGYB42X))

“We obtain novel optimisation problems via the Cartesian product of datasets, ML models, and evaluation metrics” ([Turner et al., 2021, p. 4](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=4&annotation=XBMIBB2F))

“Note that only the (local) practise problems were visible to the participants; both the feedback and final problems were hidden.” ([Turner et al., 2021, p. 4](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=4&annotation=NDQUWVFN))

“For the local practise optimisation problems, the evaluation of the objective functions was done on the participants’ hardware. However, for the test problems (the feedback and final leaderboards), the objective function had to be hidden, and therefore the participants’ submissions were run inside a Docker container in a cloud environment.” ([Turner et al., 2021, p. 5](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=5&annotation=657SG372))

“61 beat the baseline random search and 23 beat TuRBO which was the strongest baseline provided in the starter kit.” ([Turner et al., 2021, p. 5](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=5&annotation=4FCTZ2RI))

“Note that this comparison does not necessarily show that one package is better than another; it rather compares the performance of their default methods.” ([Turner et al., 2021, p. 6](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=6&annotation=JGVQNKUC))

“The submissions immediately bring one significant realisation to the forefront: surrogate-assisted optimisation is very effective” ([Turner et al., 2021, p. 6](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=6&annotation=83EU59TG))

“All of the top-20 participants used some form of surrogate-assisted optimisation. This is strongly indicative of the value of using a “surrogate model” and that intelligent modelling/decision” ([Turner et al., 2021, p. 6](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=6&annotation=BQ6YY4QY))

“Top methods on the final leaderboard and examples submissions vs random search (RS): On the left, we show what RS would have done given more function evaluations than allowed in the challenge (128). This performance curve is based on an unbiased estimate from pooling the data of N = 256 RS runs, which gives 256 × 128 = 32,768 function evaluations for each problem.” ([Turner et al., 2021, p. 8](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=8&annotation=YG8X4W3X))

“Many published papers on BO propose using only one surrogate model and one acquisition function, despite some prior research having discussed the benefits of ensembling BO methods [34].” ([Turner et al., 2021, p. 8](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=8&annotation=MHLTIUMH))

“This analysis hints that ensembling may be useful in avoiding failed models where an individual BO algorithm makes little progress.” ([Turner et al., 2021, p. 9](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=9&annotation=C6CYNVF3))

“The competition was divided into a feedback session (which the participants could monitor thorough a practise leaderboard) and a final testing session (the results of which produced the final leaderboard, as seen in Table 2).” ([Turner et al., 2021, p. 10](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=10&annotation=GP96EJK9))

“But we were so excited by the effort put in to meta-learning by these teams that we reran all submissions with full visibility into parameter names. This allowed teams to employ strategies such as making initial guesses using the found optima from problems with the same variable names under the premise that the objective functions are likely similar.” ([Turner et al., 2021, p. 10](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=10&annotation=UCKHGN7K))

“As such, it demonstrated decisively the benefits of Bayesian optimisation over random search. The top submissions showed over 100× sample efficiency gains compared to random search. First, all of the top teams used some form of BO ensemble; sometimes with very simple and easy to productionize strategies such as alternating the surrogate, acquisition function, or potentially entire optimisation algorithms.. Second, the warm start leaderboard demonstrated how warm starting from even loosely related problems often yields large performance gains.” ([Turner et al., 2021, p. 11](zotero://select/library/items/K424VXLR)) ([pdf](zotero://open-pdf/library/items/RCUYWUTK?page=11&annotation=EQKQQLNS))