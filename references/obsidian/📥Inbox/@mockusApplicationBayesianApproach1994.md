*title:* Application of Bayesian approach to numerical methods of global and stochastic optimisation
*authors:* Jonas Mockus
*year:* 1993
*tags:* #bayesian-search #grid-search 
*status:* #📦 
*related:*
- [[@akibaOptunaNextgenerationHyperparameter2019]]
- [[@turnerBayesianOptimizationSuperior2021]]

## Notes 📍

- **When is grid search or a non-uniform grid search preferable?** On a compact feasible set with unknown Lifschitzian constant a uniform grid is best for a minmax problem. Grid search comes with exponential complexity. Thus more dimensions increase the exponent. If the Lifschitzian constant is known, a non-uniform grid technique is preferable, but hardly improves complexity. With continous functions the minmax approach cannot be applied.
- Given that the function is continous and the prior distribution of the objective function $f(x)$ is chosen correctly, the Bayesian method will converge to a global minimum. Bayesian methods are thus at least as good as classical ones for a family of continous functions.
- For any fixed number of observations Bayesian methods minimise the expected deviation from the global minimum.

## Annotations 📖

“Let us to consider for example the global optimisation of the family of Lifschitzian functions with unknown constant. Then the best method in the minimax sense is a uniform grid on a compact feasible set, see Sukharev (1975). It means that this global optimisation algorithm is of exponential complexity. The number of observations is increasing as exponent of the dimension of problem. Here "observation" means an evaluation of objective function f(x) at some fixed point X. If the Lifschitzian constant is known, then some nonuniform grid technique is preferable, see Evtushenko (1985). However, even here the complexity of algorithm apparently remains exponential, perhaps with a better constant. In global optimisation of continuous functions on a compact set we cannot apply a minimax approach at all. It is well known that maximum does not exist on a set of all continuous functions. It means that for any fixed continuous function and a fixed method of search there exists some other continuous function with a larger deviation from a global minimum. So the strong condition of uniform convergence does not apply here.” ([Mockus, 1994, p. 348](zotero://select/library/items/EEI28G7U)) ([pdf](zotero://open-pdf/library/items/H6SK9Q2L?page=2&annotation=YYCPNZKX))

“We assume that a Bayesian method should converge to a global minimum of any continuous function, if an a priori distribution is chosen correctly. It means that the asymptotic of Bayesian method is at least as good as that of any classical one for a family of continuous functions. In fact it is even better. The asymptotic density of observations of Bayesian methods is considerably higher near global minimum” ([Mockus, 1994, p. 350](zotero://select/library/items/EEI28G7U)) ([pdf](zotero://open-pdf/library/items/H6SK9Q2L?page=4&annotation=RMK865UQ))

“However, the main advantage of Bayesian methods is that they minimise an expected deviation from the global minimum for any fixed number of observations” ([Mockus, 1994, p. 350](zotero://select/library/items/EEI28G7U)) ([pdf](zotero://open-pdf/library/items/H6SK9Q2L?page=4&annotation=S9PFRVY2))