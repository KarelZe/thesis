*title:* SGDR: Stochastic Gradient Descent with Warm Restarts
*authors:* Ilya Loshchilov, Frank Hutter
*year:* 2017
*tags:* #lr #lr-scheduling #cyclic #neural_network #sgd 
*status:* #📦 
*related:* 
- [[@izmailovAveragingWeightsLeads2019]] (contains "critique" about cyclic lr)
- [[@huangSnapshotEnsemblesTrain2017]] (the Hutter paper contains some follow-up experiments; confirms their findings)
- [[@smithCyclicalLearningRates2017]] (very similar idea, proposed cyclic annealing)

## Notes 
- In the paper the authors extend the idea of cyclical learning rate in deep learning by [[@smithCyclicalLearningRates2017]] with warm restarts. That is the old lr $\eta_t$ is increased while the old $\boldsymbol{x}_t$ is used for initialization.
- Cyclic annealing given by:  Within the $i$-th run, we decay the learning rate with a cosine annealing for each batch as follows:
	$$
	\eta_t=\eta_{\min }^i+\frac{1}{2}\left(\eta_{\max }^i-\eta_{\min }^i\right)\left(1+\cos \left(\frac{T_{c u r}}{T_i} \pi\right)\right),
	$$
	where $\eta_{\min}^i$ and $\eta_{\max }^i$ are ranges for the learning rate, and $T_{c u r}$ accounts for how many epochs have been performed since the last restart. 

![[sgd-warm-restarts.png]]

## Annotations

“The commonly used procedure to optimize f is to iteratively adjust xt ∈ IRn (the parameter vector at time step t) using gradient information ∇ft(xt) obtained on a relatively small t-th batch of b datapoints. The Stochastic Gradient Descent (SGD) procedure then becomes an extension of the Gradient Descent (GD) to stochastic optimization of f as follows: xt+1 = xt − ηt∇ft(xt), (1) where ηt is a learning rate. One would like to consider second-order information xt+1 = xt − ηtH−1 t ∇ft(xt), (2) but this is often infeasible since the computation and storage of the inverse Hessian H−1 t is intractable for large n.” ([Loshchilov and Hutter, 2017, p. 1](zotero://select/library/items/A5HB9Q9U)) ([pdf](zotero://open-pdf/library/items/Z4YVX9A3?page=1&annotation=W4UJT5N4))

“In this paper, we propose to periodically simulate warm restarts of SGD, where in each restart the learning rate is initialized to some value and is scheduled to decrease.” ([Loshchilov and Hutter, 2017, p. 2](zotero://select/library/items/A5HB9Q9U)) ([pdf](zotero://open-pdf/library/items/Z4YVX9A3?page=2&annotation=N9LWU5DW))

“Our empirical results suggest that SGD with warm restarts requires 2× to 4× fewer epochs than the currently-used learning rate schedule schemes to achieve comparable or even better results.” ([Loshchilov and Hutter, 2017, p. 2](zotero://select/library/items/A5HB9Q9U)) ([pdf](zotero://open-pdf/library/items/Z4YVX9A3?page=2&annotation=LZ5V9CUQ))

“Warm restarts are usually employed to improve the convergence rate rather than to deal with multimodality: often it is sufficient to approach any local optimum to a given precision and in many cases the problem at hand is unimodal.” ([Loshchilov and Hutter, 2017, p. 3](zotero://select/library/items/A5HB9Q9U)) ([pdf](zotero://open-pdf/library/items/Z4YVX9A3?page=3&annotation=6QMBUT62))

“Since the condition number is an unknown parameter and its value may vary during the search, they proposed two adaptive warm restart techniques (O’Donoghue & Candes, 2012): • The function scheme restarts whenever the objective function increases. • The gradient scheme restarts whenever the angle between the momentum term and the negative gradient is obtuse, i.e, when the momentum seems to be taking us in a bad direction, as measured by the negative gradient at that point. This scheme resembles the one of Powell (1977) for the conjugate gradient method.” ([Loshchilov and Hutter, 2017, p. 3](zotero://select/library/items/A5HB9Q9U)) ([pdf](zotero://open-pdf/library/items/Z4YVX9A3?page=3&annotation=W9GFY4VC))

“Smith (2015; 2016) recently introduced cyclical learning rates for deep learning, his approach is closely-related to our approach in its spirit and formulation but does not focus on restarts.” ([Loshchilov and Hutter, 2017, p. 3](zotero://select/library/items/A5HB9Q9U)) ([pdf](zotero://open-pdf/library/items/Z4YVX9A3?page=3&annotation=WUU844K6))

“The existing restart techniques can also be used for stochastic gradient descent if the stochasticity is taken into account. Since gradients and loss values can vary widely from one batch of the data to another, one should denoise the incoming information: by considering averaged gradients and losses, e.g., once per epoch, the above-mentioned restart techniques can be used again.” ([Loshchilov and Hutter, 2017, p. 4](zotero://select/library/items/A5HB9Q9U)) ([pdf](zotero://open-pdf/library/items/Z4YVX9A3?page=4&annotation=MLDA8FA2))

“n this work, we consider one of the simplest warm restart approaches. We simulate a new warmstarted run / restart of SGD once Ti epochs are performed, where i is the index of the run. Importantly, the restarts are not performed from scratch but emulated by increasing the learning rate ηt while the old value of xt is used as an initial solution. The amount of this increase controls to which extent the previously acquired information (e.g., momentum) is used. Within the i-th run, we decay the learning rate with a cosine annealing for each batch as follows: ηt = ηi min + 1 2 (ηi max − ηi min)(1 + cos( Tcur Ti π)), (5) where ηi min and ηi max are ranges for the learning rate, and Tcur accounts for how many epochs have been performed since the last restart. Since Tcur is updated at each batch iteration t, it can take discredited values such as 0.1, 0.2, etc. Thus, ηt = ηi max when t = 0 and Tcur = 0. Once Tcur = Ti, the cos function will output −1 and thus ηt = ηi min. The decrease of the learning rate is shown in Figure 1 for fixed Ti = 50, Ti = 100 and Ti = 200; note that the logarithmic axis obfuscates the typical shape of the cosine function. In order to improve anytime performance, we suggest an option to start with an initially small Ti and increase it by a factor of Tmult at every restart (see, e.g., Figure 1 for T0 = 1, Tmult = 2 and T0 = 10, Tmult = 2). It might be of great interest to decrease ηi max and ηi min at every new restart. However, for the sake of simplicity, here, we keep ηi max and ηi min the same for every i to reduce the number of hyperparameters involved.” ([Loshchilov and Hutter, 2017, p. 4](zotero://select/library/items/A5HB9Q9U)) ([pdf](zotero://open-pdf/library/items/Z4YVX9A3?page=4&annotation=732QC9DP))

“Since SGDR achieves good performance faster, it may allow us to train larger networks.” ([Loshchilov and Hutter, 2017, p. 7](zotero://select/library/items/A5HB9Q9U)) ([pdf](zotero://open-pdf/library/items/Z4YVX9A3?page=7&annotation=YYR692PX))

“Thus, naively building ensembles from models obtained at last epochs only (i.e., M = 3 snapshots at epochs 148, 149, 150) did not improve the results (i.e., the baseline of M = 1 snapshot at 150) thereby confirming the conclusion of Huang et al. (2016a) that snapshots of SGDR provide a useful diversity of predictions for ensembles.” ([Loshchilov and Hutter, 2017, p. 8](zotero://select/library/items/A5HB9Q9U)) ([pdf](zotero://open-pdf/library/items/Z4YVX9A3?page=8&annotation=DMQDK9SJ))