*title:* Averaging Weights Leads to Wider Optima and Better Generalization
*authors:* Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
*year:* 2018
*tags:* #sgd #lr #lr-scheduling 
*status:* #📦 
*related:*
- [[@huangSnapshotEnsemblesTrain2017]] (averaging of final model)
- [[@smithCyclicalLearningRates2017]] and [[@loshchilovSGDRStochasticGradient2017]] (cyclic learning rates)
*code:* 
- https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
- https://github.com/timgaripov/swa.

## Notes 
- An **equally-weighted average** of points traversed by SGD with both constant learning rate and cyclic learning rate can improve over several conventional benchmarks.
- The following observation motivated the research: authors run SGD with cyclical and constant learning rate schedules on a pre-trained model. They then use the first, middle and last point of each trajectories to define a 2-dim plane in the weigh space containing all affine combinations of start, mid and end point.
- After making a large step (every $\mod(i,c)$) with cyclical lr, the optimizer spends several epochs for fine-tuning with decreasing lr. SGD with fixed lr always makes steps with large sizes, which is more efficiently than cyclic lr ❗. 
- The name stochastic weight averaging comes from the fact, that the average of SGD weights is calculated. SGD proposals are approximately sampling from the loss surface of DNN, leading to stochastic weights.

![[swa-visualization 1.png]]

## Algorithm
![[stochastic-weight-averaging 1.png]]

## Annotations
“We show that simple averaging of multiple points along the trajectory of SGD, with a cyclical or constant learning rate, leads to better generalization than conventional training.” ([Izmailov et al., 2019, p. 1](zotero://select/library/items/LYLQRDUK)) ([pdf](zotero://open-pdf/library/items/QQHKVF8J?page=1&annotation=A8UJWJNU))

“we show that an equally weighted average of the points traversed by SGD with a cyclical or high constant learning rate, which we refer to as Stochastic Weight Averaging (SWA), has many surprising and promising features for training deep neural networks, leading to a better understanding of the geometry of their loss surfaces.” ([Izmailov et al., 2019, p. 1](zotero://select/library/items/LYLQRDUK)) ([pdf](zotero://open-pdf/library/items/QQHKVF8J?page=1&annotation=3RZSNU8M))

“WA achieves notable improvement for training a broad range of architectures over several consequential benchmarks.” ([Izmailov et al., 2019, p. 2](zotero://select/library/items/LYLQRDUK)) ([pdf](zotero://open-pdf/library/items/QQHKVF8J?page=2&annotation=LNJYMF9I))

“SWA is extremely easy to implement and has virtually no computational overhead compared to the conventional training schemes.” ([Izmailov et al., 2019, p. 2](zotero://select/library/items/LYLQRDUK)) ([pdf](zotero://open-pdf/library/items/QQHKVF8J?page=2&annotation=RDT93C8J))

“the name SWA has two meanings: on the one hand, it is an average of SGD weights. On the other, with a cyclical or constant learning rate, SGD proposals are approximately sampling from the loss surface of the DNN, leading to stochastic weights.” ([Izmailov et al., 2019, p. 3](zotero://select/library/items/LYLQRDUK)) ([pdf](zotero://open-pdf/library/items/QQHKVF8J?page=3&annotation=WCW8NBE3))

“We run SGD with cyclical and constant learning rate schedules starting from a pretrained point for a Preactivation ResNet-164 on CIFAR-100. We then use the first, middle and last point of each of the trajectories to define a 2-dimensional plane in the weight space containing all affine combinations of these points” ([Izmailov et al., 2019, p. 4](zotero://select/library/items/LYLQRDUK)) ([pdf](zotero://open-pdf/library/items/QQHKVF8J?page=4&annotation=B4LWMH7A))

“The main difference between the two approaches is that the individual proposals of SGD with a cyclical learning rate schedule are in general much more accurate than the proposals of a fixed-learning rate SGD. After making a large step, SGD with a cyclical learning rate spends several epochs fine-tuning the resulting point with a decreasing learning rate. SGD with a fixed learning rate on the other hand is always making steps of relatively large sizes, exploring more efficiently than with a cyclical learning rate, but the individual proposals are worse.” ([Izmailov et al., 2019, p. 4](zotero://select/library/items/LYLQRDUK)) ([pdf](zotero://open-pdf/library/items/QQHKVF8J?page=4&annotation=IMCF3D8N))

“if mod(i, c) = 0 then nmodels ← i/c {Number of models} wSWA ← wSWA·nmodels+w nmodels+1 {Update average}” ([Izmailov et al., 2019, p. 5](zotero://select/library/items/LYLQRDUK)) ([pdf](zotero://open-pdf/library/items/QQHKVF8J?page=5&annotation=IJNDVKL5))