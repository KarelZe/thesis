#lr-warmup #lr-scheduling 

- introduce notion of effective batch size (batch size when training is split across multiple gpus; see [[🧠Deep Learning Methods/Transformer/@popelTrainingTipsTransformer2018]] p. 46)
- report or store training times?
- In case of diverged training, try gradient clipping and/or more warmup steps. (found in [[🧠Deep Learning Methods/Transformer/@popelTrainingTipsTransformer2018]])
- use a higher learning rate e. g. lr=0.2
- results are the same when trained on multiple gpus, if batch size across all gpus remains the same. [[@poppeSensitivityVPINChoice2016]] confirmed this empirically.
- One might has to adjust the lr when scaling across multiple gpus [[@poppeSensitivityVPINChoice2016]] contains a nice discussion.
- Use weight decay of 0.1 for a small amount of regularization [[@loshchilovDecoupledWeightDecay2019]].


## Notes from Huang et al paper
See [[@huangTabTransformerTabularData2020]] (p. 12)
- AdamW optimizer
- const learning rate
- early stopping after 15 epochs
- hidden (embedding) dim, no of layers, no attention heads. MLP sizes are 4x and 2x the input
- 2,4,8 attention heads
- hid dim 32, 64, 128, 256
- no layers 1, 2, 3, 6, 12
- mlp selu, batchnorm, 2 hidden layers, capacity 8* l with l = d/8, second layer m*l with m in (1,3)

## Notes from uvadlc
(https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)

- One commonly used technique for training a Transformer is learning rate warm-up. This means that we gradually increase the learning rate from 0 on to our originally specified learning rate in the first few iterations. Thus, we slowly start learning instead of taking very large steps from the beginning. In fact, training a deep Transformer without learning rate warm-up can make the model diverge and achieve a much worse performance on training and testing.
- Clearly, the warm-up is a crucial hyperparameter in the Transformer architecture. Why is it so important? There are currently two common explanations.
- For instance, the original Transformer paper used an exponential decay scheduler with a warm-up. However, the currently most popular scheduler is the cosine warm-up scheduler, which combines warm-up with a cosine-shaped learning rate decay. We can implement it below, and visualize the learning rate factor over epochs.
- Some intuition on learning rate warm-up can be found in [[@liuVarianceAdaptiveLearning2021]] and https://stackoverflow.com/a/55942518/5755604
- 




For training of transformers see [[@popelTrainingTipsTransformer2018]]
Intuition behind sample weights: https://m.youtube.com/watch?v=68ABAU_V8qI
Smart batching for transformers: https://mccormickml.com/2020/07/29/smart-batching-tutorial/#why-we-pad
For training the transformer see: https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it


Ilya Sutskever. A Brief Overview of Deep Learning. http://yyue.blogspot.com/2015/01/abrief-overview-of-deep-learning.html, January 2015.

Research broader theory behind sample weighting. Research if there is a broader theory / concept to decay e. g. exponential smooting or weighted regression etc.
For random shuffling with stochastic gradient descent see [[@lecunEfficientBackProp2012]]

- Use weighting scheme for samples. See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- tips for training transformers https://huggingface.co/docs/transformers/v4.18.0/en/performance
- learning rate warmup https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
- optimizer schedulers for transformers: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules


![[visualization_of_bleu_over_time.png]]

![[bleu_no_of_gpus.png]]

Visualize model parameters:
![[viz-model-params.png]]
(from https://arxiv.org/pdf/2005.14165.pdf)

