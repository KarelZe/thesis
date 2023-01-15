#lr-warmup #lr-scheduling 

- practical tips for deep learning: http://yyue.blogspot.com/2015/01/abrief-overview-of-deep-learning.html

- training of the transformer has been found non-trivial[[@liuUnderstandingDifficultyTraining2020]]
- introduce notion of effective batch size (batch size when training is split across multiple gpus; see [[🧠Deep Learning Methods/Transformer/@popelTrainingTipsTransformer2018]] p. 46)
- report or store training times?
- In case of diverged training, try gradient clipping and/or more warmup steps. (found in [[🧠Deep Learning Methods/Transformer/@popelTrainingTipsTransformer2018]])
- use a higher learning rate e. g. lr=0.2
- results are the same when trained on multiple gpus, if batch size across all gpus remains the same. [[@poppeSensitivityVPINChoice2016]] confirmed this empirically.
- One might has to adjust the lr when scaling across multiple gpus [[@poppeSensitivityVPINChoice2016]] contains a nice discussion.
- Use weight decay of 0.1 for a small amount of regularization [[@loshchilovDecoupledWeightDecay2019]].
- on the compute cost of transformers [[@ivanovDataMovementAll2021]]

- On activation function see [[@shazeerGLUVariantsImprove2020]]


- log gradients and loss using `wandb.watch` as shown here https://www.youtube.com/watch?v=k6p-gqxJfP4 with `wandb.log({"epoch":epoch, "loss":loss}, step)` (nested in `if ((batch_ct +1) % 25) == 0:`) and `wandb.watch(model, criterion, log="all", log_freq=10)`
- watch out for exploding and vanishing gradients
- distillation, learning rate warmup, learning rate decay (not used but could improve training times and maybe accuracy) ([[@gorishniyRevisitingDeepLearning2021]])
- Use of Post-Norm (Hello [[🤖TabTransformer]]) has been deemed outdated in Transformers due to a more fragile training process (see [[@gorishniyRevisitingDeepLearning2021]]). May swap (?).
- Tips for training deep neural networks on categorical data: https://www.youtube.com/watch?v=E8C_obO1HfY 
- Mind the double descent effect https://openai.com/blog/deep-double-descent/
- https://blog.ml6.eu/how-a-pretrained-tabtransformer-performs-in-the-real-world-eccb12362950
- https://www.borealisai.com/research-blogs/tutorial-14-transformers-i-introduction/
- Might use additional tips from here: [[@liuRoBERTaRobustlyOptimized2019]])
- https://paperswithcode.com/paper/revisiting-deep-learning-models-for-tabular/review/
- [[@liuUnderstandingDifficultyTraining2020]]

## Tipps from reddit 🤖
https://www.reddit.com/r/MachineLearning/comments/z088fo/r_tips_on_training_transformers/
1.  Bigger architectures learn better and train faster
2.  Layer norms are very important
3.  Apply high learning rates to top layers and smaller rates to lower layers
4.  The batch size should be as high as possible -> write script -> keep gpu busy
5. Transformers are data hungry (must be stated in the [[@vaswaniAttentionAllYou2017]] paper)

## Notes from Borealis AI
- detailed tips: https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/

## Notes from Neptune AI
(https://neptune.ai/blog/tips-to-train-nlp-models)
- use layer-wise learning rate
- reinitialize layers
- use pre-training :-)
- code provided here: https://github.com/flowerpot-ai/stabilizer

## Notes from less wrong on training Transformers
(https://www.lesswrong.com/posts/b3CQrAo2nufqzwNHF/how-to-train-your-transformer)
-   Read the relevant literature and take note of all tricks
-   Use Colab for free GPU time.
-   Rezero or Dynamic Linear Combinations for scaling depth.
-   Shuffle data and create train, validation, test sets from the beginning.
-   Shuffle in memory if samples are otherwise correlated.
-   Train a simple fully connected network as baseline and sanity check.
-   Establish scaling laws to find bottlenecks and aim at the ideal model size.
-   Use gradient accumulation to fit larger models on a small GPU.
-   Lower the learning rate when the model stagnates, but don't start too low. (Better yet, use a cyclic learning rate schedule.)


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

