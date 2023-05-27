
title: Deep Sparse Rectifier Neural Networks
authors: Xavier Glorot, Antoine Bordes, Yoshua Bengio
year: 2011
*tags:* #neural_network #deep-learning #activations
*status:* #📥
*related:*

## Notes
- discuss advantages of ReLU
- The rectifier function rectifier $(x)=\max (0, x)$ is onesided and therefore does not enforce a sign symmetry or antisymmetry : instead, the response to the opposite of an excitatory input pattern is 0 (no response).

## Advantages
- can be used to obtain sparse representations. With uniform intialization of weights around 50 % of hidden units continous output values are real zeros.
- gradients flow well on the active path of neurons (there is no gradient vanishing effect due to activation non-linearities of sigmoid or tanh units)
- mathematical investigation is easier
- computations are cheap as now exponential function is required (p. 318)

## Disadvantages:
- hard saturation at 0 may hurt optimisation by blocking gradient back-propagation
- Another problem could arise due to the unbounded behaviour of the activations; one may thus want to use a regularizer to
prevent potential numerical problems.
- Finally, rectifier networks are subject to illconditioning of the parametrization. Biases and weights can be scaled in different (and consistent) ways while preserving the same overall network function.