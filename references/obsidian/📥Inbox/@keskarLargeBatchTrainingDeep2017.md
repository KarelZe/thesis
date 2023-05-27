*title:* On large-batch training for deep learning: generalisation gap and sharp minima
*authors:* Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang
*year:* 2016
*tags:* #normalisation 
*status:* #📥
*related:*

# Notes 

# Annotations

It has been observed in practise that when using a larger batch there is a degradation in the quality of the model, as measured by its ability to generalise.

We use the term small-batch (SB) method to denote SGD, or one of its variants like ADAM (Kingma & Ba, 2015 ) and ADAGRAD (Duchi et al., 2011), with the proviso that the gradient approximation is based on a small mini-batch. In our experiments, ADAM is used to explore the behaviour of both a small or a large batch method.