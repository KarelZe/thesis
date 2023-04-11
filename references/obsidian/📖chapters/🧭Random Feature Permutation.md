
**Points to cover:**
- What is the goal? Interpretability? Explainability? Local? Global?
- What do we want to find out?
- Why does it make sense to use random feature permutation?
- Algorithmic description
- What are the problems of random feature permutation? Which ones are addressed? Which ones are not?


----

As the 

One such 

This leaves us 


Consequently, we derive feature importances using random feature permutation, which is model-agnostic and computationally efficient.

Random feature permutation only yields global 


Our goal is to understand the contribution of a feature to the correct prediction, rather than attributing the prediction to specific features. 

Our setting is unique. 

As we defined derived features, such as the proximity to the quotes, features can not assumed to be independent. 

Substitution effects

However, feature independence is 

The features used by the model may also be different. The quote

Also machine learning classifiers have simultaneous access to 
Also feature importances, may be diluted over several features, known as as features may encode the same information redundantly

Also, features. Classical 

As such, we adapt random feature importance to our setting 

Random feature permutation was originally proposed in 


Random feature permutation is model-agnostic and can be used with different error estimates. For consistency the change in accuracy is used in our work.

The change can be estimated, as the absolute or relative difference.

Random feature permutation as proposed by b

Permuting features also

The complete algorithm is given in:

Two major drawbacks of random feature permutation, are 

One major drawback of random feature permutation is, that it doesn't help with local interpretability. Correlations are artificially broken

unrealistic permutations

---


**Notes:**
[[🧭Random Feature Permutation notes]]

 