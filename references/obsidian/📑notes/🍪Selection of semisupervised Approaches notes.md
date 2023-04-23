
An exception are ([[@levinTransferLearningDeep2022]]7), who find no improvements from pre-training the FT-Transformer.

Interesting resources on pre-training:
- https://arxiv.org/pdf/2109.07437.pdf
- https://phontron.com/class/anlp2022/assets/slides/anlp-07-pretraining.pdf
- 



[[@dalche-bucSemisupervisedMarginBoost2001]]
*SSMBoost* requires semi-supervised base learners, which 
ASSEMBLE 

Boosting, as an ensemble learning framework, is one of the most powerful classification algorithms in supervised learning. Based on the gradient descent view of boosting [Mason et al., 2000], many semi-supervised boosting methods have been proposed, such as SMarginBoost [d’Alch ́ e-Buc et al., 2002], ASSEMBLE [Bennett et al., 2002], RegBoost [Chen and Wang, 2007; Chen and Wang, 2011], SemiBoost [Mallapragada et al., 2009], SERBoost [Saffari et al., 2008] and information theoretic regularization based boosting [Zheng et al., 2009], where a margin loss function is minimized over both labeled and unlabeled data by the functional gradient descent method. The effectiveness of these methods can be ascribed to their tendency to produce large margin classifiers with a small classification error. However, these algorithms were not designed to directly maximize the margin (although some of them have the effects of margin enforcing), and the objective functions are not related to the margin in the sense that one can minimize these loss functions while simultaneously achieving a bad margin [Rudin et al., 2004]. Therefore, a natural goal is to construct classifiers that directly optimize margins as measured on both labeled and unlabeled data.

See [[@vanengelenSurveySemisupervisedLearning2020]] for different approaches.
For the existing semi-supervised boosting methods [Bennett et al., 2002; Chen and Wang, 2007; d’Alche-Buc ´ et al., 2002; Mallapragada et al., 2009; Saffari et al., 2008; Zheng et al., 2009],A Direct Boosting Approach for Semi-Supervised Classification ([[@zhaiDirectBoostingApproach]]).

[[@mallapragadaSemiBoostBoostingSemiSupervised2009]] [[@bennettExploitingUnlabeledData2002]][[@dalche-bucSemisupervisedMarginBoost2001]]

![[Pasted image 20230423083202.png]]
