Our goal is to extend Transformers and gradient-boosted trees for the semi-supervised setting, so that both methods can utilize the abundant, unlabelled data.
In our quest to perform a fair comparison between the supervised and semi-supervised variants, we aim to make the extensions minimally intrusive.

**Semis-supervised gradient-boosting**

For the existing semi-supervised boosting methods [Bennett et al., 2002; Chen and Wang, 2007; d’Alche-Buc ´ et al., 2002; Mallapragada et al., 2009; Saffari et al., 2008; Zheng et al., 2009],A Direct Boosting Approach for Semi-Supervised Classification ([[@zhaiDirectBoostingApproach]]).

Both approaches, however, require changes to the boosting procedure or the base learner. An alternative is to combine gls-gbm with self-training. Self-training is a wrapper algorithm around a supervised classifier, that incorporates its most-confident predictions of unlabelled instances into the training procedure ([[@yarowskyUnsupervisedWordSense1995]] 190). Being a model-agnostic wrapper, it does not require changes to the classifier and maximum comparability with the standard gradient-boosting is given. Also, its wide-spread adaption in literature makes it a promising choice for the use in semi-supervised trade classification.

