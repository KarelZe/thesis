*title:* Semi-Supervised Learning Literature Survey
*authors:* Xiaojin Zhu
*year:* 
*tags:* #semi-supervised #self-learning #self-training
*status:* #📥
*related:*

# Notes 

- semi-supervised learning falls between supervised and unsupervised learning

## Semi-supervised learning
“It is a special form of classification. Traditional classifiers use only labeled data (feature / label pairs) to train. Labeled instances however are often difficult, expensive, or time consuming to obtain, as they require the efforts of experienced human annotators. Meanwhile unlabeled data may be relatively easy to collect, but there has been few ways to use them. Semi-supervised learning addresses this problem by using large amount of unlabeled data, together with the labeled data, to build better classifiers.” (Zhu, p. 4)

## Self-training
“Self-training is a commonly used technique for semi-supervised learning. In selftraining a classifier is first trained with the small amount of labeled data. The classifier is then used to classify the unlabeled data. Typically the most confident unlabeled points, together with their predicted labels, are added to the training set. The classifier is re-trained and the procedure repeated. Note the classifier uses its own predictions to teach itself. The procedure is also called self-teaching or bootstrapping (not to be confused with the statistical procedure with the same name). The generative model and EM approach of section 2 can be viewed as a special case of ‘soft’ self-training.” (Zhu, p. 11)

# Annotations