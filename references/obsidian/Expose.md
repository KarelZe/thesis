
# Proposal

establish an upper limit



Based on a s

Both data sets are characterized as probablistic classification
Thus . The 

We might draw 




The selection of suitable machine learning models is influenced by: (i) the need for explainability, tabular data, support for both numeric and categorical features, . examples for categorical data and 

partly unlabelled posing additional opportunities

The problem of trade side classifaction can be framed as a (probabilistic) classification task in machine learning. The 

We review the state of the art for classification on tabular data with regard to accuracy and interpretability. Our selection will likely consider *wide ensembles* in the form of gradient boosted trees and *deep, attention-based neural networks*, such as TabNet [[@arikTabNetAttentiveInterpretable2020]] or TabTransformer [[@huangTabTransformerTabularData2020]]. Also, both model classes can naturally be enhanced to profit from partially-unlabelled data.

Thereafter, we give a thorough introduction of the models for the supervised setting. We start with the notion of classical decision trees, as covered by [[@breimanRandomForests2001]] Decision trees are inherent to tree-based boosting approaches as weak learners. Thus, emphasis is put on the selection of features and the splitting process of the predictor space into disjoint regions. We motivate the use of ensemble approaches, such as gradient boosted trees, with the poor variance property of decision trees. The subsequent chapter draws on [[@hastietrevorElementsStatisticalLearning2009]] and [[@friedmanGreedyFunctionApproximation2001]]with a focus on gradient boosting for classification. Therein we introduce necessary enhancements to the boosting procedure to support probabilistic classification and discuss arising stability issues. Further adjustments are necessary for the treatment of categorical variables. Therefore, we draw on the *ordered boosting* by [[@prokhorenkovaCatBoostUnbiasedBoosting2018]], which enhances the classical gradient boosting algorithm.

(Transformer, Attention-based models)

Previous research (e. g., [[@arikTabNetAttentiveInterpretable2020]]) could show that both tree-based and neural-network-based approaches can profit from learning on additional, unlabelled data. Thus we demonstrate how the models from above can be enhanced for the semi-supervised setting. For gradient boosted trees, self-training [[@yarowskyUnsupervisedWordSense1995]]  is used to obtain pseudo labels for unlabeled parts of the data set. The ensemble itself is trained on both true and pseudo labels. For *TabNet* we use unsupervised pretraining of the encoder as propagated in [[@arikTabNetAttentiveInterpretable2020]]. Equally, for the *TabTransformer* we pretrain the transformer layers and column embeddings through *masked language modeling* or *replaced token detection* as popularized in [[@devlinBERTPretrainingDeep2019]] and [[@clarkELECTRAPretrainingText2020]] respectively. 

**Empirical Study**

For our empirical analysis, we introduce the data sets, the generation of true labels and the applied pre-processing.

What is the focus on?

Subset of the CBOE and the ISE data set have been previously studied in [[@grauerOptionTradeClassification2022]]. Thus we align our pre-processing with their work to maintain consistency. Two major deviations will be

additional features, consider subset of features, perform standardization, 

Perform EDA

The data set is split into three disjoint sets for training, validation and testing. Similar to [[@ellisAccuracyTradeClassification2000]] and [[@ronenMachineLearningTrade2022]] we perform a classical train-test split, thereby maintaining the temporal ordering within the data. With statistical tests ~~e. g., adversarial validation~~ we verify the distribution of the and features target is maintained on the test set. Due to the sheer number of model combinations considered and the computational demand of transformers and gradient boosted trees, we expect $k$-fold cross validation to be technically infeasable.

Next, we describe the training of the supervised and semi-supervised models. This includes ... modifications to the algorithms, study of loss and learning curves. The implement. Classical rules are implemented as Scikit-learn classifier.

Hyperparameter tuning is performed using a novel Bayesian search with its roots in Gaussian processes. Compared to brute-force approaches utilised by Gu et al. (2020) and others, unpromising search regions are omitted, requiring fewer trails than bruteforce approaches like grid search. The search space will be loosely oriented to the one of Gu et al. (2020) while aiming for broader coverage, e.g., more sensitive learning rates. An implementation by Akiba et al. (2019) is used for optimizing the R2 val.

We aim for reproducability. As such, we implement sophisticated data set versioning and experiment tracking. 

**Discussion and Conclusion** 

A discussion and a conclusion follow the presentation of the results.


Problemstellung Zunächst ist der Anlass bzw. der Grund für das Forschungsvorhaben zu beschreiben. Dieser kann beispielsweise in einer mangelnden Durchdringung einer Fragestellung, eines Gebietes oder einer Theorie in der Forschung oder einem in der Praxis beobachtbaren Problem liegen. Es ist deutlich zu machen, warum die wissenschaftliche Beschäftigung mit dem Thema überhaupt als relevant erachtet wird. Ausgehend von einer präzise formulierten Forschungsfrage ist das in der Arbeit zu lösende Forschungsproblem zu definieren. Dabei ist außerdem der Stand der Forschung aufzuzeigen. Hierfür ist ggf. der Betreuer zu konsultieren. Welche vergleichbaren Forschungsprojekte wurden bereits durchgeführt und wie sind die Ergebnisse zu beurteilen? Wie ist der Stand in der wissenschaftlichen Literatur und ggf. in der Praxis? 

Ziel der Arbeit Im Rahmen der Zielsetzung ist deutlich zu machen, worin die Leistung der Arbeit bestehen soll. Die Erreichung des selbstgestellten (Haupt-)Ziels der Arbeit stellt die wichtigste Grundlage zur späteren Beurteilung der Arbeit dar. Aus diesem (Haupt-)Ziel können die Subziele (notwendig zu erbringende Teilleistungen zur Erreichung des Hauptziels), die Vorgehensweise, die Gliederung und die Argumentation der Arbeit direkt abgeleitet werden. Die Zielsetzung muss eindeutig beschrieben und überprüfbar sein. Die Ziele sollten in Form konkreter Fragen formuliert werden können. Eine solche Formulierung in Form von Fragen ist jedoch nicht erforderlich. Eine nachträgliche Änderung der Zielsetzung darf nur noch in Absprache mit dem Betreuer erfolgen. Wie wird eine eindeutige Beschreibung der Zielsetzung erreicht? Im Falle einer Thematisierung der Evaluierung eines Untersuchungsgegenstands (eines Produktes oder eines Prozesses) sollte klargestellt werden, was das Ziel dieser Untersuchung ist. Soll bei der Evaluierung von Produkten oder Prozessen in der Praxis die Methode, die zur Evaluierung der Produkte und Prozesse herangezogen wurde (also der Weg), oder die Beurteilung der Produkte oder Prozesse (also das Ergebnis) im Vordergrund stehen? Ein ähnliches Beispiel stellen Arbeiten dar, die Fallbeispiele zum Inhalt haben. Bei Fallbeispielen muss deutlich werden, ob das Beispiel lediglich der Erläuterung dient, einen Machbarkeitsnachweis darstellt oder selbst als wichtiges Ziel der Arbeit dient (z. B. bei einer Softwareentwicklung). 

Begriffserklärung An dieser Stelle muss eine Definition der zentralen Begriffe – das sind insbesondere die Fachbegriffe, die im geplanten Titel der wissenschaftlichen Arbeit stehen – erfolgen. Dabei sind ggf. Quellen zu zitieren, die zur Definition der Begriffe herangezogen wurden. Eventuelle Abweichungen von bereits existierenden Definitionen müssen begründet werden. Vorgehensweise 

Die Vorgehensweise beschreibt die Forschungsmethode, mit der die Zielsetzung der Arbeit erreicht, d. h. die Forschungsfragen beantwortet, werden sollen. Diese wird je nach Art der Arbeit (Literaturarbeit, empirische Arbeit, Softwareentwicklung etc.) stark variieren. Die Vorgehensweise im Sinne einer Forschungsmethode ist von der Vorgehensweise im Sinne eines Aufbaus der Arbeit zu unterscheiden. Die Vorgehensweise muss sich auch in der Gliederung widerspiegeln. Gliederung Die Anzahl der Gliederungspunkte in einem Kapitel bzw. der Grad der Gliederungstiefe sollte mit der Bedeutung der einzelnen Gliederungspunkte korrespondieren. Für das Exposé ist eine Gliederung der 1. und 2. Gliederungsebene ausreichend. Die Gliederung ist zu kommentieren. Das heißt, dass der Inhalt und das Ziel eines jeden Kapitels in Form von Fragen beschrieben werden müssen. Den Kommentaren sind die geschätzten Kapitelumfänge hinzuzufügen. 

Erwartete Ergebnisse Je nach Themenstellung können die erwarteten Ergebnisse auf zwei unterschiedliche Weisen interpretiert werden: In der Regel wird es sich hierbei um Beschreibungen konkreter Produkte der wissenschaftlichen Arbeit handeln. Hierzu zählen z. B. Kriterienkataloge, Evaluierungsberichte, Softwareprogramme, Umfrageergebnisse, Modelle, Methoden-/Vorgehensbeschreibungen, Bibliographien etc. Bei eher empirisch orientierten Arbeiten umfassen die erwarteten Ergebnisse bereits erste Hypothesen über die voraussichtlichen Ergebnisse der Untersuchung. Vermutungen bzw. Thesen lassen sich aber auch für nicht empirische Arbeiten aufstellen (z. B., dass eine in der Arbeit untersuchte Theorie der Organisationslehre auch auf die Softwareentwicklung übertragbar ist).

