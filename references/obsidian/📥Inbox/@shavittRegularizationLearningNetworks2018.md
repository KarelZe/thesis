*title:* Regularization Learning Networks: Deep Learning for Tabular Datasets
*authors:* Ira Shavitt, Eran Segal
*year:* 2017
*tags:* 
*status:* #📦 #tabular-data #regularization #gbm #neural_network 
*related:*
- [[@heBagTricksImage2018]]

## Notes 📍
- Authors propose a new loss function called the counterfactual loss that applies a different regularization coefficient to each weight. Also they introduce a new network architectures called regularization learning networks. Regularization learning networks use the  counterfold loss to tune its regularization hyperaparameters.
- Tabular data is esspecially challenging as a single change in a feature can have a large impact on the prediction. With image data it's different, here a change in pixels has minor effects. For some features in tabular data, the feature importance might also be minor, however. 🌰
- Thus, authors hypothesize that the large variability in the relative importance of different input features is the reason for the poor performance of DNNs.
- **Results:**
	- In their experiments GBMs achieve the best results, deep neural networks are worst, but improve with their proposed extensions. An ensemble of models can improve on the performance, esspecially for models with a high variance. 
	- While ensembling of GBMs and RLN or LMs / GBMs with GBMs improves the results in general, it deep RLN ensembles achieve the worst results.
	- Their RLNS can improve the performacne of classical deep neural networks achieving an increase explained variance by a factor of 2.75 +/- 0.05, which is yet similar to the GBMs.
- Nice visualization of weight traversal between layers:
![[traversal-of-axis.png]]

## Annotations 📖

“e propose that applying a different regularization coefficient to each weight might boost the performance of DNNs by allowing them to make more use of the more relevant inputs. However, this will lead to an intractable number of hyperparameters. Here, we introduce Regularization Learning Networks (RLNs), which overcome this challenge by introducing an efficient hyperparameter tuning scheme which minimizes a new Counterfactual Loss.” ([Shavitt and Segal, 2018, p. 1](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=1&annotation=P65QILVP))

“n contrast, the relative contribution of the input features in the electronic health records example can vary greatly: Changing a single input such as the age of the patient can profoundly impact the life expectancy of the patient, while changes in other input features, such as the time that passed since the last test was taken, may have smaller effects” ([Shavitt and Segal, 2018, p. 1](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=1&annotation=BKNL4T8F))

“We hypothesized that this potentially large variability in the relative importance of different input features may partly explain the lower performance of DNNs on such tabular datasets [11]. One way to overcome this limitation could be to assign a different regularization coefficient to every weight, which might allow the network to accommodate the non-distributed representation and the variability in relative importance found in tabular datasets” ([Shavitt and Segal, 2018, p. 2](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=2&annotation=5VIAHHAP))

“Here, we present a new hyperparameter tuning technique, in which we optimize the regularization coefficients using a newly introduced loss function, which we term the Counterfactual Loss, orLCF . We term the networks that apply this technique Regularization Learning Networks (RLNs).” ([Shavitt and Segal, 2018, p. 2](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=2&annotation=CMTVWPYT))

“When running each model separately, GBTs achieve the best results on all of the tested traits, but it is only significant on 3 of them (Figure 2). DNNs achieve the worst results, with 15% ± 1% less explained variance than GBTs on average.” ([Shavitt and Segal, 2018, p. 5](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=5&annotation=XZFERL5V))

“Constructing an ensemble of models is a powerful technique for improving performance, especially for models which have high variance, like neural networks in our task.” ([Shavitt and Segal, 2018, p. 5](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=5&annotation=QP9GARFH))

“Despite the improvement, DNN ensembles still achieve the worst results on all of the traits except for Gender and achieve results 9% ± 1% lower than GBT ensembles (Figure 4).” ([Shavitt and Segal, 2018, p. 6](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=6&annotation=UM3KUSRP))

“Indeed, as shown in Figure 5, the best performance is obtained with an ensemble of RLN and GBT, which achieves the best results on all traits except Gender, and outperforms all other ensembles significantly on Age, BMI, and HDL cholesterol (Table 1)” ([Shavitt and Segal, 2018, p. 6](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=6&annotation=LQVXNYE8))

“Evaluating feature importance is difficult, especially in domains in which little is known such as the gut microbiome. One possibility is to examine the information it supplies” ([Shavitt and Segal, 2018, p. 8](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=8&annotation=XIXZPM2I))

“Another possibility is to evaluate its consistency across different instantiations of the model. We expect that a good feature importance technique will give similar importance distributions regardless of instantiation” ([Shavitt and Segal, 2018, p. 9](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=9&annotation=LSSA3XMR))

“We hypothesize that modular regularization could boost the performance of DNNs on such tabular datasets. We introduce the Counterfactual Loss, LCF , and Regularization Learning Networks (RLNs) which use the Counterfactual Loss to tune its regularization hyperparameters efficiently during learning together with the learning of the weights of the network.” ([Shavitt and Segal, 2018, p. 9](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=9&annotation=JHE6K2Z2))

“We test our method on the task of predicting human traits from covariates and microbiome data and show that RLNs significantly and substantially improve the performance over classical DNNs, achieving an increased explained variance by a factor of 2.75 ± 0.05 and comparable results with GBTs.” ([Shavitt and Segal, 2018, p. 9](zotero://select/library/items/TUGGUIBC)) ([pdf](zotero://open-pdf/library/items/M7YM34G9?page=9&annotation=S28X7A65))