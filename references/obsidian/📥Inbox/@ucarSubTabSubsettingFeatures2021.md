*title:* SubTab: subsetting features of tabular data for self-supervised representation learning
*authors:* Talip Ucar, Ehsan Hajiramezanali, Lindsay Edwards
*year:* 2021
*tags:* #self-supervised #semi-supervised-learning
*status:* #📥
*related:*
- [[@yoonVIMEExtendingSuccess2020]] (authors reference this works for reasons why tabular representations are different and common techniques from nlp and cv don't apply)
- 
*code:*
*review:*

## Notes 📍

## Annotations 📖

“In recent years, the self-supervised learning has successfully been used to learn meaningful representations of the data in natural language processing [34, 41, 11, 28, 10, 21, 9]. A similar success has been achieved in image and audio domains [7, 15, 37, 5, 17, 13, 8]. This progress is mainly enabled by taking advantage of spatial, semantic, or temporal structure in the data through data augmentation [7], pretext task generation [11] and using inductive biases through architectural choices (e.g. CNN for images). However, these methods can be less effective in the lack of such structures and biases in the tabular data commonly used in many fields such as healthcare, advertisement, finance, and law. And some augmentation methods such as cropping, rotation, colour transformation etc. are domain specific, and not suitable for tabular setting. The difficulty in designing similarly effective methods tailored for tabular data is one of the reasons why self-supervised learning is under-studied in this domain [46].” ([Ucar et al., 2021, p. 1](zotero://select/library/items/F9DTPDH5)) ([pdf](zotero://open-pdf/library/items/MLWKHKKR?page=1&annotation=KZDPXQEV))

“he most common approach in tabular data is to corrupt data through adding noise [43]. An autoencoder maps corrupted examples of data to a latent space, from which it maps back to uncorrupted data. Through this process, it learns a representation robust to the noise in the input. This approach may not be as effective since it treats all features equally as if features are equally informative. However, perturbing uninformative features may not result in the intended goal of the corruption. A recent work takes advantage of self-supervised learning in tabular data setting by introducing a pretext task [46], in which a de-noising autoencoder with a classifier attached to representation layer is trained on  corrupted data. The classifier’s task is to predict the location of corrupted features. However, this framework still relies on noisy data in the input. Additionally, training a classifier on an imbalanced binary mask for a high-dimensional data may not be ideal to learn meaningful representations.” ([Ucar et al., 2021, p. 2](zotero://select/library/items/F9DTPDH5)) ([pdf](zotero://open-pdf/library/items/MLWKHKKR?page=2&annotation=3753MUZQ))

“In this work, we turn the problem of learning representation from a single-view of the data into the one learnt from its multiple views by dividing the features into subsets, akin to cropping in image domain or feature bagging in ensemble learning, to generate different views of the data. Each subset can be considered a different view. We show that reconstructing data from the subset of its features forces the encoder to learn better representation than the ones learnt through the existing methods such as adding noise.” ([Ucar et al., 2021, p. 2](zotero://select/library/items/F9DTPDH5)) ([pdf](zotero://open-pdf/library/items/MLWKHKKR?page=2&annotation=YC5S6D4S))