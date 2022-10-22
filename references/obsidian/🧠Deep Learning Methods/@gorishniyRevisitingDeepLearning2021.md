
title: Revisiting Deep Learning Models for Tabular Data
authors: Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko
year: 2020


## Tabular format
Tabular format is characterized by: “In these problems, data points are represented as vectors of heterogeneous features, which is typical for industrial applications and ML competitions, where neural networks have a strong non-deep competitor in the form of GBDT (Chen and Guestrin, 2016; Ke et al., 2017; Prokhorenkova et al., 2018).” (Gorishniy et al., 2021, p. 1)

## Problems in research for tabular data
"The “shallow” state-of-the-art for problems with tabular data is currently ensembles of decision trees, such as GBDT (Gradient Boosting Decision Tree) (Friedman, 2001), which are typically the top-choice in various ML competitions."
“Along with potentially higher performance, using deep learning for tabular data is appealing as it would allow constructing multi-modal pipelines for problems, where only one part of the input is tabular, and other parts include images, audio and other DL-friendly data. Such pipelines can then be trained end-to-end by gradient optimization for all modalities.” (Gorishniy et al., 2021, p. 1)

“Additionally, despite the large number of novel architectures, the field still lacks simple and reliable solutions that allow achieving competitive performance with moderate effort and provide stable performance across many tasks.” (Gorishniy et al., 2021, p. 1)

## Common architectures

“Attention-based models. Due to the ubiquitous success of attention-based architectures for different domains (Dosovitskiy et al., 2021; Vaswani et al., 2017), several authors propose to employ attentionlike modules for tabular DL as well (Arik and Pfister, 2020; Huang et al., 2020; Song et al., 2019)” (Gorishniy et al., 2021, p. 2)


“our simple adaptation of the Transformer architecture (Vaswani et al., 2017) for tabular data.” (Gorishniy et al., 2021, p. 2)

## FT-Transformer

“Second, FT-Transformer demonstrates the best performance on most tasks and becomes a new powerful solution for the field. Interestingly, FT-Transformer turns out to be a more universal architecture for tabular data:” (Gorishniy et al., 2021, p. 2)

“Finally, we compare the best DL models to GBDT and conclude that there is still no universally superior solution.” (Gorishniy et al., 2021, p. 2)





## Notation for supervised learning problem

“Notation. In this work, we consider supervised learning problems. D={(xi, yi)}in=1 denotes a dataset, where xi=(x(num) i , x(cat) i ) ∈ X represents numerical x(num) ij and categorical x(cat) ij features of an object and yi ∈ Y denotes the corresponding object label. The total number of features is denoted as k. The dataset is split into three disjoint subsets: D = Dtrain ∪ Dval ∪ Dtest, where Dtrain is used for training, Dval is used for early stopping and hyperparameter tuning, and Dtest is used for the final evaluation. We consider three types of tasks: binary classification Y = {0, 1}, multiclass classification Y = {1, . . . , C} and regression Y = R.” (Gorishniy et al., 2021, p. 3)