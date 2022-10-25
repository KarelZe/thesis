
title: The Kaggle book: data analysis and machine learning for competitive data science
authors: Konrad Banachewicz, Luca Massaron
year: 2021
tags :  #supervised-learning #unsupervised-learning #feature-enginering
status : #📦 
related: 
- [[🎄Tree-based Methods/Random Forests/@breimanRandomForests2001]]
- [[@prokhorenkovaCatBoostUnbiasedBoosting2018]]
- [[@owenHyperparameterTuningPython2022]]

# Notes
Time feature processing: Splitting a date into its elements (year, month, day); transforming it into week of the year and weekday; computing differences between dates; computing differences with key events (for instance, holidays). For dates, another common transformation is extracting time elements from a date or a time. Cyclic continuous transformations (based on sine and cosine transformations) are also useful for representing the continuity of time and creating periodic features: 
```python
cycle = 7 
df['weekday_sin'] = np.sin(2 * np.pi * df['col1'].dt.dayofweek / cycle) df['weekday_cos'] = np.cos(2 * np.pi * df['col1'].dt.dayofweek / cycle) 
```

• Numeric feature transformations: Scaling; normalization; logarithmic or exponential transformations; separating the integer and decimal parts; summing, subtracting, multiplying, or dividing two numeric features. Scaling obtained by standardization (the z-score method used in statistics) or by normalization (also called min-max scaling) of numeric features can make sense if you are using algorithms sensitive to the scale of features, such as any neural network. 
• Binning of numeric features: This is used to transform continuous variables into discrete ones by distributing their values into a number of bins. Binning helps remove noise and errors in data and it allows easy modeling of non-linear relationships between the binned features and the target variable when paired with one-hot encoding (see the Scikit-learn implementation, for instance: https://scikit-learn.org/stable/modules/generated/ sklearn.preprocessing.KBinsDiscretizer.html)
• Categorical feature encoding: One-hot encoding; a categorical data processing that merges two or three categorical features together; or the more sophisticated target encoding (more on this in the following sections). • Splitting and aggregating categorical features based on the levels: For instance, in the Titanic competition (https://www.kaggle.com/c/titanic) you can split names and surnames, as well their initials, to create new features. • Polynomial features are created by raising features to an exponent. See, for instance, this Scikit-learn function: https://scikit-learn.org/stable/modules/generated/sklearn. preprocessing.PolynomialFeatures.html.

## Imputation 

It is just like in census surveys: if someone doesn’t tell you their income, it means they are extremely poor or are extremely rich. If required by your learning algorithm, replace the missing values with the mean, median, or mode (it is seldom necessary to use methods that are more sophisticated). Just keep in mind that some models can handle missing values by themselves and do so fairly better than many standard approaches, because the missing-values handling is part of their optimization procedure. The models that can handle missing values by themselves are all gradient boosting models: (p. 212)

## Layman's feature importance
This Notebook tests the role of features in an LSTM-based neural network. First, the model is built and the baseline performance is recorded. Then, one by one, features are shuffled and the model is required to predict again. If the resulting prediction worsens, it suggests that you shuffled an important feature that shouldn’t be touched. Instead, if the prediction performance stays the same or even improves, the shuffled feature is not influential or even detrimental to the model. See Notebook: [LSTM Feature Importance | Kaggle](https://www.kaggle.com/code/cdeotte/lstm-feature-importance/notebook)


## Pseudo Labeling

1. Train your model 
2. Predict on the test set 
3. Establish a confidence measure 
4. Select the test set elements to add 
5. Build a new model with the combined data 
6. Predict using this model and submit

## Neural networks for tabular
Gradient boosting solutions still clearly dominate tabular competitions (as well as real-world projects); however, sometimes neural networks can catch signals that gradient boosting models cannot get, and can be excellent single models or models that shine in an ensemble.

**Tips:**
• Use activations such as GeLU, SeLU, or Mish instead of ReLU; they are quoted in quite a few papers as being more suitable for modeling tabular data and our own experience confirms that they tend to perform better. • Experiment with batch size. • Use augmentation with mixup (discussed in the section on autoencoders). • Use quantile transformation on numeric features and force, as a result, uniform or Gauss ian distributions. • Leverage embedding layers, but also remember that embeddings do not model everything. In fact, they miss interactions between the embedded feature and all the others (so you have to force these interactions into the network with direct feature engineering) (p 232)

## Validation

In the first two scenarios, the solution is the stratified k-fold, where the sampling is done in a controlled way that preserves the distribution you want to preserve. If you need to preserve the distribution of a single class, you can use StratifiedKFold from Scikit-learn, using a stratification variable, usually your target variable but also any other feature whose distribution you need to preserve. The function will produce a set of indexes that will help you to partition your data accordingly.

## Adversial validation

Adversarial validation is a technique allowing you to easily estimate the degree of difference between your training and test data. 
The idea is simple: take your training data, remove the target, assemble your training data together with your test data, and create a new binary classification target where the positive label is assigned to the test data. At this point, run a machine learning classifier and evaluate for the ROC-AUC evaluation metric (we discussed this metric in the previous chapter on Detailing Competition Tasks and Metrics).
If your ROC-AUC is around 0.5, it means that the training and test data are not easily distinguishable and are apparently from the same distribution. ROC-AUC values higher than 0.5 and nearing 1.0 signal that it is easy for the algorithm to figure out what is from the training set and what is from the test set: in such a case, don’t expect to be able to easily generalize to the test set because it clearly comes from a different distribution.

Options for high AUC:
- Suppression 
- Training on cases most similar to the test set 
- Validating by mimicking the test set