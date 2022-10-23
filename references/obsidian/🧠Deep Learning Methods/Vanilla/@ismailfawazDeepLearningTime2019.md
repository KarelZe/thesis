
title: Deep learning for time series classification: a review
authors: Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar, Pierre-Alain Muller
year: 2019
*tags:* #neural_network #deep-learning 
*status:* #📥
*related:*

## Notes
- Authors study the usage of DNN for time series classification
- Timeseries classification is a difficult task.
- Traditional approach is to use a nearest neighbour classifier coupled with a distance function.
- DNNs outperform the traditional approaches like COTE, HIVE-COTE...

They consider:
- MLPs
- CNNs (several variants)
- Echo state networks (a variant of the RNN)

**CNNs:**
- A convolution can be seen as applying and sliding a filter over the time series. the filters have only one dimension (time). The filter is a generic non-linear transformation of the timeseries.
- If we apply a filter of length 3 with a univariate time series, by setting the filter values to be qual to $\left[0.33, 0.33, 0.33\right]$ The convolution will result in applying a moving average with a sliding window of length 3.
-  A general form of applying the convolution for a centered time stamp $t$ is given in the following equation:
$$
C_{t}=f\left(\omega * X_{t-l / 2: t+l / 2}+b\right) \mid \forall t \in[1, T]
$$
where $C$ denotes the result of a convolution (dot product)applied on a univariate time series $X$ of length $T$ with a filter $\omega$ of length $l$, a bias parameter $b$ and a final non-linear function $f$ such as the Rectified Linear Unit (ReLU).
- The same convolution / is used for all time stamps $t \in[1, T]$. This referred to as weight-sharing.
- The filters should be learned automatically since they depend highly on the targeted dataset. To learn the convolution is followed by a discriminative classifier, which is usually preceded (local / global) pooling operation.
- One can use max / average pooling. If a global pooling operation the time series will be aggregated over the whole dimension resulting in a single value. Which similar to applying a local pooling with a sliding window's length equal to the length of the input time series. Global aggregation helps to reduce the no. of layers.
- Sometimes batch normalization operation is performed over each channel to help the network converge quickly.
- The final layer (typically a non-linear FC layer) takes a representation of the input time series (the result of the convolutions) and give a probability distribution over the class variables.