

This loss function returns the error rate on this data set $D$. For every example that the classifier misclassifies (i.e. gets wrong) a loss of 1 is suffered, whereas correctly classified samples lead to 0 loss.

Prediction:
$$
y=\operatorname{step}(f(\boldsymbol{x}))=\operatorname{step}\left(\boldsymbol{w}^T \boldsymbol{x}+b\right)
$$
Predict class 1 for $f(x)>0$ else predict class 0
Optimization:
Find $w$ such that
$$
L_0(\boldsymbol{w})=\sum_i \mathbb{I}\left(\operatorname{step}\left(\boldsymbol{w}^T \boldsymbol{x}+b\right) \neq y_i\right)
$$
where $\mathbb{I}$ returns 1 if the argument is true and $\sum$ counts the number of misclassifications

Zero-one loss:
The simplest loss function is the zero-one loss. It literally counts how many mistakes an hypothesis function h makes on the training set. For every single example it suffers a loss of 1 if it is mispredicted, and 0 otherwise. The normalized zero-one loss returns the fraction of misclassified training samples, also often referred to as the training error. The zero-one loss is often used to evaluate classifiers in multiclass/binary classification settings but rarely useful to guide optimization procedures because the function is non-differentiable and non-continuous. Formally, the zero-one loss can be stated has:
$$
\mathcal{L}_{0 / 1}(h)=\frac{1}{n} \sum_{i=1}^n \delta_{h\left(x_1\right) \neq y_i,}, \text { where } \delta_{h\left(x_1\right) \neq y_i}= \begin{cases}1, & \text { if } h\left(x_i\right) \neq y_i \\ 0, & \text { o.w. }\end{cases}
$$
This loss function returns the error rate on this data set $D$. For every example that the classifier misclassifies (i.e. gets wrong) a loss of 1 is suffered, whereas correctly classified samples lead to 0 loss.




Following a common track in research (Gu et al. (2020) and Grammig et al. (2020)), we assess the model’s performance using the pooled, predictive R2 on unseen data. The predictive R2 couples the errors of the model’s estimates over the one of the benchmark:




- Discuss what metrics are reasonable e. g., why is it reasonable to use the accuracy here? Dataset is likely balanced with a 50-50 distribution, metrics like accuracy are fine for this use case.
- Define the metrics.
- Accuracy, ROC-curve, area under the curve. Think about statistical Tests e. g., $\chi^2$-Test
- Introduce concept of a confusion matrix. Are all errors equally problematic?


We optimize for the accuracy

- extension of feature permutation https://arxiv.org/pdf/1801.01489.pdf
- nice description incuding alogirhtm https://christophm.github.io/interpretable-ml-book/feature-importance.html


From [[@raschkaModelEvaluationModel2020]]: zero-one loss and prediction accuracy. In the following article, we will focus on the prediction accuracy, which is defined as the number of all correct predictions divided by the number of examples in the dataset. We compute the prediction accuracy as the number of correct predictions divided by the number of examples $n$. Or in more formal terms, we define the prediction accuracy ACC as
$$
\mathrm{ACC}=1-\mathrm{ERR},
$$
where the prediction error, ERR, is computed as the expected value of the $zero-one$ loss over $n$ examples in a dataset $S$ :
$$
\operatorname{ERR}_S=\frac{1}{n} \sum_{i=1}^n L\left(\hat{y}_i, y_i\right) .
$$
The $zero-one$ loss $L(\cdot)$ is defined as
$$
L\left(\hat{y_i}, y_i\right)= \begin{cases}0 & \text { if } \hat{y}_i=y_i \\ 1 & \text { if } \hat{y}_i \neq y_i\end{cases}
$$