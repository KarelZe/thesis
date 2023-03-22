- Discuss what metrics are reasonable e. g., why is it reasonable to use the accuracy here? Dataset is likely balanced with a 50-50 distribution, metrics like accuracy are fine for this use case.
- Define the metrics.
- Accuracy, ROC-curve, area under the curve. Think about statistical Tests e. g., $\chi^2$-Test
- Introduce concept of a confusion matrix. Are all errors equally problematic?


We optimize for the accuracy

- extension of feature permutation https://arxiv.org/pdf/1801.01489.pdf
- nice description incuding alogirhtm https://christophm.github.io/interpretable-ml-book/feature-importance.html


From [[@raschkaModelEvaluationModel2020]]: 0-1 loss and prediction accuracy. In the following article, we will focus on the prediction accuracy, which is defined as the number of all correct predictions divided by the number of examples in the dataset. We compute the prediction accuracy as the number of correct predictions divided by the number of examples $n$. Or in more formal terms, we define the prediction accuracy ACC as
$$
\mathrm{ACC}=1-\mathrm{ERR},
$$
where the prediction error, ERR, is computed as the expected value of the $0-1$ loss over $n$ examples in a dataset $S$ :
$$
\operatorname{ERR}_S=\frac{1}{n} \sum_{i=1}^n L\left(\hat{y}_i, y_i\right) .
$$
The $0-1$ loss $L(\cdot)$ is defined as
$$
L\left(\hat{y_i}, y_i\right)= \begin{cases}0 & \text { if } \hat{y}_i=y_i \\ 1 & \text { if } \hat{y}_i \neq y_i\end{cases}
$$