*title:* ExcelFormer: a neural network surpassing gbdts on tabular data
*authors:* Jintai Chen, Jiahuan Yan, Danny Ziyi Chen, Jian Wu
*year:* 2023
*tags:* 
*status:* #📥
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖

“The investigation in (Grinsztajn et al., 2022) pointed out three inherent characteristics of tabular data that impeded known neural networks from top-tier performances, including irregular patterns of the target function, the negative effects of uninformative features, and the nonrotationally-invariant features. Based on this, we furthermore identify two points that highly promote the capabilities of neural networks on tabular data. (i) An appropriate feature embedding approach. Though it was demonstrated (Rahaman et al., 2019; Grinsztajn et al., 2022) that neural networks are likely to predict overly smooth solutions on tabular data, a deep learning model was also observed to be capable of memorising random labels (Zhang et al., 2021). Since the target function patterns are irregular and spurious correlations between the targets and features exist, an appropriate feature embedding network should well fit the irregular patterns while maintaining generalizability. (ii) A careful feature interaction approach. Since features of tabular data are non-rotationally-variant and a considerable portion of them are uninformative, it harms the generalisation when a model incorporates needless feature interactions.” ([Chen et al., 2023, p. 1](zotero://select/library/items/UKRXZCJB)) ([pdf](zotero://open-pdf/library/items/Q8RGMXPL?page=1&annotation=49WB9QDJ))

“Some previous approaches either designed feature embedding approaches (Gorishniy et al., 2022) to alleviate overly smooth solutions inspired by (Tancik et al., 2020) or employed regularisation (Katzir et al., 2020) and shallow models (Cheng et al., 2016) to promote the model generalisation, while some neural networks were equipped with sophisticated feature interaction approaches (Yan et al., 2023; Chen et al., 2022; Gorishniy et al., 2021) for better selectively feature interactions.” ([Chen et al., 2023, p. 1](zotero://select/library/items/UKRXZCJB)) ([pdf](zotero://open-pdf/library/items/Q8RGMXPL?page=1&annotation=X3Z257LE))

“Apart from model designs, various data representation approaches, such as feature embedding (Gorishniy et al., 2022), discretization of continuous features (Guo et al., 2021; Wang et al., 2020), and rule search approaches (Wang et al., 2021), were proposed against the irregular target patterns (Tancik et al., 2020; Grinsztajn et al., 2022).” ([Chen et al., 2023, p. 2](zotero://select/library/items/UKRXZCJB)) ([pdf](zotero://open-pdf/library/items/Q8RGMXPL?page=2&annotation=65MMYTUW))