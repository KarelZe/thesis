*title:* Unsupervised Word Sense Disambiguation Rivaling Supervised Methods
*authors:* David Yarowsky
*year:* 1994
*tags:* #semi-supervised #classifier #supervised-learning #probabilistic-classification 
*status:* #📦 
*related:*
- [[@zhuSemiSupervisedLearningLiterature]]

## Notes📍
- Works with any supervised, probabilistic classification algorithm.
- Some data is initially labelled and the remainder (typically 85-98 %) remains unlabeled.
- Steps:
	1. Train a supversied classifier on labelled data. Apply the classifier to the entire sample. Add examples to the training set, where the predicted class probability is above a threshold. Hence, we obtain pseudo labels.
	2. Repeat iteratively. Teh training set will tend to grow, while the residual will tend to shrink.
	3. Stop. If training parameters are held constant, algorithm will eventually converge on a stable residual set.
	4. Apply algorithm from final training step on unseen test data.

## Annotations 📖

“If one begins with a small set of seed examples representative of two senses of a word, one can incrementally augment these seed examples with additional examples of each sense, using a combination of the one-senseper-collocation and one-sense-per-discourse tendencies.” ([Yarowsky, 1995, p. 190](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=2&annotation=8E6UHIZS))

“Indeed, any supervised classification algorithm that returns probabilities with its classifications may potentially be used here.” ([Yarowsky, 1995, p. 190](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=2&annotation=G6N4IZV9))

“This could be accomplished by hand tagging a subset of the training sentences” ([Yarowsky, 1995, p. 191](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=3&annotation=CIBF7H73))

“The remainder of the examples (typically 85-98%) constitute an untagged residual.” ([Yarowsky, 1995, p. 191](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=3&annotation=VYSULHTM))

“Train the supervised classification algorithm on the SENSE-A/SENSE-B seed sets.” ([Yarowsky, 1995, p. 191](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=3&annotation=3U6E8MSK))

“Apply the resulting classifier to the entire sample set. Take those members in the residual that are tagged as SENSE-A or SENSE-B with probability above a certain threshold, and add those examples to the growing seed sets.” ([Yarowsky, 1995, p. 192](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=4&annotation=2JZLN84D))

“Optionally, the one-sense-per-discourse constraint is then used both to filter and augment this addition.” ([Yarowsky, 1995, p. 192](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=4&annotation=4NMNNSI6))

“When the training parameters are held constant, the algorithm will converge on a stable residual set.” ([Yarowsky, 1995, p. 192](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=4&annotation=ITD7FHRK))

“Repeat Step 3 iteratively. The training sets (e.g. SENSE-A seeds plus newly added examples) will tend to grow, while the residual will tend to shrink” ([Yarowsky, 1995, p. 192](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=4&annotation=8WXE69Z4))

“The classification procedure learned from the final supervised training step may now be applied to new data, and used to annotate the original untagged corpus with sense tags and probabilities.” ([Yarowsky, 1995, p. 193](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=5&annotation=HNWDELQY))

“Also, for an unsupervised algorithm it works surprisingly well, directly outperforming Schiitze's unsupervised algorithm 96.7 % to 92.2 %, on a test of the same 4 words. More impressively, it achieves nearly the same performance as the supervised algorithm given identical training contexts (95.5 % 19” ([Yarowsky, 1995, p. 195](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=7&annotation=FRKWZ93T))

“vs. 96.1%) , and in some cases actually achieves superior performance when using the one-sense-perdiscourse constraint (96.5 % vs. 96.1%). This would indicate that the cost of a large sense-tagged training corpus may not be necessary to achieve accurate word-sense disambiguation.” ([Yarowsky, 1995, p. 196](zotero://select/library/items/RUPT7G2Y)) ([pdf](zotero://open-pdf/library/items/BJB2UFED?page=8&annotation=RUY38RY4))