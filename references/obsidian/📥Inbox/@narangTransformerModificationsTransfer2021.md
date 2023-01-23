*title:* Do Transformer Modifications Transfer Across Implementations and Applications?
*authors:* Sharan Narang, Hyung Won Chung, Yi Tay, William Fedus, Thibault Fevry, Michael Matena, Karishma Malkan, Noah Fiedel, Noam Shazeer, Zhenzhong Lan, Yanqi Zhou, Wei Li, Nan Ding, Jake Marcus, Adam Roberts, Colin Raffel
*year:* 2021
*tags:* 
*status:* #transformer #layer-norm #activations #depth
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖

“First, we apply layer normalization before the selfattention and feedforward blocks instead of after. This small change has been unanimously adopted by all current Transformer implementations because it leads to more effective training (Baevski and Auli, 2019; Xiong et al., 2020).” ([Narang et al., 2021, p. 4](zotero://select/library/items/LMH6LKL2)) ([pdf](zotero://open-pdf/library/items/EA82UUN6?page=4&annotation=K6BH3JER))

“Overall, the decoder is structured similarly to the encoder, with the following changes: First, the self-attention mechanisms are “causal” which prevents the decoder from looking at future items from the target sequence when it is fed in during training” ([Narang et al., 2021, p. 15](zotero://select/library/items/LMH6LKL2)) ([pdf](zotero://open-pdf/library/items/EA82UUN6?page=15&annotation=YAPPGE2A))