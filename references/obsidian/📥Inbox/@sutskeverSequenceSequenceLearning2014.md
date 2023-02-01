*title:* Sequence to Sequence Learning with Neural Networks
*authors:* Ilya Sutskever, Oriol Vinyals, Quoc V. Le
*year:* 2014
*tags:* 
*status:* #📥
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖

“In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector” ([Sutskever et al., 2014, p. 1](zotero://select/library/items/53V4NR45)) ([pdf](zotero://open-pdf/library/items/GIFI2QHW?page=1&annotation=9R6ENRET))

“First, we used two different LSTMs: one for the input sequence and another for the output sequence, because doing so increases the number model parameters at negligible computational cost and makes it natural to train the LSTM on multiple language pairs simultaneously [18].” ([Sutskever et al., 2014, p. 3](zotero://select/library/items/53V4NR45)) ([pdf](zotero://open-pdf/library/items/GIFI2QHW?page=3&annotation=XDBIHRA3))

“The resulting LSTM has 384M parameters of which 64M are pure recurrent connections (32M for the “encoder” LSTM and 32M for the “decoder” LSTM). The complete training details are given below:” ([Sutskever et al., 2014, p. 5](zotero://select/library/items/53V4NR45)) ([pdf](zotero://open-pdf/library/items/GIFI2QHW?page=5&annotation=A6KRMP78))