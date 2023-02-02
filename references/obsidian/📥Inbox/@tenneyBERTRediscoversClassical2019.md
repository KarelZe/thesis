*title:* BERT Rediscovers the Classical NLP Pipeline
*authors:* Ian Tenney, Dipanjan Das, Ellie Pavlick
*year:* 2019
*tags:* #attention #transformer #bert # attention-heads
*status:* #📦 
*related:*
- [[@jawaharWhatDoesBERT2019]]
*code:*
*review:*

## Notes 📍

## Annotations 📖

“We build on this latter line of work, focusing on the BERT model (Devlin et al., 2019), and use a suite of probing tasks (Tenney et al., 2019) derived from the traditional NLP pipeline to quantify where specific types of linguistic information are encoded.” ([Tenney et al., 2019, p. 4593](zotero://select/library/items/AEZ4WHK5)) ([pdf](zotero://open-pdf/library/items/HKM3VWDM?page=1&annotation=5QSJKSSC))

“Building on observations (Peters et al., 2018b) that lower layers of a language model encode more local syntax while higher layers capture more complex semantics, we present two novel contributions.” ([Tenney et al., 2019, p. 4593](zotero://select/library/items/AEZ4WHK5)) ([pdf](zotero://open-pdf/library/items/HKM3VWDM?page=1&annotation=YI2DA58K))

“First, we present an analysis that spans the common components of a traditional NLP pipeline. We show that the order in which specific abstractions are encoded reflects the traditional hierarchy of these tasks. Second, we qualitatively analyze how individual sentences are processed by the BERT network, layer-by-layer. We show that while the pipeline order holds in aggregate, the model can allow individual decisions to depend on each other in arbitrary ways, deferring ambiguous decisions or revising incorrect ones based on higher-level information.” ([Tenney et al., 2019, p. 4593](zotero://select/library/items/AEZ4WHK5)) ([pdf](zotero://open-pdf/library/items/HKM3VWDM?page=1&annotation=HKZWKRYD))

“That is, it appears that basic syntactic information appears earlier in the network, while high-level semantic information appears at higher layers. We note that this finding is consistent with initial observations by Peters et al. (2018b), which found that constituents are represented earlier than coreference” ([Tenney et al., 2019, p. 4596](zotero://select/library/items/AEZ4WHK5)) ([pdf](zotero://open-pdf/library/items/HKM3VWDM?page=4&annotation=BWIW3ME9))

“We find that while this traditional pipeline order holds in the aggregate, on individual examples the network can resolve out-oforder, using high-level information like predicateargument relations to help disambiguate low-level decisions like part-of-speech. This provides new evidence corroborating that deep language models can represent the types of syntactic and semantic abstractions traditionally believed necessary for language processing, and moreover that they can model complex interactions between different levels of hierarchical information.” ([Tenney et al., 2019, p. 4597](zotero://select/library/items/AEZ4WHK5)) ([pdf](zotero://open-pdf/library/items/HKM3VWDM?page=5&annotation=SJ2X3CN2))
