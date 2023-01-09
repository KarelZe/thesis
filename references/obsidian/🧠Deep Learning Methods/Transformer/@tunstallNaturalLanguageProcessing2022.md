*title:* Natural Language Processing with Transformers
*authors:* Lewis Tunstall
*year:* 2022
*tags:* #transformer #nlp #tokenization #deep-learning 
*status:* #📦 
*related:*
*code:*
*review:*

## Notes 📍

- There are obviously different types of *positional embeddings* e. g., learnable embeddings, absolute positional representations and relative positional representations. ([[@tunstallNaturalLanguageProcessing2022]]; p. 74) In the [[@vaswaniAttentionAllYou2017]] an *absolute positional encoding* is used. Sine and cosine signals are sued to encode the position of tokens. 
- *Absolute positional embeddings* work well if the dataset is small. (p.74)

## Annotations 📖

“Naturally, we want to avoid being so wasteful with our model parameters since models are expensive to train, and larger models are more difficult to maintain. A common approach is to limit the vocabulary and discard rare words by considering, say, the 100,000 most common words in the corpus. Words that are not part of the vocabulary are classified as “unknown” and mapped to a shared UNK token. This means that we lose some potentially important information in the process of word tokenization, since the model has no information about words associated with UNK.” ([Tunstall, 2022, p. 32](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=56&annotation=8I4JJSUZ))

“Positional embeddings are based on a simple, yet very effective idea: augment the token embeddings with a position-dependent pattern of values arranged in a vector. If the pattern is characteristic for each position, the attention heads and feed-forward layers in each stack can learn to incorporate positional information into their transformations.” ([Tunstall, 2022, p. 73](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=97&annotation=SH9B5BKC))

“Absolute positional representations Transformer models can use static patterns consisting of modulated sine and cosine signals to encode the positions of the tokens. This works especially well when there are not large volumes of data available.” ([Tunstall, 2022, p. 74](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=98&annotation=WYXPF9R8))

“Relative positional representations Although absolute positions are important, one can argue that when computing an embedding, the surrounding tokens are most important. Relative positional representations follow that intuition and encode the relative positions between tokens. This cannot be set up by just introducing a new relative embedding layer at the beginning, since the relative embedding changes for each token depending on where from the sequence we are attending to it” ([Tunstall, 2022, p. 74](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=98&annotation=P3WC3ZNQ))