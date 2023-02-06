*title:* Natural Language Processing with Transformers
*authors:* Lewis Tunstall
*year:* 2022
*tags:* #transformer #nlp #tokenization #deep-learning  #layernorm
*status:* #📦 
*related:*
*code:*
*review:*

## Notes 📍

- There are obviously different types of *positional embeddings* e. g., learnable embeddings, absolute positional representations and relative positional representations. ([[@tunstallNaturalLanguageProcessing2022]]; p. 74) In the [[@vaswaniAttentionAllYou2017]] an *absolute positional encoding* is used. Sine and cosine signals are sued to encode the position of tokens. 
- *Absolute positional embeddings* work well if the dataset is small. (p.74)

## Annotations 📖

“Naturally, we want to avoid being so wasteful with our model parameters since models are expensive to train, and larger models are more difficult to maintain. A common approach is to limit the vocabulary and discard rare words by considering, say, the 100,000 most common words in the corpus. Words that are not part of the vocabulary are classified as “unknown” and mapped to a shared UNK token. This means that we lose some potentially important information in the process of word tokenization, since the model has no information about words associated with UNK.” ([Tunstall, 2022, p. 32](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=56&annotation=8I4JJSUZ))

“The feed-forward sublayer in the encoder and decoder is just a simple two-layer fully connected neural network, but with a twist: instead of processing the whole sequence of embeddings as a single vector, it processes each embedding independently. For this reason, this layer is often referred to as a position-wise feed-forward layer.” ([Tunstall, 2022, p. 70](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=94&annotation=YSJ9BSIW))

“A rule of thumb from the literature is for the hidden size of the first layer to be four times the size of the embeddings, and a GELU activation function is most commonly used. This is where most of the capacity and memorization is hypothesized to happen, and it’s the part that is most often scaled when scaling up the models.” ([Tunstall, 2022, p. 70](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=94&annotation=66BX47HG))

“Adding Layer Normalization” ([Tunstall, 2022, p. 71](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=95&annotation=CK6XCLYQ))

“As mentioned earlier, the Transformer architecture makes use of layer normalization and skip connections. The former normalizes each input in the batch to have zero mean and unity variance. Skip connections pass a tensor to the next layer of the model without processing and add it to the processed tensor.” ([Tunstall, 2022, p. 71](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=95&annotation=TF8RNHLQ))

“Post layer normalization This is the arrangement used in the Transformer paper; it places layer normalization in between the skip connections. This arrangement is tricky to train from scratch as the gradients can diverge. For this reason, you will often see a concept known as learning rate warm-up, where the learning rate is gradually increased from a small value to some maximum value during training.” ([Tunstall, 2022, p. 71](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=95&annotation=F4AN27WP))

“Pre layer normalization This is the most common arrangement found in the literature; it places layer normalization within the span of the skip connections. This tends to be much more stable during training, and it does not usually require any learning rate warm-up.” ([Tunstall, 2022, p. 71](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=95&annotation=4RWZA2EP))

“We’ve now implemented our very first transformer encoder layer from scratch! However, there is a caveat with the way we set up the encoder layers: they are totally 72 | Chapter 3: Transformer Anatom” ([Tunstall, 2022, p. 72](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=96&annotation=MH7AB3MF))

“4 In fancier terminology, the self-attention and feed-forward layers are said to be permutation equivariant—if the input is permuted then the corresponding output of the layer is permuted in exactly the same way. invariant to the position of the tokens. Since the multi-head attention layer is effectively a fancy weighted sum, the information on token position is lost.” ([Tunstall, 2022, p. 73](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=97&annotation=MMDSIEB3))

“the self-attention and feed-forward layers are said to be permutation equivariant” ([Tunstall, 2022, p. 73](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=97&annotation=TB3WJP5W))

“Positional embeddings are based on a simple, yet very effective idea: augment the token embeddings with a position-dependent pattern of values arranged in a vector. If the pattern is characteristic for each position, the attention heads and feed-forward layers in each stack can learn to incorporate positional information into their transformations.” ([Tunstall, 2022, p. 73](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=97&annotation=SH9B5BKC))

“Absolute positional representations Transformer models can use static patterns consisting of modulated sine and cosine signals to encode the positions of the tokens. This works especially well when there are not large volumes of data available.” ([Tunstall, 2022, p. 74](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=98&annotation=WYXPF9R8))

“Relative positional representations Although absolute positions are important, one can argue that when computing an embedding, the surrounding tokens are most important. Relative positional representations follow that intuition and encode the relative positions between tokens. This cannot be set up by just introducing a new relative embedding layer at the beginning, since the relative embedding changes for each token depending on where from the sequence we are attending to it” ([Tunstall, 2022, p. 74](zotero://select/library/items/HYPN9IJ9)) ([pdf](zotero://open-pdf/library/items/TVF29AAM?page=98&annotation=P3WC3ZNQ))