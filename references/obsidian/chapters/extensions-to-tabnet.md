- See [[@huangTabTransformerTabularData2020]] for extensions.
- For pratical implementation see [Self Supervised Pretraining - pytorch_widedeep (pytorch-widedeep.readthedocs.io)](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/self_supervised_pretraining.html)

- TODO: Check if TabNet can actually be considered a Transformer or if it is just attention-based?
- See paper [[@arikTabNetAttentiveInterpretable2020]]
- cover only transformer for tabular data. Explain why.
- Are there other architectures, that I do not cover? Why is this the case?
- TabNet uses neural networks to mimic decision trees by placing importance on only a few features at each layer. The attention layers in that model replace the dot-product self-attention with a type of sparse layer that allows only certain features to pass through.
- Draw on chapter decision trees [[#^5db625]]
- Visualize decision tree-like behaviour
