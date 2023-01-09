*title:* Transformer Interpretability Beyond Attention Visualization
*authors:* Hila Chefer, Shir Gur, Lior Wolf
*year:* 2021
*tags:* 
*status:* #📥
*related:*
- [[@abnarQuantifyingAttentionFlow2020]] (simple method to calculate attention over different layers, but does averaging)
*code:*
*review:*

## Notes 📍
Nice talk at https://m.youtube.com/watch?v=A1tqsEkSoLg


**Old method:**
Do not use.

![[chefer-attention-map.png]]

Head averaging is done by integrating the relevance and gradients signals:
$$
\mathbb{E}_h\left(\nabla \mathbf{A}^{(b)} \odot R^{\left(n_b\right)}\right)^{+}
$$
i.e., we multiply the relevance and gradients element by element, and only then average across the heads. A "weighted average" of the heads where the gradients are used as weights.

They compare against *rollout*. 

**New method:** 🔥
Proposed in [[@cheferGenericAttentionmodelExplainability2021]]. Remove LRP stuff.

## Annotations 📖

“For comparison, using the same notation, the rollout [1] method is given by: ˆ A(b) = I + EhA(b) (15) rollout = ˆ A(1) · ˆ A(2) · . . . · ˆ A(B) (16) We can observe that the result of rollout is fixed given an input sample, regardless of the target class to be visualized. In addition, it does not consider any signal, except for the pairwise attention scores.” ([Chefer et al., 2021, p. 786](zotero://select/library/items/9L5MKBEL)) ([pdf](zotero://open-pdf/library/items/LRAFWM53?page=5&annotation=4BLHCY3Z))

