---
title: Transformer Implementation for Time-series Forecasting
authors: Natasha Klingenbrunn
year: 2021
---

Tags: #auto_encoder #cyclic_feature #lstm

## Positional Encoding
- The time series is not processed sequentially; thus, the Transformer will not inherently learn temporal dependencies. To combat this, the positional information for each token must be added to the input.
- Positional encoding was achieved using $\sin()$ and $\cos()$ transformation.
- ![[sine_cosine_transform.png]]