The popular *Lee and Ready* algorithm combines the (reverse) tick test and quote rule into a single algorithm. The algorithm signs trades according to the quote rule trades. Trades at the midpoint of the spread, which cannot be classified with the quote rule, are classified with the (reverse) tick test. Drawing on the notation from chapter [[🔢Tick Test]] and [[🔢Quote Rule]], the LR algorithm can thus be defined as:
$$
  \begin{equation}

    \text{Trade}_{i,t}=

    \begin{cases}
      1, & \text{if}\ p_{i, t} > m_{i, t} \\
      0, & \text{if}\ p_{i, t} < m_{i, t}  \\
	  \operatorname{tick}(), & \text{else}.
    \end{cases}
  \end{equation}
$$

The strength of the algorithm lies in combining the strong classification accuracies of the quote rule with the universal applicability of the the tick test. As it requires both trade and quote data, it is less data efficient than its parts.  

One major limitation that the algorithm cannot resolve is, the classification of  resulting in a amibigious classifcation (Finucane)

The Lee and Ready algorithm has been extensively for the classifcation of option trades 
In empirical studies, the algorithm, due to its (ellis grauer etc.)


**Notes:**
[[🔢LR algorithm notes]]

```python
def foo(x):
	if x > 5:
		return 1
	elif x < 5:
		return 0
```



Wir betrachten $f:[0,1] \rightarrow \mathbb{R}$ definiert durch
$$
f(x)= \begin{cases}x^{\frac{3}{2}} \sin \frac{1}{x}, & x \in(0,1] \\ 0, & x=0\end{cases}
$$

$$
\begin{aligned}
& z: X \rightarrow \mathbb{R} \\
& X=\left\{(x, y) \in \mathbb{R}^2: x^2+y^2 \leq 8, x \neq 0, y \neq 0\right\} \\
& z(x, y)=\frac{1}{2 x y} \sqrt{x^2+y^2}
\end{aligned}
$$

$\operatorname{tick}\colon \mathbb{N} \to \left\{0,1\right\}$
$$
  \begin{equation}
    \operatorname{foo}(x) =
    \begin{cases}
      1, & \text{if}\ x > 5 \\
      0, & \text{if}\ x < 5  \\
	  \infty, & \text{else}
    \end{cases}
  \end{equation}
$$