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






```
% '

% \begin{algorithm}

  

%   % input/ouput names

%   \SetKwInOut{Input}{Input}

%   \SetKwInOut{Output}{Output}

  

%   % caption

%   % TODO: set input and output: e. g., $\hat{e} \leftarrow$ layer_norm $(e \mid \gamma, \beta)$

%   \caption{$\operatorname{\mathtt{lee-ready}}{(t_i, a_i, b_i)}$ \label{sec:alg:lee-ready-algorithm}}

  

%   \Input{%

%     $t_i$ trade price at $i$, $a_i$ ask price at $i$, and $b_i$ bid price at $i$.

%   }

%   \Output{%

%     $o_i \in\{-1,1\}$ trade initiator at $i$. \\

%   }

  

%   \BlankLine % blank line for spacing

  

%   % start of the pseudocode

%   $m_i \leftarrow \frac{1}{2}(a_i + b_i)$ \tcc*{mid spread at $i$}

  

%   \For{$1, \cdots, I$}{

%     \uIf{$t_i > m_i$}{

%       \Return{$o_i = 1$}

%     }

%     \uElseIf{$t_i < m_i$}{

%       \Return{$o_i = -1$}

%     }

%     \Else{

%       \Return{$o_i = \operatorname{\mathtt{tick}}{(t_i, a_i, b_i)}$} \tcc*{see Section \ref{sec:tick-test}.}

%     }

%   } % end for i

%   % TODO: set input and output params

% \end{algorithm}

  

% \subsubsection{Reverse Lee and Ready

%   Algorithm (0.5~p)}\label{sec:reverse-lee-and-ready-algorithm}
```


**Notes:**
[[🔢LR algorithm notes]]
