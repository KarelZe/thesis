- Why was it proposed?
- Why is it included in this work?
- What is the central idea?
- How does it perform? What do other sources write about it?
- Use the problems of the single tick test to motivate extended rules like EMO / LR?
- What lead to a fine-grained  fragmentation?





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