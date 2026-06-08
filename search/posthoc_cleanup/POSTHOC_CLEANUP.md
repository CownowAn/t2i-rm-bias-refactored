\subsection{Post-hoc Cleanup}
\label{sec:method:cleanup}

The admission criterion \eqref{eq:admit} ensures that, at the moment $g_k$ is admitted, it contributes uniquely beyond $\mathcal{S}_{t-1}$. However, an attribute $g_j$ admitted in a later iteration may render an earlier $g_k$ retrospectively redundant. To address this, we perform a leave-one-out cleanup over the final pool.

For each $g_k \in \mathcal{S}_T$, define
\begin{equation}
    \widehat{A}^{\mathrm{LOO}}(g_k) \;:=\; \widehat{A}^{\mathrm{partial}}\!\left(g_k \mid \mathcal{S}_T \setminus \{g_k\}\right)
    \;=\; \frac{N}{|\mathcal{X}|} \sum_x \widehat{\mathrm{Cov}}_x\!\left(g_k,\, e^{(-k)}_x\right),
    \label{eq:loo}
\end{equation}
where $e^{(-k)}_x$ is the per-prompt residual obtained by regressing $U^{N-1}$ on $\mathcal{S}_T \setminus \{g_k\}$. By Theorem~\ref{thm:partial-equiv}, \eqref{eq:loo} measures the unique contribution of $g_k$ given the remaining members of the final pool.

We then prune sequentially: at each step, identify the member with the smallest $\widehat{A}^{\mathrm{LOO}}$; if it falls below $\tau_{\mathrm{p}}$, remove it and recompute $\widehat{A}^{\mathrm{LOO}}$ for the remaining members. We iterate until every member satisfies $\widehat{A}^{\mathrm{LOO}}(g_k) > \tau_{\mathrm{p}}$, at which point the pool is Pareto-optimal in the sense that no member is dominated by the others.