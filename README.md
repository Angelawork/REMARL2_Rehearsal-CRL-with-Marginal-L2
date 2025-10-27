# REMARL2: Rehearsal CRL with Marginal L2
## Towards Addressing the Plasticity-Stability Dilemma in Continual Reinforcement Learning
---
Abstract:
  Continual reinforcement learning (CRL) presents a fundamental challenge in sequential decision-making, requiring agents to continuously acquire new skills while retaining previously learned behaviours. A key difficulty in CRL is balancing plasticity, the ability to adapt to new tasks, with stability, the preservation of past knowledge. In this work, we systematically evaluate the plasticity-stability trade-off in CRL by empirically benchmarking a diverse set of existing methods based on Proximal Policy Optimization (PPO) across a sequence of MinAtar games. Our findings reveal that most existing approaches tend to favor either plasticity or stability, with no single method consistently performing the best across all scenarios. Motivated by these findings, we propose Rehearsal CRL with Marginal L2 (\ourmethod{}), which integrates selective weight regularization with experience rehearsal. Our method achieves a better balance in the plasticity-stability trade-off, demonstrating the effectiveness of hybrid strategies that combine plasticity loss prevention and knowledge retention for improving CRL.
  
---

## Contribution(s)
1. We conduct a comprehensive empirical study of CRL using PPO as the base agent on Mi-
nAtar games (Young & Tian, 2019), systematically evaluating how different existing meth-
ods behave in terms of the plasticity-stability trade-off. Our benchmarking analysis high-
lights the strengths and limitations of existing methods, showing that most methods tend
to favor either plasticity or stability often with a sacrifice on the other aspect, struggling to
achieve a good balance.
Context: While prior studies have examined catastrophic forgetting (van de Ven et al.,
2024; Hayes et al., 2020) and plasticity loss (Juliani & Ash, 2024; Lyle et al., 2023; Abbas
et al., 2023), their evaluations are often constrained to specific tasks or supervised learning
settings. Our work provides a unified analysis of these challenges in reinforcement learning,
revealing the conditions under which different mitigation strategies succeed or fail, offering
insights for designing more effective continual learning methods.
2. We propose MARGINAL L2 and REHEARSAL REGULARIZATION as complementary ap-
proaches to address the plasticity-stability trade-off in CRL. By integrating these methods,
we introduce REMARL2, which achieves superior knowledge retention while maintaining
strong plasticity across tasks. Empirical results demonstrate that REMARL2 provides a ro-
bust solution for CRL in dynamic environments.
Context: Existing methods in CRL often fail to balance plasticity and stability effectively.
Plasticity-oriented approaches, such as L2 REGULARIZATION (Lyle et al., 2023), perform
well in plasticity but compromise knowledge retention. On the other hand, stability-oriented
methods, like EWC (Kirkpatrick et al., 2016) , prioritize retaining past knowledge but limit
flexibility, hindering the system’s ability to effectively learn new tasks. Rescaling-based
methods mitigate non-stationarity by adjusting for environmental changes but emphasize
plasticity over stability, leading to performance instability. These limitations highlight the
need for a more balanced approach.

