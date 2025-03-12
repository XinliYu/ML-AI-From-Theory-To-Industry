Reinforcement Learning
======================

:newconcept:`Reinforcement Learning (RL)` is a ML approach that builds a model (known as :newconcept:`policy` in RL context, denoted by $\\pi$) to decide the ML/AI system's next action. We use $x_t \in X$ to denote the system's current :newconcept:`state` at time $t$, where $x_t$ consists of all the context the system can observe at this time $t$ (before any further action), and $X$ is the :newconcept:`state space`. We let $a_t=\\pi(x_t), a_t \\in A$ be the :newconcept:`action` suggested by the policy $\\pi$, where $A$ is the :newconcept:`action space` (the set of all possible actions under policy $\\pi$). For example, 

* In the context of language models, the current state $x_t$ can be the input tokens, and the next action $a_t$ can be the next token, and the action space $A$ is the vocabulary.
* In the context of search/recommendation/Ads system, the current state $x_t$ is all the historical and runtime context the system can observe (user profile/history, runtiem session signals, etc., see also `Recommendation ML/AI System Design <../../system_design/recommendation_and_ads_system_design/01_recommendation_system_design.html>`_), and the next action $a_t$ represents the results presented to the user, and the action space is the infinitely many result combinations.

Mathematically, a policy is a probability distribution $p$ over the action space $A$ given a known state $x_t$ where, i.e.,

.. math::
   \pi(x_t) \sim p(a_t|x_t), a_t \in A, x_t \in X

Policy Optimization
-------------------

Reward & Value Functions
~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Reward Function`, :newconcept:`Value Function` (also called :newconcept:`State Value Function`) and :newconcept:`Q-Function` (also called :newconcept:`Action-Value Function`) are three fundamental and related concepts.

* **Reward Function** $R(x_t, a_t)$: Provides immediate reward about a single action $a_t$ in a given state $x$.
* **Value Function** $V(x_t)$: Estimates the total expected future rewards from being in state $x_t$. More formally, the value function is defined as the expected sum of discounted future rewards:

.. math::

   V(x_t) = \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k R(x_{t+k}, a_{t+k}) | x_t]

where $\\gamma$ is the discount factor (typically $<1$). The expectation $\\mathbb{E}$ is taken over all possible future trajectories.

* **Q-Function** $Q(x_t, a_t)$: Estimates the total expected future rewards from being in state $x_t$ and taken a specific action $a_t$, defined as:

.. math::
   
   Q(x_t, a_t) = \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k R(x_{t+k}, a_{t+k}) | x_t, a_t]

The relationship between Q-values and V-values is:

.. math::

    V(x_t) = \mathbb{E}_{a_t \sim \pi}[Q(x_t,a_t)] = \sum_{a \in A} \pi(a|x_t)Q(x_t,a)

In other words:

* $Q(x_t,a_t)$ tells us the value of taking a specific action $a_t$ at state $x_t$.
* $V(x_t)$ is the weighted average of Q-values over all possible future actions according to the policy.

The value function and Q-function help the model consider long-term consequences rather than just immediate rewards. See the following Example: Chatbot Debugging Assistant.

.. admonition:: Chatbot Debugging Assistant
   :class: example-green

   Consider a chatbot trained to help users debug code. The immediate reward function evaluates each response on:

   .. math::
    
      R(x_t, a_t) = \text{politeness}(a_t) + \text{relevance}(a_t)

   where scores range from 0 to 1 for each term.

   Consider two possible responses to "My code is giving an error":

   **Response A:** "Thank you for reaching out about your code error. How can I assist you today?"

    * Immediate reward calculation:
      
      * $\\text{politeness} = 1.0$ (very polite)
      * $\\text{relevance} = 0.3$ (generic, no debugging progress)
      * $R(x_t, a_t) = 1.0 + 0.3 = 1.3$

    * Expected trajectory (with $\\gamma = 0.9$):
      
      * $t+0$: $R = 1.3$ (initial response, polite but generic)
      * $t+1$: $R = 1.2$ (user explains error)
      * $t+2$: $R = 1.7$ (bot asks for stack trace: politeness=0.7, relevance=1.0)
      * $t+3$: $R = 1.5$ (user provides stack trace)
      * $t+4$: $R = 1.8$ (bot begins actual debugging)

   * Value calculation from initial state:
      
     .. math::
        
        V(x_t) &= \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k R(x_{t+k}, a_{t+k}) | x_t] \\
        &= 1.3 + 0.9(1.2) + 0.9^2(1.7) + 0.9^3(1.5) + 0.9^4(1.8) \\
        &= 1.3 + 1.08 + 1.377 + 1.097 + 1.190 \\
        &= 6.044

   * Q-function calculation for this state-action pair:
      
     .. math::
        
        Q(x_t, a_t) &= \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k R(x_{t+k}, a_{t+k}) | x_t, a_t] \\
        &= 1.3 + 0.9(1.2) + 0.9^2(1.7) + 0.9^3(1.5) + 0.9^4(1.8) \\
        &= 1.3 + 1.08 + 1.377 + 1.097 + 1.190 \\
        &= 6.044

   **Response B:** "Could you share the error message and stack trace you're seeing?"

   * Immediate reward calculation:
      
      * $\\text{politeness} = 0.7$ (direct but still professional)
      * $\\text{relevance} = 1.0$ (immediately useful for debugging)
      * $R(x_t, a_t) = 0.7 + 1.0 = 1.7$
      
   * Expected trajectory (with $\\gamma = 0.9$):
      
      * $t+0$: $R = 1.7$ (direct request for stack trace)
      * $t+1$: $R = 1.5$ (user provides stack trace)
      * $t+2$: $R = 1.8$ (bot begins debugging with complete info)
      * $t+3$: $R = 1.7$ (debugging progress)
      * $t+4$: $R = 1.6$ (resolution)

   * Value calculation from initial state:
      
     .. math::
        
        V(x_t) &= \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k R(x_{t+k}, a_{t+k}) | x_t] \\
        &= 1.7 + 0.9(1.5) + 0.9^2(1.8) + 0.9^3(1.7) + 0.9^4(1.6) \\
        &= 1.7 + 1.35 + 1.458 + 1.241 + 1.058 \\
        &= 6.807

   * Q-function calculation for this state-action pair:
      
     .. math::
        
        Q(x_t, a_t) &= \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k R(x_{t+k}, a_{t+k}) | x_t, a_t] \\
        &= 1.7 + 0.9(1.5) + 0.9^2(1.8) + 0.9^3(1.7) + 0.9^4(1.6) \\
        &= 1.7 + 1.35 + 1.458 + 1.241 + 1.058 \\
        &= 6.807

   Note that in this simplified example with deterministic transitions and a single trajectory per action, the Q-value equals the Value since there's no uncertainty in the trajectories. In a real system with stochastic transitions, Q-function would evaluate the expected rewards conditioned on taking action a_t in state x_t, while Value function would evaluate expected rewards under the policy's action choices.

   Response B achieves a higher cumulative value (6.807 > 6.044) because it leads to a trajectory with higher future rewards through more efficient problem-solving. While both responses eventually get to asking for the stack trace, Response B does so immediately, leading to faster problem resolution. This demonstrates how considering long-term value can help select actions that might not maximize immediate politeness but lead to more efficient problem-solving.

Another related concept is :newconcept:`Advantage Estimation`, denoted by $\\hat{A}_t$, and estimates how much better or worse a particular action brings in comparison to the current state. It is computed as:

.. math::

    \hat{A}_t(x_t, a_t) = Q(x_t, a_t) - V(x_t)

In reality, if it is not convenient to estimate $V(x)$ (for example there isn't another model for $V$), then the advantage can be instead calculated as:

.. math::

    \hat{A}_t = Q(x_t, a_t) - Q(x_{t-1}, a_{t-1})

where $a_{t-1}$ is the previous action that already happened, and $Q(x_{t-1}, a_{t-1})$ is the previous value.


Deep Policy Networks
~~~~~~~~~~~~~~~~~~~~

:newconcept:`Deep Reinforcement Learning (Deep RL)` leverages deep neural netowrks for RL. There are two major types, :newconcept:`Deep Policy Networks` to model both policy $\\pi(a_t|x_t)$, and :newconcept:`Deep Q-Networks (DQN)` to model long-term value. Although both are :newconcept:`action-centric modeling`, they have key difference in their training objectives in theory.

* The training targets/labels for a deep policy network is the best next action(s).
* The training targets for a deep-Q network is the long-term value.

However, in modern deep RL, :ub:`the boundary between a policy network and a Q-network have been significantly blurred`. Many "policy networks" in science publications and materials that target long-term values are actually Q-networks. 

* If we already know which next action has best long-term value, we can use these long-term optimal actions as labels to train a policy network directly.
* Conversely, if our interest primarily lies in identifying optimal actions rather than explicitly computing their long-term values, normalizing the toal Q-values across all actions as $1$ can turn a Q-network into a probability distribution over actions $p(x_t, a_t) = p(a_t|x_t)$, given that current state $x_t$ is usually known and thus $p(x_t) = 1$. This effectively makes the Q-network behave like a policy - a probability distribution $p(a_t|x_t)$.

.. note::
  
  Value function is not often leveraged for deep RL for two main reasons:
  
  * It is not action-centric, not suitable for most modern ML/AI applications.
  * Real-world ML/AI systems typically have vast or infinite state spaces (e.g., extensive user profiles, session contexts, etc.), making direct evaluation of every possible state infeasible.

Policy Networks
^^^^^^^^^^^^^^^


Multi-Armed Bandit
^^^^^^^^^^^^^^^^^^

Deep Q-Networks
^^^^^^^^^^^^^^^