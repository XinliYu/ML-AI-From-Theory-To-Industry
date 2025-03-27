Reinforcement Learning
======================

:newconcept:`Reinforcement Learning (RL)` is a ML approach that builds a model (known as :newconcept:`policy` in RL context, denoted by $\\pi$) to decide the ML/AI system's next action (in RL context, a ML/AI system is also called an :newconcept:`agent`). We use $x_t \in X$ to denote the system's current :newconcept:`state` at time $t$, where $x_t$ consists of all the context the system can observe at this time $t$ (before any further action), and $X$ is the :newconcept:`state space`. We let $a_t=\\pi(x_t), a_t \\in A$ be the :newconcept:`action` suggested by the policy $\\pi$, where $A$ is the :newconcept:`action space` (the set of all possible actions under policy $\\pi$). For example,

* In the context of language models, the current state $x_t$ can be the input tokens, and the next action $a_t$ can be the next token, and the action space $A$ is the vocabulary.
* In the context of search/recommendation/Ads system, the current state $x_t$ is all the historical and runtime context the system can observe (user profile/history, runtiem session signals, etc., see also `Recommendation ML/AI System Design <../../system_design/recommendation_and_ads_system_design/01_recommendation_system_design.html>`_), and the next action $a_t$ represents the results presented to the user, and the action space is the infinitely many result combinations.

Mathematically, a policy is a probability distribution $p$ over the action space $A$ given a known state $x_t$ where, i.e.,

.. math::
   \pi(x_t) \sim p(a_t|x_t), a_t \in A, x_t \in X


Reward & Value Functions
------------------------

:newconcept:`Reward Function`, :newconcept:`Value Function` (also called :newconcept:`State Value Function`) and :newconcept:`Q-Function` (also called :newconcept:`Action-Value Function`) are three fundamental and related concepts.

* **Reward Function** $R(x_t, a_t)$: Provides immediate reward after a single action $a_t$ is taken at a given state $x_t$.
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

The value function and Q-function enable RL models to consider long-term consequences rather than just immediate rewards, supporting a crucial feature for RL - :newconcept:`Exploration-Exploitation Tradeoff`:

  * **Exploration**: Trying new actions to discover potentially better strategies
  * **Exploitation**: Using known good strategies to maximize rewards

This trade-off is crucial for real-world business applications. The following example of a "Chatbot Debugging Assistant" demonstrates how an immediately higher reward does not necessarily lead to the best long-term outcome.

* While exploitation functions like a classic supervised learning approach that leverages existing knowledge to obtain immediate rewards, exploration enables the RL framework to :ub:`discover valuable actions that may not be immediately apparent or sufficiently represented in the training data`.
* These new observations can :ub:`enhance the model's breadth and diversity of knowledge`, leading to improved future rewards compared to solely exploiting current knowledge without exploring underrepresented actions.
* Therefore, the RL approach is typically maximizing the :newconcept:`Long-Term Cumulative Reward` over time rather than focusing solely on the immediate reward, through the exploration-exploitation trade-off.
* In practice, this trade-off is :ub:`typically controlled by RL framework's hyper-parameters`.

.. admonition:: Example: Chatbot Debugging Assistant
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
--------------------

:newconcept:`Deep Reinforcement Learning (Deep RL)` leverages deep neural networks within the reinforcement learning framework. There are two major types: :newconcept:`Deep Policy Networks`, which directly model policies \(\pi(a_t|x_t)\), and :newconcept:`Deep Q-Networks (DQN)`, which directly estimate the long-term cumulative value of actions.

* Both methods are :newconcept:`action-centric modeling`. The common goal of both policy networks and DQNs is to determine system-side actions that maximize long-term business rewards (e.g., revenue, user engagement, retention).

  * In practical search, recommendation, or advertising systems, both methods typically operate at the session, sub-session, or engagement level rather than at the individual interaction level (e.g., clicks). For instance:

    * **Search/Ads**: One user query combined with interactions on the query results or ads.
    * **Recommendations**: One browsing session on a content feed along with related user interactions (e.g., likes, replies).
    * **Ads**: An email campaign coupled with subsequent user interactions with the email content.

* Their major difference is their modeling mechanisms and optimization objectives.

  * **Deep Policy Networks** explicitly model the policy by predicting a probability distribution over action space $A$ given the current state $x_t$. Their training objective typically involves :ub:`maximizing the expected cumulative future rewards (returns)`. Optionally, policy networks can be trained together with supervised signals alongside RL signals.
  * **Deep Q-Networks (DQN)** directly model the action-value function $Q(x_t, a_t)$, representing the expected cumulative future reward for taking an action $a_t$ at state $x_t$. DQNs are trained by :ub:`minimizing the temporal difference (TD) error`, where the network’s predicted values are iteratively aligned with target estimates based on observed rewards and future Q-values. Once trained, optimal actions are chosen by selecting the highest estimated Q-values.

The :newconcept:`Multi-Armed Bandits (MABs)` is :ub:`a special RL approach popular in its application to search/recommendation/ads systems`.

* It can be viewed as simplified value learning that assumes the system state will not be impacted by actions (kind of single-state Q-learning but not exactly). It still helps balance the exploration-exploitation trade-off by dynamically selecting items to present to users (through sampling), while aiming to maximize system's future cumulative performance (in comparison to always choosing the optimal actions).
* MAB assumes the system state is not significantly impacted by actions, effectively considering each action's outcome as independent of previous actions. This assumption aligns with scenarios where the system or user state remains relatively stable over short periods. Even when sessions exist, session history can be incorporated as context signals (known as :newconcept:`Contextual Bandit`). For engagements that are far apart, any state shifts can be treated as independent interactions, with recent user interactions input as context.

.. note::

    Value function is rarely involved for deep RL for search/recommendation/ads, because it is not action-centric (the value function itself is not sufficient for action selection).

.. note::

    The "long-term value" concept in Multi-Armed Bandits (MABs) differs from traditional RL. In MABs, "long-term" typically refers to cumulative performance improvement across the entire future decision horizon in a non-sequential setting. This is :ub:`conceptually closer to continuously improving a supervised learning model through additional data collection`. In contrast, for traditional RL approaches, "long-term value" often specifically refers to maximizing cumulative rewards within episodes or sessions where current decisions affect future states, although this process ultimately aims to improve model performance across the entire future decision horizon.


Multi-Armed Bandits (MABs)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Multi-Armed Bandits (MABs) framework provides a principled approach to balancing exploration and exploitation in decision-making under uncertainty. It derives its name from the "one-armed bandit" casino slot machines, conceptualizing a scenario where an agent must decide which of several slot machines (arms) to play to maximize cumulative rewards.

In the standard MAB setting:

* The agent has $K$ arms (actions) to choose from: $\mathcal{A} = \{a_1, a_2, ..., a_K\}$
* Each arm $a$ has an unknown reward distribution with mean $\mu_a$
* In each round $t$, the agent selects an arm $a_t$, takes action, and observes a reward $r_t$ (provided by a reward function)

  * In search/recommendation/ads context, this behavior is modified to select a slate of $k$ arms (i.e., top-$k$ selection from the item candidates) to propose to the users, and observes a total reward $r_t$ from the outcome of user engagement with the arms (items).

* The objective is to maximize the cumulative reward $\sum_{t=1}^T r_t$ over $T$ rounds

The performance of a MAB algorithm is typically measured by :newconcept:`regret`, which quantifies the difference between the rewards obtained by always choosing the optimal arm and the rewards obtained by the algorithm:

.. math::

   R(T) = T \cdot \max_a \mu_a - \mathbb{E}\left[\sum_{t=1}^T r_t\right]

Modeling Without Explicit Uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In MABs context, **modeling without explicit uncertainty refers** to direct regression toward target rewards. Assume the MABs framework employs a model (e.g. :refconcept:`Transformer Architecture`) to capture sequential dependencies in user behavior:

.. math::

  r_{\mathbf{u},\mathbf{I}} = f_\mathbf{\theta}(\mathbf{u}, \mathbf{I}, \mathbf{H})

where:

* :math:`\mathbf{u}` represents encoded user features
* :math:`\mathbf{I}` represents encoded item features (multiple but limited number of items); we target to rank them and take the top-$k$
* :math:`\mathbf{H}` represents encoded runtime contextual features

  * :math:`\mathbf{H}` is a sequential encodings of past interactions in the current runtime session (e.g., by a `TransformerEncoder <02_transformer_models.html#code-transformer-encoder>`_)..

    .. math::

        \mathbf{H} = \text{SequentialEncoder}([\mathbf{h_1}, \mathbf{h_2}, ..., \mathbf{h_n}])

    where:

    * :math:`\mathbf{h}_j` is the encoding of the j-th interaction (including its post-action results).
    * :math:`\mathbf{H}` is the contextualized representation of the interaction history.
    * When using neural models, a masking mechanism similar to `transformer decoder masking <02_transformer_models.html#code-transformer-masking>`_ need to be applied.

* :math:`f_\mathbf{\theta}` is a model with parameters :math:`\mathbf{\theta}`

The following is an example design of $f$, consisting of two modules.

1. :ub:`User-Item Interaction Module`: Captures the interactions between user, current item, and historical context, and mix the features. For example, using a `TransformerEncoder <02_transformer_models.html#code-transformer-encoder>`_.

   .. math::

      \mathbf{u}, \mathbf{I}, \mathbf{H} = \text{UserItemInteraction}([\mathbf{u}, \mathbf{I}, \mathbf{H}])

   For simplicity we still use the same letters to denote the post-mixing features.

2. :ub:`Reward Head`: Estimates the expected reward $\\hat{r}_{\\mathbf{u},\\mathbf{I}}$ for each user-item pair

   .. math::

        \hat{r}_{\mathbf{u},\mathbf{I}} = \text{RewardHead}(\mathbf{u},\mathbf{I}, \mathbf{H})

   The reward head can be a simple MLP between $\\mathbf{u}$ and $\\mathbf{I}$ (ignoring $\\mathbf{H}$), or before that we can further pre-process by pooling across $[\\mathbf{u}, \\mathbf{H}]$, depending on the requirements and experiments.

In practice, different types of losses can be used together with multiple head,

Regression Loss
"""""""""""""""

The training objective is straightforward regression to predict reward values. For example, using :newconcept:`Mean Absolute Error (MAE)` or :newconcept:`Mean Square Error (MSE)` loss. Additional regularization terms (e.g., :newconcept:`L2 regularization`) may be added to prevent overfitting.

* MAE can be :ub:`straightforwardly interpreted` as how far off predictions are on average. MAE :ub:`gradient is not continuous` but it is :ub:`not so sensitive to outliers`.
* MSE has :ub:`smooth gradient` but :ub:`sensitive to outliers`.

.. math::

  \mathcal{L}_{\text{MAE}}(\mathbf{\theta}) = \text{mean}(|(r - \hat{r})|) + \text{regularization}

  \mathcal{L}_{\text{MSE}}(\mathbf{\theta}) = \text{mean}((r - \hat{r})^2) + \text{regularization}

Other regression losses might also be suitable,

* The :newconcept:`Huber Loss` combines the best properties of **MSE Loss** and **MAE Loss** by being quadratic for small errors and linear for large errors, making it less sensitive to outliers.

  .. math::

      \mathcal{L}_{\text{Huber}}(\mathbf{\theta}) = \text{mean}\left(\sum_{i=1}^{N} L_\delta(r_i - \hat{r}_i)\right) + \text{regularization}

  where:

  .. math::

      L_\delta(a) =
      \begin{cases}
      \frac{1}{2}a^2 & \text{for } |a| \leq \delta \\
      \delta(|a| - \frac{1}{2}\delta) & \text{otherwise}
      \end{cases}

  The parameter :math:`\delta` controls the transition point between quadratic and linear behavior. Smaller values of :math:`\delta` make the loss more robust to outliers but may slow down learning for small errors. In the MAB context, Huber loss is particularly useful when:

  * Reward distributions have heavy tails or occasional extreme values
  * You want stability in training without completely discarding large errors

  :newconcept:`Log-Cosh Loss` is a smooth approximation of the Huber Loss that is twice differentiable everywhere, making the gradient more smooth and suitable for optimization algorithms that use second derivatives.

  .. math::

        \mathcal{L}_{\text{LogCosh}}(\mathbf{\theta}) = \text{mean}(\sum \log(\cosh(r_i - \hat{r}_i))) + \text{regularization}

  where :math:`\cosh(x) = \frac{e^x + e^{-x}}{2}` is the :newconcept:`hyperbolic cosine function`. The gradient of this loss is sigmoid-like - the derivative of :math:`\log(\cosh(a))` is :math:`\tanh(a) = 2\sigma(2a) - 1` where $\\sigma$ is the standard logistic function.

  .. note::

      The somewhat strange formulation of Huber loss (like the coefficient $\\frac{1}{2}$) is to ensure a continuous derivative.

      * For $|a| ≤ δ$: the derivative is $a$.
      * For $|a| > δ$: the derivative is $δ \\cdot \\text{sign}(a)$.
      * At $|a| = δ$: both sides have the same derivative ($±δ$).

      This is effectively capping the loss gradient, as showing in the following visualization. We also compare with Log-Cosh loss in the visualization, whose gradient is sigmoid-like.

      .. react-component:: ../../_static/images/modeling/classic_modeling/reinforcement_learning/HuberLossVisualizer.tsx
         :width: auto
         :max-width: 1000px
         :center:
         :katex:

  .. note::

        The true gradient of a neural network :math:`f_{\mathbf{\theta}}` is computed with respect to all its parameters :math:`\mathbf{\theta}`. However, this complete gradient is complex to analyze directly, so we typically focus on gradients at the output/loss layer with respect to intermediate variables like :math:`\mathbf{z}_{\mathbf{\theta}}` (e.g., predicted reward scores, logits, etc., which are parameterized by :math:`\mathbf{\theta}`). By the chain rule:

        .. math::

            \frac{\partial \mathcal{L}}{\partial \mathbf{\theta}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\mathbf{\theta}}} \cdot \frac{\partial \mathbf{z}_{\mathbf{\theta}}}{\partial \mathbf{\theta}}

        The gradient term :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\mathbf{\theta}}}` acts as a scaling factor that can effectively amplify or zero out downstream gradients, making it critical for model behavior. This is why our discussions of gradients typically focus on variables near the output/loss layer rather than tracking through the entire network.


* :newconcept:`Quantile Regression` provides a more complete view of the relationship between prediction and outcome by estimating conditional quantiles rather than just the conditional mean.

  For a specific quantile :math:`\tau \in (0,1)`, the loss is:

  .. math::

        \mathcal{L}_{\text{Quantile}}(\mathbf{\theta}, \tau) = \text{mean}(\rho_\tau(r_i - \hat{r}_i)） + \text{regularization}

  where:

    .. math::

        \rho_\tau(a) =
        \begin{cases}
        \tau \cdot a & \text{if } a \geq 0 \\
        (1-\tau) \cdot (-a) & \text{if } a < 0
        \end{cases}

  when :math:`\tau = 0.5`, this corresponds to :newconcept:`Median Regression`, which is $0.5 \\times (\\text{MAE loss})$. Values of :math:`\tau` closer to 0 or 1 focus on lower or upper quantiles of the distribution, respectively.

  The parameter :math:`\tau` controls which quantile is estimated through asymmetric weighting of errors:

  * When the model predicts too high (:math:`\hat{r}_i > r_i`, so :math:`a < 0`), the error is weighted by :math:`(1-\tau)`
  * When the model predicts too low (:math:`\hat{r}_i < r_i`, so :math:`a > 0`), the error is weighted by :math:`\tau`

  This setup will optimize toward $P(r_i > \\hat{r}_i) = \tau$, meaning only :math:`\tau`-quantile is underestimation. This makes :ub:`quantile regression particularly suitable for a risk-averse model` - we set a large $\\tau$ (e.g., 0.9) to ensure only a small fraction is overestimation.

  .. note::

      During optimization, the model is incentivized to position its predictions at the :math:`\tau`-quantile due to the mathematics of the loss function. Let's examine the derivative of the loss function with respect to the prediction :math:`\hat{r}_i`:

        .. math::

            \frac{\partial \rho_\tau(r_i - \hat{r}_i)}{\partial \hat{r}_i} =
            \begin{cases}
            -\tau & \text{if } r_i > \hat{r}_i \text{ (i.e., } a > 0 \text{)} \\
            (1-\tau) & \text{if } r_i < \hat{r}_i \text{ (i.e., } a < 0 \text{)}
            \end{cases}

      At the optimum, the expected gradient should equal zero:

        .. math::

            \mathbb{E}\left[\frac{\partial \rho_\tau(r_i - \hat{r}_i)}{\partial \hat{r}_i}\right] = 0

      This expectation can be written as:

        .. math::

            -\tau \cdot P(r_i > \hat{r}_i) + (1-\tau) \cdot P(r_i < \hat{r}_i) = 0

      Rearranging, we get:

        .. math::

            \tau \cdot P(r_i > \hat{r}_i) = (1-\tau) \cdot P(r_i < \hat{r}_i)

      This simplifies to:

        .. math::

            P(r_i > \hat{r}_i) = \tau

      Meaning that the probability of the true value exceeding the prediction is exactly :math:`\tau`, which is precisely the definition of the :math:`\tau`-quantile.

      .. react-component:: ../../_static/images/modeling/classic_modeling/reinforcement_learning/QuantileLossVisualizer.tsx
         :width: auto
         :max-width: 1000px
         :center:
         :katex:


Classification & Ordinal Loss
"""""""""""""""""""""""""""""

As mentioned earlier, MABs can be viewed as a generalization of supervised learning where the training targets can be any numerical numbers in general. As a special case, if the reward is binary $0$ or $1$ (or $-1$, $+1$, as long as it distinguishes the two classes), then the above model effectively becomes supervised learning that can work with **binary classification loss**. In this case, we follow supervised-learning conversion to denote the label (reward) as $y$ and the estimation as $\\hat{y}$.

Reward functions in practice are often synthetic and discontinuous in nature (e.g., :ref:`ecommerce reward function <code-example-ecommerce-reward-function>`), even if it appears to be numeric. Therefore a common strategy to simplify the reward is using ordinal categories $C$. The categorization can be done by

* Simply rounding the numerical rewards.
* Using milestone events, such as ``{no-action:0, click:1, dwell-60sec-plus:2, add-to-cart:3, purchase:4}``.

The the reward head will typically predict a distribution over the categories $\\hat{y}_i \sim \\hat{p}_{i, c}, c \\in C$, where $\\hat{p}_{i, c}$ is the probability selecting item $i$ will result in a reward in category $c$. This loss design is suitable when there are clear milestone events in the application, and the reward is itself synthetic and based on the milestone events. Then **multi-class classification loss** or **ordinal classification loss** can then be applied. During inference, a reward can still be estimated and apply the :refconcept:`MBAs Exploration Strategies`.

.. math::

    \hat{r}_i = \sum_{c \in C} c \cdot \hat{p}_{i,c}

We start with :newconcept:`Classification Loss`.

* The most common binary classification loss is :newconcept:`Cross-Entropy Loss`. The following is the binary case and the multi-class case.

  .. math::

        \mathcal{L}_{\text{CE}} = -\text{mean}(y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)

        \mathcal{L}_{\text{multiclass-CE}} = -\text{mean}(\sum_{c \in C} \mathbb{I}(y_i = c) \log(\hat{p}_{i,c})

  where $\\mathbb{I}$ is the identity function.

  .. note::

        The gradient of cross-entropy loss with respect to the logit is the logistic function.

        .. react-component:: ../../_static/images/modeling/classic_modeling/reinforcement_learning/CrossEntropyLossVisualizer.tsx
            :width: auto
            :max-width: 1000px
            :center:
            :katex:

  To handle class imbalance issue, one may introduce weights :math:`w^+` and :math:`w^-` to reflect the inverse frequency of positive vs. negative classes. The :newconcept:`Weighted Cross-Entropy Loss` is fomulated as:

   .. math::

        \mathcal{L}_{\text{weighted-CE}}
            = -\text{mean}
            (
            w^+ \,y_i \,\log(\hat{y}_i)
            + w^- \,(1 - y_i)\,\log\bigl(1 - \hat{y}_i\bigr)
            ).

   .. math::

        \mathcal{L}_{\text{weighted-multiclass-CE}}(\mathbf{\theta}) = -\text{mean}\left(\sum_{c \in C} w_c \cdot \mathbb{I}(y_i = c) \log(\hat{p}_{i, c})\right)

  where :math:`w^+` is larger if positives are rarer; :math:`w^-` is smaller if negatives are more frequent, and similarly for :math:`w_k`.

* :newconcept:`Focal Loss` is another variant of cross-entropy, especially helpful for :ub:`extremely imbalanced binary or multi-class classification` tasks and :ub:`focusing on hard cases`.

  .. math::

      \mathcal{L}_{\text{focal-CE}}
        = -\,\alpha \,(1 - p)^\gamma \,y \,\log(p)
          \;-\;\bigl(1 - \alpha\bigr)\,p^\gamma \,(1 - y)\,\log(1 - p),

  where:

  * :math:`\alpha` is a weighting factor (e.g., balancing positives vs. negatives, similar to the weighted CE).
  * :math:`\gamma` (the :newconcept:`focusing parameter`) :math:`\ge 0` controls how strongly to reduce the loss for well-classified samples, effectively :ub:`down-weighting easy examples` and focus on misclassified or hard examples:

    * If the example is **easy** (e.g., :math:`y = 1` and :math:`p \approx 1`), then :math:`(1 - p)^\gamma` further reduces its loss, freeing capacity to learn from harder examples.
    * :math:`\gamma = 0` recovers standard cross-entropy (i.e., no down-weighting).
    * Larger :math:`\gamma` places more emphasis on hard or misclassified samples, diminishing the gradient for trivially correct ones.
    * Commonly used with :math:`\gamma = 2` or :math:`\gamma = 4` and :math:`\alpha \in [0.25, 0.75]`. Tuning is often required to match the dataset.

      *  By emphasizing hard examples, focal loss may cause the model to :ub:`output less confident probabilities`, leading to under-confident predictions.

  In the multi-class case, the model output a softmax distribution :math:`\mathbf{p} = (p_1, p_2, \dots, p_k)`, with a true label :math:`y \in \{1, 2, \dots, k\}`. A multi-class focal loss is similarly formulated as

  .. math::

        \mathcal{L}_{\text{focal-multiclass-CE}} = -\text{mean}\left(\sum_{c \in C}
        \mathbb{I}(y_i = c)
        \,\alpha_c
        \,\bigl(1 - \hat{p}_{i,c}\bigr)^\gamma
        \,\log\bigl(\hat{p}_{i,c}\bigr)\right)

  where :math:`\alpha_k` is the optional weight for class :math:`k`. If :math:`p_y` is large (easy sample), :math:`(1 - p_y)^\gamma` suppresses its loss contribution.

  .. note::

     **Origins**
     Focal loss was originally proposed for **object detection** (Lin et al., ICCV 2017), where background examples vastly outnumber foreground objects. It has since been adopted in other highly imbalanced settings.

     **How it is used in recommendation, search, and ads?**

     * While focal loss is viable in these domains (due to frequent class imbalance, e.g., very low click or conversion rates), however, :ub:`data sampling, cross-entropy with negative sampling or weighted cross-entropy is the dominant approach`. Even for recommendation and ads systems, they typically still find :ub:`cross-entropy adequate once negative sampling or weighting is well-tuned`.
     * The major reason is the focal loss penalizing easy cases, potentially making the model under-confident, and complicating the :refconcept:`Probability Calibration` that is very important for business interpretation.
     * When to Consider Focal Loss:

       1. **Extreme Rare Positives**: If the dataset has an extremely low positive rate (e.g., far below 1%) and standard negative sampling isn’t enough, focal loss can help highlight rare but important positives.
       2. **Flood of Trivial Negatives**: If there are a large number of obviously irrelevant impressions, focal loss can reduce their overshadowing effect and shift focus to borderline (hard) examples.
       3. **Experimental Tuning**: If cross-entropy + negative sampling is under-performing, trying focal loss with different :math:`\gamma` and :math:`\alpha` may improve recall for the minority.

       :ub:`A typical use case for above is Anomaly Detection`, such as network intrusion detection, where the positive cases are extremely rase.

       In practice, although cross-entropy is typically found adequate, focal loss can still be an auxiliary loss function in :refconcept:`Multi-Loss Learning` so that its "focus" can softly feedback and impact the main loss (typically Cross-Entropy).

* :newconcept:`Hinge Loss` is another option for binary classification. Unlike cross-entropy which focuses on probabilities, hinge loss enforces a margin between classes.

  * Hinge loss considers :ub:`negative class label as -1 rather than 0`.
  * Hinge loss is :ub:`directly applied on the logits`, not on after softmax, because its formula and mechanism requires non-probabilistic scores.

  For binary classification with labels :math:`y \in \{-1, +1\}` and model output :math:`\hat{y}` (a continuous numeric score), the hinge loss is defined as:

  .. math::

      \mathcal{L}_{\text{Hinge}} = \text{mean}(\max(0, \text{margin} - y \cdot \hat{y}))

  Key properties of hinge loss:

  * $\\text{margin}$ is a positive number and is typically set to 1. Higher margin in theory only scales model parameter.
  * When :math:`r \cdot \hat{y} \geq \text{margin}`, the loss is zero - the example is correctly classified and outside the margin
  * When :math:`r \cdot \hat{y} < \text{margin}`, a penalty is applied - either the example is misclassified (:math:`r \cdot \hat{y} < 0`) or falls within the margin (:math:`0 \leq y \cdot \hat{y} < \text{margin}`)
  * The :ub:`zero-gradient region` (when :math:`y \cdot \hat{y} > \text{margin}`) helps prevent overfitting by not pushing already well-classified points further

  Comparison with cross-entropy loss:

  * [Pro] Hinge loss enforces the margin, offers especially robustness for hard marginal cases.
  * [Pro & Con] Hinge loss has zero gradient for well-classified examples but tackling the hard marginal examples.

    * However, this has similar adverse effect as :refconcept:`Focal Loss` that will make model less confident in its predictions.

  * [Con] Hinge loss :ub:`does not produce probability scores`, which can hurt interpretability if a probabilistic output is needed.
  * [Con] Hinge loss is less common nowadays, but still :ub:`specifically applies to pairwise preference learning` due to its capability to separate two preferences with a margin. Cross-entropy is the de facto standard for classification tasks in modern frameworks, with extensive tooling for performance support.

  .. note::

        Hinge loss was traditionally popular in margin-based classifiers like :refconcept:`Support Vector Machines (SVMs)`.

        Hinge loss can be extended to multi-class settings (e.g., :refconcept:`Crammer-Singer Formulation`) or ordinal classification where constraints enforce ordering among classes, each with its own hinge penalty if that order is violated. However, preference learning is where today hinge loss is most commonly employed today. Pairwise preference learning is also dominant today in preference learning, therefore binary hinge loss usually suffices.

        .. react-component:: ../../_static/images/modeling/classic_modeling/reinforcement_learning/HingeLossVisualizer.tsx
            :width: auto
            :max-width: 1000px
            :center:
            :katex:

Another category of loss functions is the :newconcept:`Ordinal Classification Loss`. This is less common, but still see their often applications where milestone events are especially strong (e.g., in e-commerce, or ads). The most popular losses in this category are :ub:`adapted` from classification CE losses.

* The :newconcept:`All-Threshold Loss` (a.k.a. :newconcept:`Cumulative Probability Loss`) is a common approach for ordinal regression that models $|C|-1$ binary thresholds for |C| ordinal categories:

  .. math::

        \mathcal{L}_{\text{all-threshold}} = -\text{mean}\left(\sum_{j=1}^{|C|-1} \left[
        \mathbb{I}(y_i > j) \log(\hat{p}_{y_i>j}) +
        \mathbb{I}(y_i \leq j) \log(1-\hat{p}_{y_i>j})
        \right]\right)

  where :math:`\hat{p}_{y_i>j}` represents the probability that the predicted outcome for item i exceeds threshold j (:ub:`cumulative probabilities` above thresholds).

  .. admonition:: Example: Cumulative Probabilities Above Thresholds
        :class: example-green

        Consider a sequence of $|C|=5$ ordinal categories ``{no-action:0, click:1, dwell-60sec-plus:2, add-to-cart:3, purchase:4}``. The All-Threshold model predicts $|C|-1=4$ thresholds:

        * :math:`\hat{p}_{y_i>0}`: Probability of at least clicking (categories 1-4)
        * :math:`\hat{p}_{y_i>1}`: Probability of at least dwelling 60+ seconds (categories 2-4)
        * :math:`\hat{p}_{y_i>2}`: Probability of at least adding to cart (categories 3-4)
        * :math:`\hat{p}_{y_i>3}`: Probability of purchasing (category 4)

        For a user who adds to cart but doesn't purchase (true label 3):

        * The model should predict high probabilities for :math:`\hat{p}_{y_i>0}`, :math:`\hat{p}_{y_i>1}`, and :math:`\hat{p}_{y_i>2}`
        * The model should predict a low probability for :math:`\hat{p}_{y_i>3}`

  This approach preserves ordinality because:

  * The thresholds have an inherent ordering (:math:`\hat{p}_{y_i>0} \geq \hat{p}_{y_i>1} \geq \hat{p}_{y_i>2} \geq \hat{p}_{y_i>3}`)
  * It mathematically enforces that higher categories cannot be more likely than lower ones, preserving the ordinal relationship between categories in C.
  * A mistake predicting category 2 when the true category is 3 incurs less penalty than predicting category 0

  Despite the mathematical difference, all-threshold loss is essentially a :refconcept:`Multi-Label Binary Cross-Entropy Loss`. The :ub:`manipulation is on the label side`, so that if the highest category one training example belongs to is $j$, then it also belongs to higher categories like $j+1$. During inference, the individual probabilities for each category can be derived from these cumulative thresholds:

  .. math::

        P(\hat{y}_i = c) = P(\hat{y}_i > c-1) - P(\hat{y}_i > c)

  This convenience makes "all-threshold loss" a most popular ordinal classification loss.

* The :newconcept:`Ordinal Cross-Entropy Loss` directly extends standard cross-entropy by incorporating a distance penalty that increases with the ordinal distance between predicted and true classes:

  .. math::

      \mathcal{L}_{\text{ordinal-CE}}(\mathbf{\theta}) = -\text{mean}\left(\sum_{c \in C} w_{|y_i - c|} \cdot \mathbb{I}(y_i = c) \log(\hat{p}_{i, c})\right)

  where this is essentially a :refconcept:`Weight Cross-Entropy Loss`, and :math:`w_{|y_i - c|}` is a weight that increases with the distance between the true class :math:`y_i` and class c. Common weight formulations include squared distance (:math:`w_d = d^2`) or exponential distance (:math:`w_d = e^d - 1`).


Preference & Rank Loss
""""""""""""""""""""""

There exists a common scenario where the reward itself is implicit or hard to label directly. For instance, in search, recommendation, and advertising systems, user preferences are often only revealed through relative choices rather than through absolute ratings (e.g., thumb up one post instead of another, click one search result instead of the other). In these contexts, :newconcept:`Preference Learning` and :newconcept:`Rank Learning` approaches become especially valuable. 

:newconcept:`Pairwise Preference Loss` functions model the relative preference between pairs of items. They are particularly useful when direct reward values are unavailable but relative preferences can be annotated and inferred from user behavior. Nowadays pairwise preference learning is popular because it requires simpler labels, and is driven by its application in LLM development (a.k.a. :newconcept:`Preference Alignment`, focusing on aligning LLM with human preferences, enhancing their utility in terms of helpfulness, truthfulness, safety, and harmlessness). 

* In search/recommendation/ads systems, pairwise prefernece is mostly applied to offline learning of user prefernece. However, in some scenarios where there are only two candidates (e.g., amazon places one preferred ad right below a product intro section, and the other less preferred ad on the side), then pairwise preference learning can be applied to runtime system.
* In applications where final candidates can be often limited to two (e.g., chatbot through beam search), then pairwise preference learning can be applied to runtime system.

.. Note::

   In realistic annotation practice, :ub:`asking annotators to rank preference, especially pairwise preference, is much easier than other annotations`, less noisy and suitable for large-scale annotations. Instead of annotating A is relevant, B is not relevant (it becomes hard when relevance is not appearant), it is :ub:`easier to answer which one of A and B is more relevant`.

In pairwise preference, a pair of items $i$ and $j$ are assumed, and we use $i \\succ j$ to denote that item $i$ is preferred over item $j$ by the labels. We also denote $s_i$ and $s_j$ as the predicted scores for items $i$ and $j$ respectively. Popular pairwise losses are usually adaptation from classification losses. 

* We view $i \\succ j$ as the "positive label", and denoting $y_{ij} = 1$ if item $i$ is preferred over item $j$, and $y_{ij} = 0$ (or $y_{ij} = -1$ for hinge loss) otherwise.
* We view $\\Delta s_{_ij} = s_i - s_j$ as "reward", and $\\sigma(\\Delta s_{ij}) = \\frac{1}{1 + e^{-(s_i - s_j)}}$ as the "positive probability" ($\\sigma$ is the standard logistic function).
* Due to its pairwise nature, vague and marginal examples are more frequent in preference-labeled data. 
  
  * Many comparisons might be between almost equally relevant examples from annotated data. 
  * Human feedback for recommendation/ads are vague, i.e., user clicking on item A does not necessarily mean item B is irrelevant. 

Then the adaptation naturally follows.

* The :newconcept:`Pairwise Cross-Entropy Loss` (also known as the :newconcept:`Pairwise Logistic Loss`), and the :newconcept:`Pairwise Hinge Loss` (also called :newconcept:`Margin Ranking Loss`).

  .. math::

      \mathcal{L}_{\text{pairwise-CE}} = -\text{mean}\left( y_{ij} \log(\sigma(\Delta s_{ij})) + (1 - y_{ij}) \log(1 - \sigma(\Delta s_{ij})) \right)

  .. math::

      \mathcal{L}_{\text{pairwise-hinge}} = \text{mean}\left( \max(0, \text{margin} - \text{sign}(y_{ij}) \cdot (\Delta s_{ij})) \right)

  
  The pairwise hinge loss is more common in preference learning than in classification, because 
  
  * Hinge loss enforces a margin (see :refconcept:`Hinge Loss`), and therefore it offers robustness against marginal cases (marginal cases are more often in preference learning). Such robustness is especially valued in ads.
  * Probabilistic interpretation is less important in pairwise preference learning. We can always convert it to probability by applying the logistic function.
  * Still the most common practice is used as jointly with cross-entropy loss.
  
* :newconcept:`RankNet Loss` has exactly the same formula as the :refconcept:`Pairwise Cross-Entropy Loss`, but 

  .. math::

      \mathcal{L}_{\text{RankNet}} = -\text{mean}\left( y_{ij} \log(\sigma(\Delta s_{ij})) + (1 - y_{ij}) \log(1 - \sigma(\Delta s_{ij})) \right)

  but simply allow three level of labels
  
  * \( y_{ij} = 1 \) if item \( i \) is preferred over item \( j \).
  * \( y_{ij} = 0 \) if item \( j \) is preferred over item \( i \).
  * \( y_{ij} = 0.5 \) if items \( i \) and \( j \) are equally preferred. In this case the minimum loss is achieved when $s_i = s_j$, aligned with the semantic meaning of $y_{ij} = 0.5$.

  This is effectively handling the scenario that many comparisons between equally relevant examples.

  .. note:: ** Exact Loss Behavior When $y_{ij} = 0.5$

      When \( y_{ij} = 0.5 \), it signifies that items \( i \) and \( j \) are equally preferred. In this case, the RankNet loss function becomes:

      .. math::

         \mathcal{L}_{\text{RankNet}} = -\left( 0.5 \log(\sigma(\Delta s_{ij})) + 0.5 \log(1 - \sigma(\Delta s_{ij})) \right)

      This simplifies to:

      .. math::

         \mathcal{L}_{\text{RankNet}} = -0.5 \left( \log(\sigma(\Delta s_{ij})) + \log(1 - \sigma(\Delta s_{ij})) \right)

      Given that \( \sigma(x) + \sigma(-x) = 1 \), the loss further simplifies to:

      .. math::

         \mathcal{L}_{\text{RankNet}} = -0.5 \left( \log(\sigma(\Delta s_{ij})) + \log(\sigma(s_j - s_i)) \right)


      Simplifying the logarithmic terms:

      .. math::

         \mathcal{L}_{\text{RankNet}} = -0.5 \left( -\log(1 + e^{-(\Delta s_{ij})}) + \log(e^{-(\Delta s_{ij})}) - \log(1 + e^{-(\Delta s_{ij})}) \right)

      .. math::

         \mathcal{L}_{\text{RankNet}} = -0.5 \left( -\log(1 + e^{-(\Delta s_{ij})}) - (\Delta s_{ij}) - \log(1 + e^{-(\Delta s_{ij})}) \right)

      Combining terms:

      .. math::

         \mathcal{L}_{\text{RankNet}} = -0.5 \left( -2\log(1 + e^{-(\Delta s_{ij})}) - (\Delta s_{ij}) \right)

      .. math::

         \mathcal{L}_{\text{RankNet}} = \log(1 + e^{-(\Delta s_{ij})}) + 0.5(\Delta s_{ij})
      
      * When $\\Delta s_{ij} >> 0$, the logarithm term approaches $0$, and the loss becomes $0.5(\\Delta s_{ij})$. When $\\Delta s_{ij} << 0$, the logarithm term approaches $-\\Delta s_{ij}$, and the loss becomes $-0.5(\\Delta s_{ij})$. Therefore $\\mathcal{L}_{\text{RankNet}} \\approx 0.5|\\Delta s_{ij}|$ when $|\\Delta s_{ij}|$ is large.
      * Obviously the loss achieves minimum $0$ when $|\\Delta s_{ij}| = 0$


:newconcept:`Listwise Ranking Loss` functions consider entire ranked lists, rather than just pairs of items. They aim to optimize the overall quality of a ranking.


:newconcept:`ListNet Loss` converts ranking scores into probability distributions using the softmax function and then compares these distributions using cross-entropy:

.. math::

    P(i|\mathbf{s}) = \frac{e^{s_i}}{\sum_{j} e^{s_j}}

    P(i|\mathbf{y}) = \frac{e^{y_i}}{\sum_{j} e^{y_j}}

    \mathcal{L}_{\text{ListNet}} = -\text{mean}\left( \sum_{i} P(i|\mathbf{y}) \log(P(i|\mathbf{s})) \right)

where:

* $\mathbf{s}$ is the vector of predicted scores
* $\mathbf{y}$ is the vector of true relevance scores (which may be binary or graded)
* $P(i|\mathbf{s})$ and $P(i|\mathbf{y})$ are the probabilities assigned to item $i$ in the predicted and true distributions, respectively

ListNet has these key properties:

* It naturally handles multiple levels of relevance
* It optimizes the entire ranking as a unit
* It's :ub:`permutation invariant within items of the same relevance`

This loss is particularly effective for web search ranking where different documents can have varying degrees of relevance to a query.


:newconcept:`LambdaRank Loss` doesn't have a closed-form expression but is derived from the gradient of RankNet loss, scaled by the change in a non-differentiable evaluation metric (like NDCG or MAP):

.. math::

    \lambda_{ij} = \frac{\partial \mathcal{L}_{\text{RankNet}}}{\partial s_i} \cdot |\Delta \text{NDCG}_{ij}|

where:

* $\lambda_{ij}$ is the lambda gradient for a pair of items $(i, j)$
* $\Delta \text{NDCG}_{ij}$ is the change in the NDCG metric if items $i$ and $j$ were swapped in the ranking

The key insight of LambdaRank is that:

* It focuses model updates on item pairs where :ub:`improving the order would significantly impact the target metric`
* Higher up in the ranking gets more weight (errors in top positions are penalized more)
* It allows direct optimization of non-differentiable ranking metrics

This loss is extensively used in commercial search engines like Bing and is effective for directly optimizing ranking quality metrics that users care about.



:newconcept:`LambdaMART Loss` extends LambdaRank by incorporating it into a gradient boosting framework using regression trees. While not strictly a loss function, it uses the lambda gradients to guide tree construction:

.. math::

    \mathcal{L}_{\text{LambdaMART}} \approx \sum_{i,j: y_i > y_j} \lambda_{ij} \log(1 + e^{-(s_i - s_j)})

LambdaMART:

* Combines the strengths of boosted trees with lambda gradients
* Shows excellent empirical performance on benchmark datasets
* Handles :ub:`non-linear relationships effectively without requiring feature engineering`
* Is the basis for many production ranking systems in search and recommendation

Attention-based Listwise Context Loss


Recent approaches incorporate attention mechanisms to model contextual effects in rankings:

.. math::

    \mathcal{L}_{\text{AttListRank}} = -\text{mean}\left( \sum_{i} P(i|\mathbf{y}) \log(P(i|\text{Attn}(\mathbf{s}, \mathbf{C}))) \right)

where:

* $\text{Attn}(\mathbf{s}, \mathbf{C})$ applies self-attention over the scores considering the context $\mathbf{C}$
* The context $\mathbf{C}$ may include information about item positions, query intent, user history, etc.

This approach:

* Captures position bias and item interactions within a list
* Models how items might :ub:`complement or substitute for each other`
* Accounts for diversity in the ranking

This type of loss is especially valuable in recommendation systems where item diversity and complementarity are important, beyond just relevance.

Pointwise Rank Loss with Position Bias


Some approaches incorporate position bias into pointwise formulations to achieve listwise effects:

Position-aware Logistic Loss


.. math::

    \mathcal{L}_{\text{pos-logistic}} = -\text{mean}\left( \sum_{i} w(pos_i) \cdot (y_i \log(\sigma(s_i)) + (1 - y_i) \log(1 - \sigma(s_i))) \right)

where:

* $w(pos_i)$ is a position-dependent weight function, often decreasing with position
* Common weight functions include $w(pos) = \frac{1}{\log_2(pos + 1)}$ (DCG-like) or $w(pos) = e^{-\alpha \cdot pos}$

This approach:

* Approximates listwise behavior while keeping the simplicity of pointwise methods
* :ub:`Emphasizes correct predictions for top positions`
* Is computationally efficient compared to true listwise methods

Expected Reciprocal Rank (ERR) Loss


:newconcept:`ERR Loss` models user behavior as a cascade process where users scan results from top to bottom and may stop at any position if satisfied:

.. math::

    \mathcal{L}_{\text{ERR}} = -\text{mean}\left( \sum_{i} \frac{R(y_i)}{pos_i} \prod_{j < i} (1 - R(y_j)) \right)

where:

* $R(y)$ is a function mapping relevance to satisfaction probability, e.g., $R(y) = \frac{2^y - 1}{2^{y_{max}}}$
* $j < i$ refers to items ranked above item $i$

ERR loss:

* Models user satisfaction and the probability of stopping at each position
* :ub:`Highly penalizes irrelevant items in top positions`
* Has been shown to correlate well with user engagement metrics

Direct Metric Optimization


Modern approaches often attempt to directly optimize ranking metrics through differentiable approximations.

ApproxNDCG Loss


:newconcept:`ApproxNDCG Loss` creates a differentiable approximation of the Normalized Discounted Cumulative Gain (NDCG) metric:

.. math::

    \text{rank}_i \approx 1 + \sum_{j \neq i} \sigma(\alpha(s_j - s_i))

    \mathcal{L}_{\text{ApproxNDCG}} = 1 - \frac{\sum_{i} \frac{2^{y_i} - 1}{\log_2(1 + \text{rank}_i)}}{\text{IDCG}}

where:

* $\alpha$ controls the sharpness of the sigmoid approximation
* $\text{IDCG}$ is the ideal DCG (when items are perfectly ranked by relevance)

This loss:

* Directly optimizes a differentiable proxy for NDCG
* Avoids the need for pair sampling strategies
* Provides a :ub:`more direct path to optimizing the actual evaluation metric`

NeuralNDCG Loss


:newconcept:`NeuralNDCG Loss` uses neural networks to model permutations and directly optimize NDCG:

.. math::

    P_{\theta}(\pi|\mathbf{s}) = \text{SoftSortNet}_{\theta}(\mathbf{s})

    \mathcal{L}_{\text{

The connection between reinforcement learning and ranking is particularly strong, as ranking can be viewed as a sequential decision process.

### Policy Gradient for Ranking

:newconcept:`Policy Gradient for Ranking` treats item selection as a policy and uses reinforcement learning gradients:

.. math::

    \mathcal{L}_{\text{PGRank}} = -\mathbb{E}_{\pi \sim P_{\theta}(\pi|\mathbf{s})}[R(\pi, \mathbf{y})]

    \nabla_{\theta} \mathcal{L}_{\text{PGRank}} \approx -\mathbb{E}_{\pi \sim P_{\theta}(\pi|\mathbf{s})}[R(\pi, \mathbf{y}) \cdot \nabla_{\theta} \log P_{\theta}(\pi|\mathbf{s})]

where:
* $R(\pi, \mathbf{y})$ is the reward function (often an IR metric like NDCG or MAP)
* $P_{\theta}(\pi|\mathbf{s})$ is the probability of permutation $\pi$ under the current model

This approach:
* Naturally handles the exploration-exploitation tradeoff
* Can optimize non-differentiable metrics directly
* Is well-suited for online learning scenarios with user feedback

### Actor-Critic for Ranking

:newconcept:`Actor-Critic Ranking` models combine policy gradient with value function approximation:

.. math::

    \mathcal{L}_{\text{Actor}} = -\mathbb{E}_{\pi \sim P_{\theta}(\pi|\mathbf{s})}[(R(\pi, \mathbf{y}) - V_{\phi}(\mathbf{s})) \cdot \nabla_{\theta} \log P_{\theta}(\pi|\mathbf{s})]

    \mathcal{L}_{\text{Critic}} = \mathbb{E}_{\pi \sim P_{\theta}(\pi|\mathbf{s})}[(R(\pi, \mathbf{y}) - V_{\phi}(\mathbf{s}))^2]

where:
* $V_{\phi}(\mathbf{s})$ is a learned value function approximating expected reward

This approach:
* Reduces variance in gradient estimates
* Handles the credit assignment problem for sequence-level rewards
* Is effective for dynamic ranking problems where relevance depends on previously ranked items

In advertising systems, this can model how earlier ad impressions might influence the effectiveness of later ones within a user's session.

## Practical Considerations

When implementing preference and ranking losses in real systems, several practical factors must be considered:

### Training Data Generation

* **Implicit vs. Explicit Feedback**: Most search/ad systems rely on implicit feedback (clicks, dwell time) which requires careful interpretation due to position bias and presentation effects
* **Negative Sampling Strategies**: How non-clicked or non-interacted items are sampled significantly impacts model quality
  * **Uniform Sampling**: Simple but often ineffective
  * **Hard Negative Mining**: Focusing on challenging negatives that are close to the decision boundary
  * **In-batch Negatives**: Using other positives in the batch as negatives for efficiency
* **Debiasing Techniques**: Methods like Inverse Propensity Scoring (IPS) to account for position bias in click data:

  .. math::
      
      \hat{r}_{\text{debiased}} = \frac{r_{\text{observed}}}{p_{\text{observation}}}

### Computational Efficiency

* **Pair Generation**: Pure pairwise approaches generate $O(n^2)$ pairs for a list of $n$ items, which can be prohibitive
* **Sampling Strategies**: Practical implementations sample pairs or use acceleration techniques:
  * Focusing on top-k items
  * Leveraging approximate nearest neighbor search for hard negatives
  * Lambda-based approaches that focus on impactful pairs

### Multi-Objective Optimization

Real-world ranking systems often optimize multiple objectives simultaneously:

* **Trade-offs**: Balancing relevance against diversity, revenue, freshness, etc.
* **Constrained Optimization**: Ensuring minimum performance on secondary metrics while optimizing the primary objective
* **Pareto Optimization**: Finding solutions that cannot be improved on one metric without harming another

### Warm-Starting Ranking Models

* **Transfer Learning**: Using pretrained general models before fine-tuning with ranking losses
* **Two-Stage Approaches**: Combining pointwise losses for warm-starting with pairwise/listwise losses for refinement

.. math::

    \mathcal{L}_{\text{combined}} = \alpha \cdot \mathcal{L}_{\text{pointwise}} + (1 - \alpha) \cdot \mathcal{L}_{\text{pairwise/listwise}}

where $\alpha$ typically decreases during training to gradually shift from pointwise to ranking optimization.




MBAs Exploration Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following are common :newconcept:`MBAs Exploration Strategies`.

1. :ub:`ε-greedy for slate selection`: The next action $a_t$ (consisting of result items) has probability $\\epsilon$ to mix random items with top items by $\\hat{r}$.

   .. math::

     a_t =
     \begin{cases}
     \text{Top-}k\text{ items by } \hat{r} & \text{with probability } 1-\epsilon \\
     \text{Hybrid Slate} & \text{with probability } \epsilon
     \end{cases}

   where "Hybrid Slate" combines some top-ranked items with some random items to ensure partial exploration.

   Implementation details:

   .. code-block:: python
      :class: folding
      :name: epsilon_greedy_slate

        import numpy as np
        import random

        def epsilon_greedy_slate(predictions, k=5, epsilon=0.1, exploration_ratio=0.4):
            predictions = np.array(predictions)
            num_items = len(predictions)

            if k >= num_items:
                # If k is greater than or equal to the number of items, return all items sorted by predictions
                return np.argsort(predictions)[::-1]

            if random.random() < epsilon:
                # Hybrid exploration: some top items, some random items
                num_explore = max(1, int(k * exploration_ratio))

                if num_explore == k:
                    # If all items are for exploration, return all items sorted (pure exploitation)
                    return np.argsort(predictions)[::-1][:k]

                num_exploit = k - num_explore

                # Get indices of all items sorted in descending order of predictions
                sorted_indices = np.argsort(predictions)[::-1]

                # Select top items for exploitation
                top_indices = sorted_indices[:num_exploit]

                # Select remaining items for exploration
                remaining_indices = sorted_indices[num_exploit:]
                random_indices = np.random.choice(remaining_indices, num_explore, replace=False)

                # Combine exploitation and exploration indices
                return np.concatenate((top_indices, random_indices))
            else:
                # Pure exploitation: select top-k items
                return np.argsort(predictions)[::-1][:k]


2. :ub:`Sampling from Softmax`: The next action $a_t$ is a sampled according to the softmax-normalized distribution based on the estimated reward values. The sampling is exactly the same as when we perform :refconcept:`beam search` for a generative model.

   .. math::

        P(a_t^{(i)} = j) = \frac{\exp(\hat{r}^{(i)}/\tau)}{\sum \exp(\hat{r}/\tau)}

   where $a_t^{(i)} = j$ denotes the $i$th item in $a_t$ and $\\hat{r}^{(i)}$ is the estimated reward for user interacting with the $i$th item, and $\\tau$ is the **temperature**. You can also apply **top_k** and/or **top_p** normalization on the softmax before the sampling. This sampling can be achieved by ``np.random.choice`` with ``replace=False``, or ``torch.multinomial`` with ``replacement=False``.

   .. note::

      The original "sampling from softmax" for MABs is performed in a sequential way - sampling first result, then normalize the distribution again, and sampling the next, until all top-$k$ items are selected. The distribution is different in this way, more favoring items of lower probabilities. However, this is less often used now: it does not have build-in support from popular packages, and the behavior like whether to favor low-probability ones can be controlled by parameters (e.g., temperature) as mentioned above.

3. :ub:`Upper Confidence Bound (UCB) Selection`: This technique adds an additional term to the estimated reward $\\hat{r}$ by penalizing items that have been frequently presented to users (e.g., popular items that appear in many results).

  .. math::

     \text{UCB}(i) = \hat{r}^{(i)} + \alpha \sqrt{\frac{\log(t)}{N_i + 1}}

  where

  * $\\hat{r}^{(i)}$ is the estimated reward for user interacting with the $i$th item.
  * $N_i$ is the number of times item $i$ has been selected.

    * The typical practice is consider a pre-defined time window (e.g., last week, last month); but of course you may count through entire history.
    * You may apply decay factor to gradually phase out old counts (e.g., exponential scale $x^{(\\text{current_time}-\\text{history_time})}, x \\in (0, 1)$).
    * "+1" to prevent zero count.

  * $t$ is the total number of interactions during the same time window, and $\\alpha$ is a parameter controlling the exploration-exploitation trade-off (higher value for more exploration). This formula balances exploitation (first term) with exploration (second term), giving higher scores to items with promising rewards or those less frequently shown.

  UCB can be further extended with a diversity term similar to the :refconcept:`Intra-List Diversity`, called :newconcept:`Diversity-Aware UCB`:

  .. math::
     \text{UCB}_\text{diverse_select_once}(i) = \text{UCB}(i) - \beta \sum_{j \neq i} \text{sim}(\mathbf{I}_i, \mathbf{I}_{a_t^{(j)}})

     \text{UCB}_\text{diverse_sequential_select}(i) = \text{UCB}(i) - \beta \sum_{j=1}^{i-1} \text{sim}(\mathbf{I}_i, \mathbf{I}_{a_t^{(j)}})

  where:

  * The first term is the standard UCB score.
  * The second term penalizes similarity to already selected items. $\\beta$ is another parameter controlling the weight of diversity penalty (higher $\\beta$ giving more weight to diversity). The is the more popular approach with simplified computation.
  * You may direct rank all items by $\\text{UCB}_\\text{diverse_select_once}$ and select top $k$; or you may selected sequentially using $\\text{UCB}_\\text{diverse_sequential_select}$, where each selection affecting subsequent choices, and at $i$th selection step, the diversity penalty is calculated for all remaining items against selected items.

    * Either way, a similarity matrix is pre-computed between every pair of items. For sequential selection, use mask to select similarities during calculation. A torch implementation of both diversity-aware UCB formulas is as follows,

      .. code-block:: python
            :class: folding
            :name: diversity_aware_ucb_torch

            import torch

            def diversity_aware_ucb_select_once(predictions, counts, total_interactions, similarity_matrix, k=5, alpha=1.0, beta=0.5):
                """
                Select a diverse slate of items using UCB with diversity penalty (single selection step) in PyTorch.

                Args:
                    predictions: Tensor of predicted rewards [n_items]
                    counts: Tensor of item selection counts [n_items]
                    total_interactions: Total number of interactions (scalar)
                    similarity_matrix: Precomputed similarity matrix [n_items, n_items]
                    k: Number of items to select
                    alpha: UCB exploration parameter
                    beta: Diversity penalty weight

                Returns:
                    Tensor of selected item indices
                """
                device = predictions.device
                n_items = predictions.shape[0]

                # Compute UCB scores
                ucb_scores = predictions + alpha * torch.sqrt(torch.log(torch.tensor(total_interactions, dtype=torch.float, device=device)) / (counts + 1))

                # Compute diversity penalty
                diversity_penalty = beta * torch.sum(similarity_matrix, dim=1)

                # Compute final scores with diversity adjustment
                diverse_scores = ucb_scores - diversity_penalty

                # Select top-k items based on diverse scores
                selected_indices = torch.argsort(diverse_scores, descending=True)[:k]

                return selected_indices

            def diversity_aware_ucb_sequential_select(predictions, counts, total_interactions, similarity_matrix, k=5, alpha=1.0, beta=0.5):
                """
                Select a diverse slate of items using UCB with a diversity penalty (PyTorch version).

                Args:
                    predictions: Tensor of predicted rewards [n_items]
                    counts: Tensor of item selection counts [n_items]
                    total_interactions: Total number of interactions
                    similarity_matrix: Pre-computed item similarity matrix [n_items, n_items]
                    k: Number of items to select
                    alpha: UCB exploration parameter
                    beta: Diversity penalty weight

                Returns:
                    Tensor of selected item indices
                """
                n_items = len(predictions)

                # Calculate initial UCB scores for all items
                ucb_scores = predictions + alpha * torch.sqrt(torch.log(torch.tensor(total_interactions)) / (counts + 1))

                # Create a mask to track available items (True=available, False=selected or unavailable)
                available_mask = torch.ones(n_items, dtype=torch.bool, device=predictions.device)

                # Tensor to store selected indices
                selected_indices = []

                # Select items sequentially
                for i in range(min(k, n_items)):
                    if i > 0:
                        # Get selected indices as tensor
                        selected_tensor = torch.tensor(selected_indices, device=predictions.device)

                        # Calculate diversity penalty using matrix operations
                        diversity_penalty = beta * torch.sum(
                            similarity_matrix[available_mask][:, selected_tensor], dim=1
                        )

                        # Apply diversity penalty to available items
                        current_scores = ucb_scores[available_mask] - diversity_penalty
                    else:
                        # For the first item, no diversity penalty
                        current_scores = ucb_scores[available_mask]

                    # Find the best item
                    best_item_idx = torch.argmax(current_scores).item()

                    # Convert to original index
                    original_idx = torch.arange(n_items, device=predictions.device)[available_mask][best_item_idx].item()

                    # Add to selected items
                    selected_indices.append(original_idx)

                    # Mark as unavailable
                    available_mask[original_idx] = False

                    # Break if we've selected all available items
                    if not torch.any(available_mask):
                        break

                return torch.tensor(selected_indices, device=predictions.device)


  .. note::

      The term "Upper Confidence Bound" comes from the statistical concept of confidence intervals. Here's why it has this name:

      * In statistics, when we estimate a parameter (like the mean reward of an item), we typically create a confidence interval around our estimate. This interval has a lower bound and an upper bound, and we can be confident (to some statistical level) that the true value lies within this range.
      * This second term is derived from `Hoeffding's inequality <https://en.wikipedia.org/wiki/Hoeffding%27s_inequality>`_, which bounds the probability that an empirical mean deviates from the true mean by more than a certain amount.
      * The term $\\sqrt{\\frac{1}{N_i + 1}}$ is related to `Variance Scaling <https://en.wikipedia.org/wiki/Analysis_of_variance>`_. If an observation in the real world is assumed to sampled from the underlying normal distribution, then :ub:`the variance of the normal distribution is inversely proportional to the number of observations`.

4. **Thompson Sampling**: A technique to sample results from a pre-defined distribution (e.g., normal distribution, beta distribution). Without uncertainty modeling, the model directly regress against the reward targets and does not have build-in support for distribution parameter estimation. However, we can still build a distribution in a rule-based way slate selection. The following code demonstrates a random noise based on past item frequency (``counts[i]``, the same the $N_i$ in above UCB). The variance formula has a statistical foundation in variance scaling.

  .. code-block:: python

     def approximate_thompson_sampling_slate(predictions, counts, k=5, noise_scale=0.1):
         # Add noise proportional to uncertainty (approximated by inverse count)

         noisy_predictions = np.copy(predictions)
         for i in range(len(predictions)):
             # More noise for less frequently selected items
             # From variance scaling: the variance of the latent normal distribution is inversely proportional to the number of observations
             # +1 to prevent zero count
             uncertainty = 1.0 / np.sqrt(counts[i] + 1)
             noisy_predictions[i] += np.random.normal(0, noise_scale * uncertainty)

         # Select top-k items based on noisy predictions
         return np.argsort(noisy_predictions)[::-1][:k]

5. **Monte Carlo Dropout**: During inference,

   1. Turn on dropout and run inference $m$ times to obtain $m$ slates of reward estimations.
   2. Estimate mean and variance for each item.
   3. Apply Thompson Sampling to resample a reward for each item.
   4. Rerank the sampled rewards and take top-$k$ items.

   The following code specifically enable dropout during inference for ``Dropout`` modules.

   .. code-block:: python
        :class: folding
        :name: enable_dropout_inference

        import torch
        import torch.nn as nn

        def enable_dropout_inference(model):
            """
            Recursively enable dropout layers during inference.

            Args:
                model (nn.Module): The PyTorch model to modify

            Returns:
                None (modifies the model in-place)
            """
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    # Enable dropout during inference
                    module.train()

In practice, it is usually a mix of strategies. Strategy simplicity and computational efficiency is also a major consideration. :ub:`Combining context-aware UCB (select once) softmax sampling enables a flexible strategy` integrating reward score, past occurrences, diversity and controllable randomness. Plus, the whole process can be efficiently implemented with a package like Pytorch.

.. code-block:: python

    # Example data
    predictions = torch.tensor([0.2, 0.5, 0.3, 0.4, 0.1])
    counts = torch.tensor([10, 5, 8, 2, 7])
    total_interactions = counts.sum().item()
    similarity_matrix = torch.tensor([
        [1.0, 0.2, 0.3, 0.4, 0.1],
        [0.2, 1.0, 0.5, 0.3, 0.2],
        [0.3, 0.5, 1.0, 0.6, 0.4],
        [0.4, 0.3, 0.6, 1.0, 0.5],
        [0.1, 0.2, 0.4, 0.5, 1.0]
    ])

    # `selection_size` is how many items we want to select from the total 5 items
    # This is the `top-k` selection that has been discussed above.
    # However, `top_k` also refers to a parameter in softmax sampling (normalizing only across the k highest scores);
    #  thus we rename this parameter as `selection_size` to avoid confusion.
    selection_size = 3

    # UCB parameters
    alpha = 1.0
    beta = 0.5

    # Softmax parameters
    temperature = 1.0
    top_p = 0.95  # top_p removes outlier item in the bottom quantile in softmax sampling
    top_k = 5 # top_k indicates only to normalize across k highest probable items in softmax sampling

    # Compute UCB scores with diversity penalty
    ucb_scores = compute_ucb_scores(predictions, counts, total_interactions, alpha, beta, similarity_matrix)

    # Select items using softmax sampling
    selected_indices = softmax_sampling(ucb_scores, temperature, selection_size, top_p, top_k)

Exploration With Explicit Uncertainty Modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




Upper Confidence Bound (UCB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

UCB algorithms select arms based on optimistic estimates of potential rewards:

.. math::

   a_t = \arg\max_a \left( \hat{\mu}_a + c \sqrt{\frac{\ln t}{n_a}} \right)

Where:

- $\hat{\mu}_a$ is the estimated mean reward for arm $a$
- $c$ controls the exploration level
- $\ln t$ represents the natural logarithm of the current round
- $n_a$ is the number of times arm $a$ has been pulled

The UCB term $c \sqrt{\frac{\ln t}{n_a}}$ represents the uncertainty in the reward estimate, which decreases as an arm is pulled more frequently.

Thompson Sampling
^^^^^^^^^^^^^^^^^

Thompson Sampling adopts a Bayesian approach:

1. Maintain a probability distribution over the mean reward of each arm
2. Sample from each distribution to get a potential reward value for each arm
3. Select the arm with the highest sampled value

For Bernoulli rewards (e.g., click/no-click), Beta distributions are typically used:

- Initialize each arm with prior Beta($\alpha_0$, $\beta_0$)
- After observing outcome $r$ for arm $a$:
  - If $r = 1$ (success): Update to Beta($\alpha_a + 1$, $\beta_a$)
  - If $r = 0$ (failure): Update to Beta($\alpha_a$, $\beta_a + 1$)

The selection rule becomes:

.. math::

   a_t = \arg\max_a \theta_a,~\text{where}~\theta_a \sim \text{Beta}(\alpha_a, \beta_a)

4. Contextual Bandits
^^^^^^^^^^^^^^^^^^^^^

Contextual bandits extend the MAB framework by incorporating context information:

- In each round $t$, observe context $x_t \in \mathcal{X}$
- Select arm $a_t$ based on both context and past rewards
- Observe reward $r_t$

The goal is to learn a policy $\pi: \mathcal{X} \rightarrow \mathcal{A}$ that maps contexts to arms.

For linear contextual bandits, the expected reward is modeled as:

.. math::

   \mathbb{E}[r|a,x] = x^T\theta_a

Where $\theta_a$ is an unknown parameter vector for arm $a$.

Popular contextual bandit algorithms include:

- LinUCB: $a_t = \arg\max_a (x_t^T\hat{\theta}_a + \alpha\sqrt{x_t^T A_a^{-1} x_t})$
- Neural Bandits: Use neural networks to model the relationship between contexts and rewards

Advanced MAB Techniques
^^^^^^^^^^^^^^^^^^^^^^^

Batched Bandits

Process feedback in batches rather than individually, crucial for systems where immediate feedback is unavailable:

.. math::

   \hat{\mu}_a^{(b)} = \frac{1}{n_a^{(b)}} \sum_{t \in \mathcal{B}_b} r_t \cdot \mathbb{I}[a_t = a]

Where $\mathcal{B}_b$ represents the set of interactions in batch $b$.

Non-Stationary Bandits

Address environments where reward distributions change over time using techniques like:

- Sliding window averaging: $\hat{\mu}_a = \frac{1}{|\mathcal{W}|} \sum_{t \in \mathcal{W}} r_t \cdot \mathbb{I}[a_t = a]$
- Exponential weighting: $\hat{\mu}_a = \frac{\sum_{i=1}^{n_a} \gamma^{n_a-i} r_i}{\sum_{i=1}^{n_a} \gamma^{n_a-i}}$

Where $\mathcal{W}$ is a window of recent observations and $\gamma \in (0,1)$ is a decay factor.

Combinatorial Bandits

Select multiple arms simultaneously under various constraints:

- Budget constraints: $\sum_{a \in S_t} c_a \leq B$
- Diverse selection: $d(a_i, a_j) \geq \delta$ for $a_i, a_j \in S_t$

Where $S_t$ is the set of selected arms at round $t$.





Training Objectives
~~~~~~~~~~~~~~~~~~~

For Neural Thompson Sampling with continuous rewards, the primary training objective is to minimize the prediction error while capturing uncertainty:


2. **Heteroscedastic Loss**: When modeling uncertainty, the network can predict both mean and variance:

  .. math::

     \mathcal{L}_{het}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{(r_i - \mu_i)^2}{2\sigma_i^2} + \frac{1}{2}\log \sigma_i^2\right)

  Where:
  - :math:`\mu_i = f_\theta^{\mu}(\mathbf{u}_i, \mathbf{I}_i, \mathbf{H}_i)` is the predicted mean reward
  - :math:`\sigma_i^2 = f_\theta^{\sigma}(\mathbf{u}_i, \mathbf{I}_i, \mathbf{H}_i)` is the predicted variance

3. **Regularization Terms**: To prevent overfitting and encourage exploration:

  .. math::

     \mathcal{L}_{reg}(\theta) = \lambda_1 \|\theta\|_2^2 + \lambda_2 \|\nabla_{\mathbf{x}}f_\theta(\mathbf{x})\|_2^2

  Where the second term (gradient norm) encourages smoother predictions

4. **Ensemble Diversity Loss**: When training ensemble models:

  .. math::

     \mathcal{L}_{div}(\theta_1,...,\theta_K) = -\lambda_d \sum_{i \neq j} \text{dist}(f_{\theta_i}, f_{\theta_j})

  Where :math:`\text{dist}` measures functional diversity between ensemble members

The final training objective combines these components:

.. math::

  \mathcal{L}(\theta) = \mathcal{L}_{MSE}(\theta) + \mathcal{L}_{reg}(\theta) + \mathcal{L}_{div}(\theta_1,...,\theta_K)

Or, when modeling uncertainty directly:

.. math::

  \mathcal{L}(\theta) = \mathcal{L}_{het}(\theta) + \mathcal{L}_{reg}(\theta) + \mathcal{L}_{div}(\theta_1,...,\theta_K)


Implementation with Neural Thompson Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Data Collection**: For each user-item pair, store:

  - User features (:math:`\mathbf{u}`)
  - Item features (:math:`\mathbf{I}`)
  - Sequential interaction history (:math:`\mathbf{H}`)
  - Observed reward values (:math:`r`)
  - Timestamps and contextual metadata

2. **Bayesian Neural Network Training**: Train the model to estimate both reward and uncertainty:

  .. math::

     P(r|\mathbf{u}, \mathbf{I}, \mathbf{H}, \theta) = \mathcal{N}(f_\theta^{\mu}(\mathbf{u}, \mathbf{I}, \mathbf{H}), f_\theta^{\sigma}(\mathbf{u}, \mathbf{I}, \mathbf{H})^2)

  This distribution represents the model's confidence about reward predictions for each user-item pair.

3. **Uncertainty Estimation Methods**: Different approaches for approximating posterior sampling:

  a. **MC Dropout**: Apply dropout during inference to sample from an approximate posterior:

     .. math::

        \hat{r}_{\mathbf{u},\mathbf{I}}^{(s)} = f_{\theta, \text{dropout}}(\mathbf{u}, \mathbf{I}, \mathbf{H})

     for :math:`s = 1, 2, \ldots, S` samples

  b. **Ensemble Sampling**: Maintain :math:`K` neural networks trained with different initializations:

     .. math::

        \hat{r}_{\mathbf{u},\mathbf{I}}^{(k)} = f_{\theta_k}(\mathbf{u}, \mathbf{I}, \mathbf{H})

     For inference, randomly select one network :math:`k \in \{1,2,...,K\}` to generate predictions

4. **Real-time Serving Process**:

  a. Retrieve user context and interaction history
  b. Generate candidate items for recommendation
  c. For each candidate item set:
     - Sample from the posterior distribution using chosen uncertainty method
     - Rank items based on sampled rewards
  d. Apply diversity adjustments to create the final recommendation slate
  e. Serve recommendations and collect new interaction data for continuous learning


3. **Uncertainty Estimation**: For Thompson Sampling with neural networks, we need posterior uncertainty estimates. This can be done through:

  a. **MC Dropout**: Apply dropout during inference to sample from an approximate posterior

     .. math::

        \hat{r}_{u,i}^{(s)} = f_{\theta, \text{dropout}}(\mathbf{x}_{u,i})

     for :math:`s = 1, 2, \ldots, S` samples

  b. **Neural Ensemble**: Maintain :math:`K` neural networks trained on bootstrap samples or with different initializations

     .. math::

        \hat{r}_{u,i}^{(k)} = f_{\theta_k}(\mathbf{x}_{u,i})

     for :math:`k = 1, 2, \ldots, K` ensemble members

4. **Real-time Serving**:

  - When user :math:`u` views product :math:`p`, generate context features including user profile, current item, and interaction history
  - For each candidate product :math:`a` in the catalog:
    - Sample a reward prediction using MC dropout or by selecting one random network from the ensemble
    - Calculate expected reward with uncertainty
  - Recommend top 5 products by sampled reward, ensuring diversity

Handling Combinatorial Actions and Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For recommending multiple items simultaneously (slate recommendation):

1. **Diverse Sampling Strategy**: Instead of greedily selecting top-k items:

  .. math::

     \pi(\mathbf{a} | \mathbf{x}) = \prod_{i=1}^{k} \frac{\exp(\hat{r}_{u,a_i}/\tau)}{\sum_{j \notin \{a_1,...,a_{i-1}\}} \exp(\hat{r}_{u,j}/\tau)}

  Where :math:`\tau` is a temperature parameter controlling exploration

2. **Submodular Diversity Function**: Incorporate a diversity term to maximize slate coverage:

  .. math::

     S(\mathbf{a}) = \sum_{i=1}^{k} \hat{r}_{u,a_i} - \lambda \sum_{i=1}^{k}\sum_{j=1, j \neq i}^{k} \text{sim}(a_i, a_j)

  Where :math:`\text{sim}(a_i, a_j)` measures similarity between items and :math:`\lambda` controls diversity weight

3. **Thompson Sampling with IPS Correction**: For partial feedback (user engages with only some recommendations):

  .. math::

     \hat{r}_{IPS} = \frac{r_{observed}}{\pi(a_{observed} | \mathbf{x})}

  This inverse propensity scoring allows learning from partial observations

Continuous Learning and Adaptation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transformer-based neural bandit implements advanced techniques for online learning:

1. **Streaming Batch Updates**: The model parameters are updated in mini-batches as new data arrives:

  .. math::

     \theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t, \mathcal{D}_t)

  Where :math:`\mathcal{D}_t` is a batch of recent interactions

2. **Catastrophic Forgetting Mitigation**: Implement experience replay to maintain performance on older patterns:

  .. math::

     \mathcal{L}_{replay}(\theta) = (1-\alpha)\mathcal{L}(\theta, \mathcal{D}_{recent}) + \alpha\mathcal{L}(\theta, \mathcal{D}_{memory})

  Where :math:`\alpha` controls the mixture of recent and memory samples

3. **Distributional Shift Detection**: The system monitors feature distribution changes and triggers retraining:

  .. math::

     D_{KL}(P_{t}(\mathbf{x}) || P_{t-\Delta}(\mathbf{x})) > \tau_{shift}

  Where :math:`D_{KL}` is the Kullback-Leibler divergence between current and historical feature distributions



Policy Networks
^^^^^^^^^^^^^^^



Deep Q-Networks
^^^^^^^^^^^^^^^


