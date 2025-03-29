Supervised Learning
===================

Generic Neural Modeling Architecture
------------------------------------

In modern practice, supervised learning models for search/recommendation/ads commonly employ :refconcept:`Transformer Architecture` as their foundational modeling architecture for both the recall and precision stages (see :refconcept:`Staged Filtering`).

.. math::

  \hat{r_1}, \hat{r_2}, ..., \hat{r_T} = f_\mathbf{\theta}(\mathbf{u}, \mathbf{Q}, \mathbf{I})

where

* :math:`\mathbf{u}` represents encoded pre-indexed user features (including user profile, interaction history, etc.).
* :math:`\mathbf{Q}` represents encoded runtime query and context features (including query, device context, session context/history, etc.).

  * :math:`\mathbf{Q}` is a sequential encodings of past interactions in the current runtime session (e.g., by a `TransformerEncoder <02_transformer_models.html#code-transformer-encoder>`_)..

    .. math::

        \mathbf{Q} = \text{SequentialEncoder}([\mathbf{q_1}, \mathbf{q_2}, ..., \mathbf{q_n}])

    where:

    * :math:`\mathbf{q}_j` is the encoding of the j-th interaction (including its post-action results).
    * :math:`\mathbf{Q}` is the contextualized representation of the interaction history.
    * A causal (auto-regressive) masking mechanism similar to `transformer decoder masking <02_transformer_models.html#code-transformer-masking>`_ is typically applied to ensure each interaction is only contextualized by previous (not future) interactions.

* :math:`\mathbf{I}` represents encoded item features of a limited number of items.
* :math:`f_\mathbf{\theta}` is a model with parameters :math:`\mathbf{\theta}`.

  * :math:`f_\mathbf{\theta}` learns to make estimations $\\hat{r_1}, ..., \\hat{r_T}$ of multiple target business metrics/labels $r_1, ..., r_T$. This is known as :newconcept:`Multi-Task Learning (MTL)` and modern supervised learning for production systems typically employ such multi-task learning strategy. Even if only one or two of the metrics are the critical (e.g., direct business goal metrics), we might still consider learning other related metrics/labels as :newconcept:`Supplemental Tasks`, as :ub:`multi-tasking usually in general benefits model performance, stability and robustness`.

    * MTL has been shown to :ub:`encourage comprehensive model learning from structures and patterns common to multiple tasks`, improving data efficiency and reducing overfitting risk, and thus leading to better generalization and predictive accuracy.
    * MTL has been shown to better generalize model to unseen data (as a result of reduced overfitting and better generalization), thereby enhancing stability.
    * MTL has also been shown help model better resist noise and variability in the data (as a result of comprehensive and generalized learning), thereby improving robustness.

The following is an example design of $f$ for a precision-stage model, consisting of two modules. The :ub:`precision stage model targets to rank the items` given all information from :math:`\mathbf{u}, \mathbf{Q}, \mathbf{I}`, and present the top-$k$ results to the user.

1. :ub:`User-Item Interaction Module`: Captures the interactions between user, current item, and historical context, and mix the features. For example, using a `TransformerEncoder <02_transformer_models.html#code-transformer-encoder>`_.

   .. math::

      \mathbf{u}, \mathbf{Q}, \mathbf{I} = \text{UserItemInteraction}([\mathbf{u}, \mathbf{Q}, \mathbf{I}])

   For simplicity we still use the same letters to denote the post-mixing features. Usually more mixing layers for precision, and less mixing layers for recall.

2. :ub:`Prediction Head`: Estimates the expected outcome $\\hat{r}$ for potential user-item interaction, where this "outcome" can be a reward (numerical value), an original label, or a categorical label.

   .. math::

        \hat{r} = \text{RewardHead}(\mathbf{u},\mathbf{Q}, \mathbf{I})

   The prediction head can optionally further pool across $[\\mathbf{u}, \\mathbf{Q}]$ to obtain a finalized $\\mathbf{u}$, then a simple layer can be applied to $\\mathbf{u}$ and $\\mathbf{I}$ to convert them into logits, for example:
   
   * May optionally first apply another :refconcept:`Positionwise Feed-Forward Networks` to enhance prediction head capability. 
   * Combine $\\mathbf{u}$ and every item in $\\mathbf{I}$ (e.g., concat), then apply a linear layer to transform then into desired prediction outputs (can be scores, two-class or multi-class logits).

     * This is typically for precision stage where the model targets to learn the item relevance and ranking. 
     * Not suitable for recall stage because user/item embeddings are combined in the output before passing to the loss.

   * Or directly compute inner products as prediction outputs (can be scores or two-class logits).
     
     * This is typically for recall stage where the model targets to learn embedding similarities and be able to output embeddings for both user and items separately.

The scores and logits are then passed to loss functions. Development of modern production search/recommendation/ads models (where ranking is fundamental) typically require learning from different types of loss functions, attaching multiple prediction heads to a shared representation network. This approach, formally known as `Multi-Objective Optimization (MOO)` (a.k.a. :newconcept:`Multi-Task Learning (MTL)`).

* This approach allows systems to flexibly and simultaneously optimize for several complementary objectives that :ub:`capture different aspects of user behavior and business goals`.
* :ub:`Each head is intended to be lightweight` (e.g., a fully-connected linear layer, or a simple MLP) in order for the model to learn a strong shared representation.

.. figure:: /_static/images/modeling/classic_modeling/supervised_learning/supervised_learning_model_architecture.png
   :alt: Recommendation System Architecture with Specialized Prediction Heads
   :width: 80%
   :align: center

   MOO model architecture for search/recommendation/ads systems.


Based on their learning objectives, loss heads in recommendation systems can be categorized into several types:

1. **Regression Heads**: Estimate continuous values like synthetic rewards/engagement scores, revenue (earnings per mile), playback time, or numeric ratings, using `Regression Loss`_ functions for optimization.
2. **Classification Heads**: Predict probabilities of discrete (e.g., simply click or not, convert or not) or progression events (e.g., view → click → add-to-cart → purchase) through sequential user journey, using `Classification & Ordinal Loss`_ functions for optimization.
3. **Ranking Heads**: Optimize the order of items in the result list, such as search result ranking, feed ranking, using `Pairwise Preference Loss`_ functions and `Listwise Ranking Loss`_ functions for optimization.
4. **Joint Loss Heads**: Combine multiple objectives (may already exists in other heads) into a single unified optimization target. This is usually the main training goal.

MOO architectures offer several advantages over single-objective or joint-objective models:

1. **Shared Representation Learning**: Lower network layers learn generalizable features useful across multiple tasks, leading to more robust representations.
2. **Complementary Signal Integration**: Diverse signals (clicks, conversions, engagement time) from different heads feedback complementary information to the model in a soft way, enriching the model's understanding.
3. **Training Efficiency**: Learning multiple losses from the same training examples improves sample efficiency.
4. **Business Flexibility**: These architectures are particularly valuable for mature recommendation platforms where multiple stakeholders have different priorities (e.g., user engagement teams vs. monetization teams).

   * While maintaining a unified model architecture, different prediction heads can align with different KPIs, and be flexibly weighted to reflect negotiated business priorities. However, the weights do need carefully experiments to balance competing priorities.


The recall-stage model is mostly the same as precision-stage model, with a key difference - it must decouple the input embeddings from the item embeddings (decoupling :math:`\mathbf{u}` and :math:`\mathbf{Q}` from :math:`\mathbf{I}`), and cannot transform :math:`\mathbf{I}` during training or runtime. This decoupling is widely known as the :newconcept:`Two-Tower Architecture`.

  * This is to mimic the runtime scenario that recall layer is a retrieval layer, and a single query embedding :math:`\mathbf{q} = h(\mathbf{u}, \mathbf{Q})` will be computed solely based on :math:`\mathbf{u}` and :math:`\mathbf{Q}` in order to retrieved items from a vast embedding store consisting of possibly billions of items. There might optionally be a :ub:`lightweight positionwise transformation layer on item embeddings` $\mathbf{I}$ to transform the item embeddings to required dimensions. All item embeddings are pre-computed and indexed; therefore no item embedding transformations are allowed during runtime.
  * The recall-stage model training focus is on optimizing embedding similarities and distances between $\\mathbf{q}$ and $\\mathbf{I}$, rather than direct rankings, using `Contrastive Loss`_ functions for optimization. Negative examples play a crucial role for the training.

Regression Loss
---------------

:newconcept:`Regression Loss`

MAE & MSE
~~~~~~~~~

The training objective is straightforward regression to predict reward values. For example, using :newconcept:`Mean Absolute Error (MAE)` or :newconcept:`Mean Square Error (MSE)` loss. Additional regularization terms (e.g., :newconcept:`L2 regularization`) may be added to prevent overfitting.

* MAE can be :ub:`straightforwardly interpreted` as how far off predictions are on average. MAE :ub:`gradient is not continuous` but it is :ub:`not so sensitive to outliers`.
* MSE has :ub:`smooth gradient` but :ub:`sensitive to outliers`.

.. math::

  \mathcal{L}_{\text{MAE}}(\mathbf{\theta}) = \text{mean}(|(r - \hat{r})|) + \text{regularization}

.. math::

  \mathcal{L}_{\text{MSE}}(\mathbf{\theta}) = \text{mean}((r - \hat{r})^2) + \text{regularization}


Huber Loss & Log-Cosh Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :newconcept:`Huber Loss` combines the best properties of **MSE Loss** and **MAE Loss** by being quadratic for small errors and linear for large errors, making it less sensitive to outliers.

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

    .. react-component:: ../../_static/images/modeling/classic_modeling/supervised_learning/HuberLossVisualizer.tsx
        :width: auto
        :max-width: 1000px
        :center:
        :katex:

.. note::

    The true gradient of a neural network :math:`f_{\mathbf{\theta}}` is computed with respect to all its parameters :math:`\mathbf{\theta}`. However, this complete gradient is complex to analyze directly, so we typically focus on gradients at the output/loss layer with respect to intermediate variables like :math:`\mathbf{z}_{\mathbf{\theta}}` (e.g., predicted reward scores, logits, etc., which are parameterized by :math:`\mathbf{\theta}`). By the chain rule:

    .. math::

        \frac{\partial \mathcal{L}}{\partial \mathbf{\theta}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\mathbf{\theta}}} \cdot \frac{\partial \mathbf{z}_{\mathbf{\theta}}}{\partial \mathbf{\theta}}

    The gradient term :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\mathbf{\theta}}}` acts as a scaling factor that can effectively amplify or zero out downstream gradients, making it critical for model behavior. This is why our discussions of gradients typically focus on variables near the output/loss layer rather than tracking through the entire network.


Quantile Regression Loss
~~~~~~~~~~~~~~~~~~~~~~~~

* :newconcept:`Quantile Regression Loss` provides a more complete view of the relationship between prediction and outcome by estimating conditional quantiles rather than just the conditional mean.

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

      .. react-component:: ../../_static/images/modeling/classic_modeling/supervised_learning/QuantileLossVisualizer.tsx
         :width: auto
         :max-width: 1000px
         :center:
         :katex:


Classification & Ordinal Loss
-----------------------------

As mentioned earlier, MABs can be viewed as a generalization of supervised learning where the training targets can be any numerical numbers in general. As a special case, if the reward is binary $0$ or $1$ (or $-1$, $+1$, as long as it distinguishes the two classes), then the above model effectively becomes supervised learning that can work with **binary classification loss**. In this case, we follow supervised-learning conversion to denote the label (reward) as $y$ and the estimation as $\\hat{y}$.

Reward functions in practice are often synthetic and discontinuous in nature (e.g., :ref:`ecommerce reward function <code-example-ecommerce-reward-function>`), even if it appears to be numeric. Therefore a common strategy to simplify the reward is using ordinal categories $C$. The categorization can be done by

* Simply rounding the numerical rewards.
* Using milestone events, such as ``{no-action:0, click:1, dwell-60sec-plus:2, add-to-cart:3, purchase:4}``.

The the reward head will typically predict a distribution over the categories $\\hat{y}_i \sim \\hat{p}_{i, c}, c \\in C$, where $\\hat{p}_{i, c}$ is the probability selecting item $i$ will result in a reward in category $c$. This loss design is suitable when there are clear milestone events in the application, and the reward is itself synthetic and based on the milestone events. Then **multi-class classification loss** or **ordinal classification loss** can then be applied. During inference, a reward can still be estimated and apply the :refconcept:`MBAs Exploration Strategies`.

.. math::

    \hat{r}_i = \sum_{c \in C} c \cdot \hat{p}_{i,c}

Cross-Entropy Loss, Weighted Cross-Entropy Loss & Focal Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The most common binary classification loss is :newconcept:`Cross-Entropy Loss`. The following is the binary case and the multi-class case.

  .. math::

        \mathcal{L}_{\text{CE}} = -\text{mean}(y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)

        \mathcal{L}_{\text{multiclass-CE}} = -\text{mean}(\sum_{c \in C} \mathbb{I}(y_i = c) \log(\hat{p}_{i,c})

  where $\\mathbb{I}$ is the identity function.

  .. note::

        The gradient of cross-entropy loss with respect to the logit is the logistic function.

        .. react-component:: ../../_static/images/modeling/classic_modeling/supervised_learning/CrossEntropyLossVisualizer.tsx
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

Hinge Loss
~~~~~~~~~~

* :newconcept:`Hinge Loss` is another option for binary classification. Unlike cross-entropy which focuses on probabilities, hinge loss enforces a margin between classes.

  * Hinge loss considers :ub:`negative class label as -1 rather than 0`.
  * Hinge loss is :ub:`directly applied on the logits`, not on after softmax, because its formula and mechanism requires non-probabilistic scores.

  For binary classification with labels :math:`y \in \{-1, +1\}` and model output :math:`\hat{y}` (a continuous numeric score), the hinge loss is defined as:

  .. math::

      \mathcal{L}_{\text{Hinge}} = \text{mean}(\max(0, \text{margin} - y \cdot \hat{y}))

  Key properties of hinge loss:

  * $\\text{margin}$ is a positive number and is typically set to 1. Higher margin in theory only scales model parameter.
  * When :math:`y \cdot \hat{y} \geq \text{margin}`, the loss is zero - the example is correctly classified and outside the margin
  * When :math:`y \cdot \hat{y} < \text{margin}`, a penalty is applied - either the example is misclassified (:math:`r \cdot \hat{y} < 0`) or falls within the margin (:math:`0 \leq y \cdot \hat{y} < \text{margin}`)
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

        .. react-component:: ../../_static/images/modeling/classic_modeling/supervised_learning/HingeLossVisualizer.tsx
            :width: auto
            :max-width: 1000px
            :center:
            :katex:


All-Threshold Loss & Ordinal Cross-Entropy Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Pairwise Preference Loss
------------------------

There exists a common scenario where the reward itself is implicit or hard to label directly. For instance, in search, recommendation, and advertising systems, user preferences are often only revealed through relative choices rather than through absolute ratings (e.g., thumb up one post instead of another, click one search result instead of the other). In these contexts, :newconcept:`Preference Learning` and :newconcept:`Rank Learning` approaches become especially valuable. 

:newconcept:`Pairwise Preference Loss` models the relative preference between pairs of items. They are particularly useful when direct reward values are unavailable but relative preferences can be annotated and inferred from user behavior. Nowadays pairwise preference learning is popular because it requires simpler labels, and is driven by its application in LLM development (a.k.a. :newconcept:`Preference Alignment`, focusing on aligning LLM with human preferences, enhancing their utility in terms of helpfulness, truthfulness, safety, and harmlessness). 

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

Pairwise Cross-Entropy Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :newconcept:`Pairwise Cross-Entropy Loss` (also known as the :newconcept:`Pairwise Logistic Loss`), and the :newconcept:`Pairwise Hinge Loss` (also called :newconcept:`Margin Ranking Loss`).

.. math::

   \mathcal{L}_{\text{pairwise-CE}} = -\text{mean}\left( y_{ij} \log(\sigma(\Delta s_{ij})) + (1 - y_{ij}) \log(1 - \sigma(\Delta s_{ij})) \right)

.. math::

   \mathcal{L}_{\text{pairwise-hinge}} = \text{mean}\left( \max(0, \text{margin} - \text{sign}(y_{ij}) \cdot (\Delta s_{ij})) \right)

  
The pairwise hinge loss is more common in preference learning than in classification, because
  
* Hinge loss enforces a margin (see :refconcept:`Hinge Loss`), and therefore it offers robustness against marginal cases (marginal cases are more often in preference learning). Such robustness is especially valued in ads.
* Probabilistic interpretation is less important in pairwise preference learning. We can always convert it to probability by applying the logistic function.
* Still the most common practice is used as jointly with cross-entropy loss.
  
RankNet Loss
~~~~~~~~~~~~

:newconcept:`RankNet Loss` has exactly the same formula as the :refconcept:`Pairwise Cross-Entropy Loss`, but

.. math::

   \mathcal{L}_{\text{RankNet}} = -\text{mean}\left( y_{ij} \log(\sigma(\Delta s_{ij})) + (1 - y_{ij}) \log(1 - \sigma(\Delta s_{ij})) \right)

but simply allow three level of labels
  
* $y_{ij} = 1$ if item $i$ is preferred over item $j$.
* $y_{ij} = 0$ if item $j$ is preferred over item $i$.
* $y_{ij} = 0.5$ if items $i$ and $j $ are equally preferred. In this case the minimum loss is achieved when $s_i = s_j$, aligned with the semantic meaning of $y_{ij} = 0.5$.

This is effectively handling the scenario that many comparisons between equally relevant examples.

.. admonition:: ** Exact Loss Behavior When ** $y_{ij} = 0.5$

   :class: note

   When $y_{ij} = 0.5$, it signifies that items $i$ and $j$ are equally preferred. In this case, the RankNet loss function becomes:

   .. math::

      \mathcal{L}_{\text{RankNet}} = -\left( 0.5 \log(\sigma(\Delta s_{ij})) + 0.5 \log(1 - \sigma(\Delta s_{ij})) \right)

   This simplifies to:

   .. math::

      \mathcal{L}_{\text{RankNet}} = -0.5 \left( \log(\sigma(\Delta s_{ij})) + \log(1 - \sigma(\Delta s_{ij})) \right)

   Given that $\\sigma(x) + \\sigma(-x) = 1$, the loss further simplifies to:

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


Listwise Ranking Loss
---------------------

:newconcept:`Listwise Ranking Loss` functions have become popular for directly optimizing the overall quality of an entire ranking list, contrasting with pairwise preference learning that focuses on localized rank optimization. These functions typically :ub:`assume each candidate item has a relevance score label` (denoted by $y_i$), and :ub:`items are ranked based on the relevance scores` in descending order. Traditionally, a large scale of relevance scores has been obtained through :refconcept:`Pairwise Preference Learning`, where models learn from relative preferences between item pairs. Recent advancements have introduced the use of LLMs to generate relevance scores for training data.

Some listwise ranking losses (`Rank-Aware Positionwise BCE`_, `LambdaRank Loss`_) fundamentally involve non-differentiable ranking require re-ranking after model updates. This re-ranking step is a core part of their training process, but itself is non-differentiable and requires approximations. The changing ranks can create :ub:`complex and unstable training dynamics due to the discontinued gradients`. Several practical mitigations:

* Don't re-rank after every single batch update (which would be expensive), but rather at certain intervals (e.g., every N batches or every epoch) to balance computational efficiency with training effectiveness.
* For each item, memorize its past few ranks and make a weighted average over them to smooth the rank change.

Other listwise ranking losses assume the relevance score distribution already encodes ranking information, and work on top of the score distribution (`ListNet Loss`_, `ApproxNDCG Loss`_).

Rank-Aware Positionwise BCE
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Rank-Aware Positionwise BCE` is a simpled weighted adaptation of binary cross-entropy.

.. math::

    \mathcal{L}_{\text{rank-BCE}} = -\text{mean}\left( \sum_{i} w_{\text{rank}_{i}} \cdot (y_i \log(\sigma(s_i)) + (1 - y_i) \log(1 - \sigma(s_i))) \right)

where

* $\\sigma$ is the logistic function to convert the score to probability.
* $w_{\\text{rank}_{i}}$ is a position-dependent weight function, typically decreasing with position like DCG. Common weight functions include $w_{\\text{rank}_{i}} = \\frac{1}{\\log_2(\\text{rank}_i + 1)}$ (DCG-like) or $w_{\\text{rank}_{i}} = e^{-\\alpha \\cdot \\text{rank}_i}$

This approach:

* Directly incorporates position information into the loss function, aligning with real-world metrics like NDCG and user behavior where top positions matter more.
* Less computationally expensive than pure listwise approaches. Empirically it is found a good approximation of listwise behavior while keeping the simplicity of a positionwise BCE.


LambdaRank Loss
~~~~~~~~~~~~~~~

:newconcept:`LambdaRank Loss` doesn't have a closed-form expression but is derived from the gradient of RankNet loss, scaled by the change in a non-differentiable evaluation metric (like NDCG or MAP):

.. math::

    \lambda_{ij} = \frac{\partial \mathcal{L}_{\text{RankNet}}}{\partial s_i} \cdot |\Delta \text{NDCG}_{ij}|

where

* $\\lambda_{ij}$ represents the gradient of the loss function with respect to the score of item i (s_i), considering its relationship with item j.
* $\\Delta \\text{NDCG}_{ij}$ is the change in the NDCG metric if items $i$ and $j$ were swapped in the ranking.
* This loss is like weighted average of the RankNet loss for every pairs in the list, where the weight is the NDCG change if item $i$ and $j$ were to swap. Higher positions in the ranking get more weight (errors in top positions are penalized more)

.. note::
    
    LambdaRank enables optimization that considers NDCG improvements without requiring NDCG to be differentiable. This is sometimes called "implicit gradient-based optimization" - we're not computing the true gradient of NDCG, but we're constructing a gradient that empirically behaves similarly to what that gradient would be if it could be computed.

    Starting with RankNet, we define the probability that item $i$ should be ranked higher than item $j$:
   
    .. math::
        
        P_{ij} = \sigma(s_i - s_j)
   
    where $\sigma$ is the sigmoid function. We can compute its gradient with respect to $s_i$
   
    .. math::
        
        \frac{\partial L_{ij}}{\partial s_i} = (P_{ij} - 1)
   
   When $i$ should be ranked higher than $j$:
   
   * If the model predicts otherwise ($s_i < s_j$), then $P_{ij}$ is close to $0$, making this gradient close to $-1$, which is a strong push to increase $s_i$ for minimization and possibly flipping the order between $i$, $j$. LambdaRank then modifies this gradient by scaling it with $|\\Delta \\text{NDCG}_{ij}|$.
   * If the model predicts this correctly ($s_i > s_j$), then $P_{ij}$ is close to 1, and the gradient is near zero. Even though $|\\Delta \\text{NDCG}_{ij}|$ might be substantial (if $i$ and $j$ have very different relevance scores), multiplying it by a gradient close to zero still results in a very small lambda value.

LambdaRank introduces the idea of directly optimizing ranking quality metrics that users care about, and it was originally proposed for neural networks. However, today :ub:`LambdaRank is mostly applied to GBDT` approaches.

* :refconcept:`LambdaMART` is an approach built on top of "LambdaRank" and uses above-mentioned lambda gradients $\\lambda_{ij}$ from LambdaRank to guide the construction of regression trees for GBDT models.
* Neural models nowadays more often use differentiable NDCG approximations like :refconcept:`ApproxNDCG`, :refconcept:`SoftNDCG` or :refconcept:`NeuralNDCG`.

LambdaRank loss does not have closed form, it can still be implemented in pytorch by directly interfering with the gradient calculation.

.. code-block:: python
   :class: folding
   :name: LambdaRankLoss

    import torch
    from torch.nn import Module
    import numpy as np
    from sklearn.metrics import ndcg_score

    class LambdaRankLoss(Module):
        def __init__(self, k=10):
            """
            Initialize LambdaRank loss with scikit-learn's NDCG implementation.
            
            Args:
                k (int): The 'k' in NDCG@k
            """
            super(LambdaRankLoss, self).__init__()
            self.k = k
            
        def forward(self, scores, labels):
            """
            Forward pass with scikit-learn NDCG calculation.
            """
            placeholder_loss = torch.zeros(1, requires_grad=True)
            scores_grad = torch.zeros_like(scores)
            
            # Move to numpy for sklearn compatibility
            scores_np = scores.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            # Get current ranking
            current_indices = np.argsort(-scores_np)  # Descending order
            
            # Compute lambdas for all pairs
            batch_size = len(scores)
            for i in range(batch_size):
                for j in range(batch_size):
                    if i == j or labels_np[i] <= labels_np[j]:
                        continue
                    
                    # Compute RankNet gradient
                    s_diff = scores[i] - scores[j]
                    ranknet_grad = 1.0 / (1.0 + torch.exp(s_diff))
                    
                    # Compute NDCG delta using sklearn
                    ndcg_delta = self._compute_ndcg_delta_sklearn(
                        labels_np, scores_np, current_indices, i, j)
                    
                    # Combine for lambda
                    lambda_ij = ranknet_grad * ndcg_delta
                    
                    # Update gradients
                    scores_grad[i] -= lambda_ij
                    scores_grad[j] += lambda_ij
            
            # Register hook
            scores.register_hook(lambda grad: scores_grad)
            return placeholder_loss
        
        def _compute_ndcg_delta_sklearn(self, labels, scores, current_indices, i, j):
            """
            Calculate NDCG delta using scikit-learn's implementation.
            """
            # Find positions in the ranking
            pos_i = np.where(current_indices == i)[0][0]
            pos_j = np.where(current_indices == j)[0][0]
            
            # Only consider changes affecting top-k
            if pos_i >= self.k and pos_j >= self.k:
                return 0.0
            
            # Create new ranking with i and j swapped
            new_indices = current_indices.copy()
            new_indices[pos_i], new_indices[pos_j] = new_indices[pos_j], new_indices[pos_i]
            
            # Create ranked score arrays for NDCG calculation
            current_scores = np.zeros_like(scores)
            new_scores = np.zeros_like(scores)
            
            # Fill in the scores in ranking order
            for idx, orig_idx in enumerate(current_indices):
                current_scores[idx] = scores[orig_idx]
            
            for idx, orig_idx in enumerate(new_indices):
                new_scores[idx] = scores[orig_idx]
            
            # Reshape for sklearn (expects 2D arrays)
            y_true = labels.reshape(1, -1)
            current_scores = current_scores.reshape(1, -1)
            new_scores = new_scores.reshape(1, -1)
            
            # Calculate NDCG for both rankings
            current_ndcg = ndcg_score(y_true, current_scores, k=self.k)
            new_ndcg = ndcg_score(y_true, new_scores, k=self.k)
            
            # Return absolute difference
            return abs(new_ndcg - current_ndcg)

You may also try an existing implementation by `Pytorch-DirectML <https://pypi.org/project/torch-directml/>`_ from Microsoft. For example,

.. code-block:: python
   :class: folding
   :name: LambdaRankLoss-DirectML

    from allrank.models.losses import lambdaLoss
    from allrank.data.dataset_loading import load_libsvm_dataset
    from allrank.models.model import make_model
    from allrank.training.train_utils import fit

    # Load dataset
    train_ds = load_libsvm_dataset("path/to/train.txt")
    val_ds = load_libsvm_dataset("path/to/val.txt")

    # Create model
    model = make_model(input_dim=train_ds.shape[1], 
                    hidden_dim=256, 
                    dropout=0.1)

    # Train with LambdaRank
    fit(model, 
        train_ds, 
        val_ds, 
        loss_function=lambdaLoss(weighing_scheme="ndcgLoss2_scheme"),
        lr=0.001,
        n_epochs=100)

`LightGBM <https://github.com/microsoft/LightGBM>`_ also has build-in support for LambdaRank. Following is a complete example using LambdaRank with LightGBM.

.. code-block:: python
   :class: folding
   :name: LambdaRankLoss-LightGBM-Complete-Example

    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.datasets import load_svmlight_file
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import ndcg_score

    # =====================================================
    # 1. Data Loading and Preparation
    # =====================================================

    def load_letor_data(file_path):
        """
        Load a LETOR-formatted dataset (SVMLight/LibSVM format with query IDs).
        
        LETOR format example:
        <relevance> qid:<qid> 1:<feature1> 2:<feature2> ... #comment
        2 qid:1 1:0.1 2:0.3 3:0.9 #docid=GX001-01
        
        Returns:
            X: features array
            y: relevance labels
            qids: query IDs for each document
            comment: comment strings if present
        """
        # Load the SVMLight formatted file
        # This format is common for learning-to-rank datasets (MSLR, Yahoo, LETOR)
        X, y, qids, comment = load_svmlight_file(file_path, query_id=True)
        
        # Convert sparse matrix to dense if needed
        X = X.toarray()
        
        # Convert relevance labels and qids to numpy arrays
        y = np.array(y, dtype=np.float32)
        qids = np.array(qids, dtype=np.int32)
        
        print(f"Loaded dataset with {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(qids))} unique queries")
        return X, y, qids, comment

    # Load the dataset (e.g., MSLR-WEB10K)
    # You can download it from: https://www.microsoft.com/en-us/research/project/mslr/
    X, y, qids, _ = load_letor_data("path/to/MSLR-WEB10K/Fold1/train.txt")

    # =====================================================
    # 2. Data Splitting
    # =====================================================

    # Split data while preserving query groups
    def split_data_by_queries(X, y, qids, test_size=0.2, random_state=42):
        """
        Split the data into train and test sets while keeping documents from
        the same query in the same set.
        """
        # Get unique query IDs
        unique_qids = np.unique(qids)
        
        # Split the query IDs (not the individual documents)
        train_qids, test_qids = train_test_split(
            unique_qids, test_size=test_size, random_state=random_state
        )
        
        # Create masks for train and test data
        train_mask = np.isin(qids, train_qids)
        test_mask = np.isin(qids, test_qids)
        
        # Apply masks to get train and test sets
        X_train, y_train, qids_train = X[train_mask], y[train_mask], qids[train_mask]
        X_test, y_test, qids_test = X[test_mask], y[test_mask], qids[test_mask]
        
        print(f"Train set: {X_train.shape[0]} samples, {len(np.unique(qids_train))} queries")
        print(f"Test set: {X_test.shape[0]} samples, {len(np.unique(qids_test))} queries")
        
        return X_train, X_test, y_train, y_test, qids_train, qids_test

    # Split data
    X_train, X_test, y_train, y_test, qids_train, qids_test = split_data_by_queries(X, y, qids)

    # =====================================================
    # 3. Preparing LightGBM Dataset
    # =====================================================

    # Create LightGBM datasets
    # The group parameter is crucial for learning-to-rank
    # It tells LightGBM which documents belong to which query

    def create_lgb_dataset(X, y, qids):
        """
        Create a LightGBM dataset with group information.
        
        Args:
            X: Feature matrix
            y: Target labels (relevance scores)
            qids: Query IDs for each document
            
        Returns:
            LightGBM Dataset with group information
        """
        # Count documents per query to create the group array
        # LightGBM needs to know how many documents are in each query group
        query_counts = []
        for qid in np.unique(qids):
            count = np.sum(qids == qid)
            query_counts.append(count)
        
        # Create LightGBM dataset with group information
        lgb_dataset = lgb.Dataset(
            data=X, 
            label=y,
            group=query_counts,  # This tells LightGBM which docs belong to which query
            free_raw_data=False  # Keep the raw data in memory
        )
        
        return lgb_dataset

    # Create training and validation datasets
    train_dataset = create_lgb_dataset(X_train, y_train, qids_train)
    test_dataset = create_lgb_dataset(X_test, y_test, qids_test)

    # =====================================================
    # 4. Configure LightGBM Parameters for LambdaRank
    # =====================================================

    # Set up parameters for learning-to-rank with LambdaRank
    # LightGBM supports multiple ranking objectives

    params = {
        # Specify that we're doing a ranking task
        'objective': 'lambdarank',
        
        # Optimization metric to use (NDCG@10)
        'metric': 'ndcg',
        
        # Focus on optimizing NDCG at position 10
        'ndcg_eval_at': [1, 3, 5, 10],
        
        # LambdaRank specific parameters
        'lambdarank_truncation_level': 10,  # Depth for computing NDCG in LambdaRank
        
        # Learning parameters
        'learning_rate': 0.1,
        'max_depth': 7,
        'num_leaves': 31,
        'min_data_in_leaf': 50,
        
        # Regularization
        'lambda_l1': 0.1,  # L1 regularization
        'lambda_l2': 0.1,  # L2 regularization
        
        # Other parameters
        'feature_fraction': 0.8,  # Use a subset of features per tree (prevents overfitting)
        'bagging_fraction': 0.8,  # Use a subset of data per tree
        'bagging_freq': 5,        # Perform bagging every 5 iterations
        
        # Verbosity
        'verbose': 1,
        
        # Force categorical features if any (in this example, we're assuming all are numerical)
        # 'categorical_feature': [0, 1]  # Uncomment if you have categorical features
    }

    # =====================================================
    # 5. Training the LambdaRank Model
    # =====================================================

    # Train the model
    num_rounds = 100  # Number of boosting iterations

    # LightGBM's native early stopping
    evals_result = {}  # Dictionary to store evaluation results

    print("Training LightGBM LambdaRank model...")
    model = lgb.train(
        params=params,
        train_set=train_dataset,
        num_boost_round=num_rounds,
        valid_sets=[train_dataset, test_dataset],
        valid_names=['train', 'test'],
        evals_result=evals_result,
        early_stopping_rounds=20,  # Stop if performance doesn't improve for 20 rounds
        verbose_eval=10  # Print evaluation every 10 iterations
    )

    # =====================================================
    # 6. Evaluation and Analysis
    # =====================================================

    # Extract scores by query for evaluation
    def get_scores_by_query(X, y, qids, model):
        """
        Get predictions grouped by query for proper ranking evaluation.
        
        Returns:
            Dictionary mapping each query ID to its true labels and predicted scores
        """
        # Get predictions for all documents
        y_pred = model.predict(X)
        
        # Group by query ID
        query_results = {}
        for qid in np.unique(qids):
            mask = qids == qid
            query_results[qid] = {
                'y_true': y[mask],
                'y_pred': y_pred[mask]
            }
        
        return query_results

    # Get predictions by query
    test_results = get_scores_by_query(X_test, y_test, qids_test, model)

    # Calculate NDCG for each query and average
    ndcg_scores = {k: [] for k in [1, 3, 5, 10]}

    for qid, result in test_results.items():
        true_labels = result['y_true'].reshape(1, -1)
        predicted_scores = result['y_pred'].reshape(1, -1)
        
        # Skip queries with only one relevance level (NDCG not meaningful)
        if len(np.unique(true_labels)) <= 1:
            continue
            
        # Calculate NDCG at different cutoffs
        for k in ndcg_scores.keys():
            if len(true_labels[0]) >= k:  # Only if we have enough documents
                score = ndcg_score(true_labels, predicted_scores, k=k)
                ndcg_scores[k].append(score)

    # Print average NDCG scores
    print("\nTest Set Evaluation:")
    for k, scores in ndcg_scores.items():
        if scores:  # Check if we have scores for this k
            avg_score = np.mean(scores)
            print(f"NDCG@{k}: {avg_score:.4f}")

    # =====================================================
    # 7. Feature Importance Analysis
    # =====================================================

    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    # Display top 20 features
    print("\nTop 20 important features:")
    print(importance_df.head(20))

    # =====================================================
    # 8. Model Saving and Loading
    # =====================================================

    # Save the model
    model.save_model('lightgbm_lambdarank_model.txt')
    print("\nModel saved to lightgbm_lambdarank_model.txt")

    # Load the model (if needed)
    loaded_model = lgb.Booster(model_file='lightgbm_lambdarank_model.txt')

    # =====================================================
    # 9. Applying the Model to New Data
    # =====================================================

    def predict_rankings(model, new_data, query_ids):
        """
        Apply the model to new data and return rankings for each query
        
        Args:
            model: Trained LightGBM model
            new_data: Features for new documents
            query_ids: Query IDs for the new documents
            
        Returns:
            Dictionary mapping each query ID to ranked document indices
        """
        # Get predictions
        predictions = model.predict(new_data)
        
        # Group by query
        rankings = {}
        for qid in np.unique(query_ids):
            # Get indices where query ID matches
            indices = np.where(query_ids == qid)[0]
            
            # Get predictions for this query
            query_preds = predictions[indices]
            
            # Sort indices by prediction score (descending)
            ranked_indices = indices[np.argsort(-query_preds)]
            
            # Store the ranked indices
            rankings[qid] = ranked_indices
        
        return rankings

    # Example of using the model on new data (using test data as an example)
    new_rankings = predict_rankings(model, X_test, qids_test)

    # Show example ranking for first query
    first_query = list(new_rankings.keys())[0]
    print(f"\nExample ranking for query {first_query}:")
    ranked_indices = new_rankings[first_query]
    for rank, idx in enumerate(ranked_indices[:10], 1):  # Show top 10
        print(f"Rank {rank}: Document idx={idx}, Score={model.predict([X_test[idx]])[0]:.4f}, True relevance={y_test[idx]}")

ListNet Loss
~~~~~~~~~~~~

:newconcept:`ListNet Loss` converts both target relevance score labels $y_i$ and predicted scores $s_i$ into probability distributions using the softmax function.

.. math::

    s_i = \frac{e^{s_i}}{\sum_{j} e^{s_j}}

    y_i = \frac{e^{y_i}}{\sum_{j} e^{y_j}}

Then the loss is a cross entropy.

.. math::

    \mathcal{L}_{\text{ListNet}} = -\text{mean}\left(y_i \log(s_i) \right)

ListNet converts the ranking label to a probability distribution, and the model learns to give higher scores to more relevant items to match the probability distribution.

ApproxNDCG Loss
~~~~~~~~~~~~~~~

:newconcept:`ApproxNDCG Loss` creates a differentiable approximation of the :refconcept:`Normalized Discounted Cumulative Gain` (NDCG) metric. It first tries to approximate an item's predicted rank by a differential function.

* When $s_jj > s_i$ (items ranked about item $i$), the logistic function approaches 1; when $s_j < s_i$ (items ranked below item $i$) the logistic function approaches 0. Therefore the sum of the logistic function is assumed to approximate the total number of other items above item $i$, which is the rank of item $i$.
* $\\alpha$ controls the sharpness of the sigmoid approximation.

.. math::

    \text{rank}_i \approx 1 + \sum_{j \neq i} \sigma(\alpha(s_j - s_i))


Then DCG is approximated as

.. math::

   \text{ApproxDCG} \approx \sum_{i} \frac{2^{y_i} - 1}{\log_2(1 + \text{rank}_i)}

where $y_i$ is the relevance score for item $i$ (treated as gain), and therefore the ApproxNDCG loss

.. math::

    \mathcal{L}_{\text{ApproxNDCG}} = 1 - \frac{\text{ApproxDCG}}{\text{IDCG}}

This loss directly optimizes a differentiable proxy for NDCG, providing a :ub:`more direct path to optimizing the actual evaluation metric`.

Contrastive Loss
----------------

:newconcept:`Contrastive Loss` functions are specifically designed to shape an embedding model's output space by pulling similar ("positive") examples closer while pushing dissimilar ("negative") examples apart. This is particularly beneficial for recall-stage models in search, recommendation, or advertising systems, where the objective is to efficiently retrieve potentially relevant items from vast candidate pools. These loss functions focus explicitly on optimizing embedding spaces rather than direct ranking.

In recall-stage models, interactions surpassing a pre-defined engagement milestone are typically classified as positive examples (:ub:`lower bar than the precision stage`). For instance, in music recommendation, songs played beyond 30 seconds or interactions exceeding other specified thresholds might constitute positives, though thresholds may vary based on platform and content type.

Negative Examples
~~~~~~~~~~~~~~~~~

:newconcept:`Negative Examples` are vital in developing effective recall-stage models. Given the sparse nature of positive user-item interactions, negative sampling :ub:`addresses data imbalance` by selecting a representative subset of non-interacted (negative) items to enhance model training.

The primary objectives of embedding training with negative examples include:

* Creating query and item embeddings optimized for efficient similarity search, ensuring queries are close to relevant items and distant from irrelevant ones.
* Enabling scalable approximate nearest neighbor retrieval techniques such as FAISS, HNSW, or ScaNN, facilitating handling extremely large item catalogs (potentially billions of items).

However, accurately determining negative examples poses significant challenges due to the inherent sparsity of positive interactions. The key question is how to identify reliable negative samples effectively. Non-interacted items are not always genuinely negative; they may simply be unknown or unexposed to the user. Mislabeling potential positives as negatives adversely affects recall performance. To improve negative sample quality, the following strategies are recommended:

1. [Include] Only items with explicit negative user feedback (e.g., thumbs-down, low ratings, quick abandonment).
2. [Include] Only items previously ranked higher and shown to the user before their lowest positively interacted item. A stricter approach could require repeated occurrences of these items.
3. [Exclude] Items exceeding certain engagement thresholds (e.g., viewed for 30+ seconds).
4. [Exclude] Items within a certain affinity radius (e.g., within 3 hops) on the user-item interaction graph, as these may indicate latent user interest.

Common negative sampling techniques include:

* **Random Sampling**: Negatives randomly selected from the entire item catalog.
* **Popularity-Based Sampling**: Negatives sampled proportionally to item popularity, with caps recommended to prevent oversampling highly popular items.
* **Hard Negative Mining**: Strategically selecting negatives that the current model incorrectly classifies as potential positives.
* [Meta Technique] **Stratified Sampling**: Stratify negative samples across different categories, genres, or other raw features. Ensure representation from different item types proportional to their distribution.
* [Meta Technique] **Bootstrapping**: Build an initial model with simple random sampling and then iteratively add more negative samples on top. Assume one positive example targets to pair with 50 negative examples in every round of training.

  * Begin with simple random sampling (e.g., starting with 10 negative examples) and build an initial model.
  * Extract embeddings from the initial models (e.g., 100 embeddings). Perform clustering algorithms (e.g., k-means) to obtain 10 new negative examples. Add new negative examples to the training data.
  * Continuous the iterations until 50 negative examples are sampled for each positive example. After each iteration, the new negative examples would bias more to hard negatives.
  * The sampled negative examples are representative, consisting of initial random sample, and then gradually easier negatives to harder negatives.

While these approaches improve confidence in negative sample selection, they also introduce potential biases and reduce representativeness. A recommended solution to mitigate these biases is incorporating :refconcept:`preference learning` as an additional multi-task objective. Preference learning involves modeling user preferences between item pairs, as discussed by `Pairwise Preference Loss`_. This technique :ub:`leverages richer training data` can help :ub:`promote a nuanced, continuous representation` of user preferences while the model still learns to maintain hard binary distinction among the explicit pairs of positive/negative examples.

.. admonition:: **Why Negative Examples Are Less Prominent in Precision Layers**
   :class: note

   While negative examples are crucial for recall-stage models, they play a diminished role in precision-stage models for several reasons:

   1. **Different Optimization Goals**: The precision layer focuses on fine-grained ranking of candidates already filtered by the recall stage, rather than distinguishing relevant from irrelevant items.
   2. **Candidate Set Characteristics**: Precision layers operate on a much smaller set of candidates (dozens to hundreds) that have already passed basic relevance filters, eliminating most obvious negatives.
   3. **Ambiguity in Negative Labels**: Within the filtered candidate set, the distinction between "not interesting" and "not seen yet" becomes more nuanced and critical.
   4. **Architectural Differences**: Precision models often employ cross-attention mechanisms between query and item features, enabling richer interaction modeling without relying heavily on contrastive learning.
   5. **Rich Feature Utilization**: Precision layers leverage complex, computation-intensive features that wouldn't be practical to compute for the recall stage, reducing reliance on explicit negatives.

   Instead of leveraging contrastive losses that heavily rely on negative examples, precision-stage models often benefit more from cross-item attention mechanisms, listwise ranking losses and other specialized prediction heads.

Pairwise Contrastive Loss
~~~~~~~~~~~~~~~~~~~~~~~~~

Assume :math:`\mathbf{q} = h(\mathbf{u}, \mathbf{Q})` obtains an integrated query embedding, :newconcept:`Pairwise Contrastive Loss` pushes the query toward relevant items, and away from irrelevant items in the embedding space.

.. math::

      \mathcal{L}_{\text{contrastive}} = \frac{1}{2} \sum_{i} y_{i} \text{d}^2(\mathbf{q}, \mathbf{e}_i) + (1 - y_{i}) \max(0, \text{margin} - \text{d}^2(\mathbf{q}, \mathbf{e}_i))

where

* :math:`\mathbf{q}` is the query embedding derived from user features :math:`\mathbf{u}` and context features :math:`\mathbf{Q}`
* :math:`\mathbf{e}_i` is the embedding of item i
* :math:`y_{i} = 1` if item i is relevant to the query (positive example), 0 otherwise (negative example)
* :math:`\text{d}(\mathbf{q}, \mathbf{e}_i)` is a distance function between embeddings (typically Euclidean or cosine distance)
* The margin parameter enforces a minimum distance between query and irrelevant items

The Contrastive Loss can be understood as having two components that work together:

1. For relevant items (:math:`y_{i} = 1`): The term :math:`\text{d}^2(\mathbf{q}, \mathbf{e}_i)` penalizes distance between query and item embeddings, pulling them closer together.
2. For irrelevant items (:math:`y_{i} = 0`): The term :math:`\max(0, \text{margin} - \text{d}^2(\mathbf{q}, \mathbf{e}_i))` penalizes embeddings that are closer than the margin (similar to `Hinge Loss`_).

   * When :math:`\text{margin} \leq \text{d}(\mathbf{q}, \mathbf{e}_i)`, the loss is zero as the constraint is satisfied.
   * When :math:`\text{margin} >  \text{d}(\mathbf{q}, \mathbf{e}_i)`, the loss is positive, pushing embeddings apart.


Triplet Loss
~~~~~~~~~~~~

:newconcept:`Triplet Loss` works with triplets consisting of a query, a relevant (positive) item, and an irrelevant (negative) item:

.. math::

   \mathcal{L}_{\text{triplet}} = \max(0, \text{margin} - (d(\mathbf{q}), \mathbf{e}_n) -  (d(\mathbf{q}, \mathbf{e}_p)))

where

* :math:`\mathbf{q}` is the query embedding derived from user features :math:`\mathbf{u}` and context features :math:`\mathbf{Q}`
* :math:`\mathbf{e}_p` is the embedding of a relevant (positive) item
* :math:`\mathbf{e}_n` is the embedding of an irrelevant (negative) item
* :math:`d(\cdot,\cdot)` is a distance function (typically Euclidean or cosine distance)
* The margin enforces a minimum difference between distances of query-positive and query-negative pairs

The formula can be interpreted as:

* When :math:`\text{margin} - (d(\mathbf{q}, \mathbf{e}_p) - d(\mathbf{q}, \mathbf{e}_n)) \leq 0`, the loss is zero, meaning the model has successfully learned that the query is closer to the relevant item than to the irrelevant item by at least the margin amount
* When :math:`\text{margin} - (d(\mathbf{q}, \mathbf{e}_p) - d(\mathbf{q}, \mathbf{e}_n)) > 0`, the model is penalized proportionally to how much the constraint is violated

.. admonition:: Relationship to Hinge Loss
   :class: note

   Triplet Loss is a direct extension of :refconcept:`Hinge Loss` to the similarity learning domain, both using the $\\max(0, \\cdot)$ operator:

   * **Standard Hinge Loss for binary classification**: :math:`\max(0, \text{margin} - y \hat{y})` to mask out loss for well-enough predictions, and hence encourage incorrect or not-good-enough :math:`\hat{y}` estimations move toward $y$.
   * **Triplet Loss for enforcing a margin between relative distances in the embedding space**: :math:`\max(0, \text{margin} - (d(\mathbf{q}, \mathbf{e}_n) - d(\mathbf{q}, \mathbf{e}_p)))` to mask out loss well-apart positive/negative examples, and hence place more focus on moving remaining positive/negative examples apart.

Batch Contrastive Loss
~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Batch Contrastive Loss` extends contrastive learning principles to entire batches, utilizing the positive example and multiple negative examples simultaneously to improve training efficiency and stability, and thus better embedding quality. :newconcept:`InfoNCE Loss` is widely used and formulated as the following

.. math::

   \mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{s_{i,i+}/\tau}}{\sum_{j=1}^{B} e^{s_{i,j}/\tau}}

where

* :math:`B` is the batch size and :math:`s_{i,j}` is the similarity score between user i and item j.
* :math:`\tau` is a temperature parameter that controls the sharpness of the distribution.

:newconcept:`Batch Softmax Loss` is is a special case of InfoNCE Loss when the temperature parameter is $1$.

.. math::

   \mathcal{L}_{\text{batch-softmax}} = \mathcal{L}_{\text{InfoNCE}}|_{\tau = 1}

These batch contrastive methods offer key advantages:

* :ub:`Efficient computation`: Leverages batch-level parallelism on modern hardware, significantly accelerating training.
* :ub:`Enhanced representation quality`: Promotes clear embedding differentiation by jointly considering multiple negative examples together for contrasts.


Relating Business Metrics To Loss
---------------------------------

When designing recommendation/search/ads systems, bridging the gap between business metrics and model optimization objectives is a fundamental challenge. While businesses measure success through metrics like revenue, retention, and user satisfaction, machine learning models are trained through technical loss functions like cross-entropy or mean squared error. This disconnection between what we optimize for (loss functions) and what we actually care about (business metrics) represents one of the central challenges in applied machine learning for search/recommendation/ads systems. This section explores strategies for translating business KPIs into machine learning objectives.

Common Business Metrics
~~~~~~~~~~~~~~~~~~~~~~~

The following table provides an overview of common business metrics used in search, recommendation, and ads systems.

* **Engagement & Interaction Metrics** reflecting service quality (item level or user leve)

  * Click-Through Rate (CTR)
  * Bounce Rate
  * Session Duration
  * First-Page Keyword Rankings
  * Diversity
  * Serendipity (Novelty)
  * Engagement Score
  * User Satisfaction Score
  * Task Completion Rate
  * Daily/Monthly Active Users (DAU/MAU)

* **Conversion and Monetization Metrics** reflecting the business prosperity and profitability.

  * Conversion Rate (CVR)
  * Click-to-Conversion Rate
  * Revenue Per Mille (RPM)
  * Customer Lifetime Value (CLV)
  * Return on Ad Spend (ROAS)
  * Return on Marketing Investment (ROMI)

* **Cost & Efficiency Metrics** reflecting the business cost and efficiency.

  * Cost Per Click (CPC)
  * Cost Per Mille (CPM)
  * Cost Per Lead (CPL)
  * Cost Per Acquisition (CPA)
  * Customer Acquisition Cost (CAC)

* **Retention & Loyalty Metrics** reflecting user's long-term satisfaction with the service.

  * Retention Rate
  * N-Day Retention
  * Churn Rate
  * Customer Lifetime Value (CLV)

* **Analysis Metrics**

  * Impressions
  * Exploration Rate
  * Attribution Metrics


.. list-table:: Common Business Metrics in Search, Recommendation, and Advertising Systems
   :header-rows: 1
   :widths: 20 50 10 10 10

   * - Metric
     - Description
     - Search
     - Recommendation
     - Ads
   * - Click-Through Rate (CTR)
     - Percentage of impressions that result in clicks. Formula: CTR = (Clicks / Impressions) × 100%
     - ✓
     - ✓
     - ✓
   * - Conversion Rate (CVR)
     - Percentage of users who complete a desired action (e.g., purchase, signup). Formula: CVR = (Conversions / Total Users) × 100%
     - ✓
     - ✓
     - ✓
   * - Click-to-Conversion Rate
     - Percentage of clicks that result in a conversion. Formula: (Conversions / Clicks) × 100%
     -
     - ✓
     - ✓
   * - Bounce Rate
     - Percentage of sessions where users leave without taking further action. Formula: Bounce Rate = (Single-Page Sessions / Total Sessions) × 100%
     - ✓
     - ✓
     - ✓
   * - Session Duration
     - Average time users spend per session. Formula: Total Duration of All Sessions / Number of Sessions
     - ✓
     - ✓
     -
   * - Diversity
     - Variety of items recommended to users. For example, averaging pairwise similarity of recommended items
     -
     - ✓
     -
   * - Serendipity (Novelty)
     - Measure of how "surprising" yet relevant recommendations are. For example, percentage of recommended items from categories the user hasn't previously interacted with but finds relevant
     -
     - ✓
     -
   * - First-Page Keyword Rankings
     - Track the ranking positions of targeted keywords on the first page of search engine results, reflecting SEO performance
     - ✓
     -
     -
   * - Task Completion Rate
     - Percentage of users who successfully complete their intended task
     - ✓
     - ✓
     -
   * - User Satisfaction Score
     - Direct feedback from users about their experience
     - ✓
     - ✓
     - ✓
   * - Retention Rate
     - Percentage of users who return after their first visit within a specific timeframe
     -
     - ✓
     -
   * - N-Day Retention
     - Percentage of users who return on the nth day after their first visit
     -
     - ✓
     -
   * - Churn Rate
     - Percentage of users who stop using a service over a specific period. Formula: Churn Rate = (Number of Users Lost during Period / Total Users at Start of Period) × 100%
     - ✓
     - ✓
     -
   * - Revenue Per Mille (RPM)
     - Revenue generated per 1,000 impressions. Formula: RPM = (Total Revenue / Total Impressions) × 1,000
     - ✓
     - ✓
     - ✓
   * - Cost Per Click (CPC)
     - Average cost paid for each click. Formula: CPC = Total Cost / Number of Clicks
     -
     -
     - ✓
   * - Cost Per Mille (CPM)
     - Cost per 1,000 ad impressions. Formula: CPM = (Total Cost / Impressions) × 1,000
     -
     -
     - ✓
   * - Cost Per Lead (CPL)
     - Cost effectiveness of generating new leads. In the context of ads, a "lead" refers to a potential customer who has shown some interest in a product or service through a qualifying action. Formula: CPL = Total Marketing Spend / Number of New Leads
     -
     -
     - ✓
   * - Cost Per Acquisition (CPA)
     - Average cost incurred to acquire a customer through an ad campaign. Formula: CPA = Total Advertising Cost / Number of Acquisitions
     -
     -
     - ✓
   * - Customer Acquisition Cost (CAC)
     - Total cost of acquiring a new customer, including marketing and sales expenses. Formula: CAC = Total Marketing and Sales Expenses / Number of New Customers
     -
     - ✓
     - ✓
   * - Return on Ad Spend (ROAS)
     - Revenue generated relative to advertising costs. Formula: ROAS = Revenue from Advertising / Cost of Advertising
     -
     -
     - ✓
   * - Return on Marketing Investment (ROMI)
     - Revenue generated for each dollar spent on marketing. Formula: ROMI = (Sales Growth - Marketing Cost) / Marketing Investment × 100%
     -
     - ✓
     - ✓
   * - Customer Lifetime Value (CLV)
     - Total revenue expected from a customer throughout their relationship with the platform
     -
     - ✓
     - ✓
   * - Daily/Monthly Active Users (DAU/MAU)
     - Number of unique users who engage with the platform daily or monthly
     - ✓
     - ✓
     -
   * - Engagement Score
     - Indicating level of user interaction with content, such as likes, shares, mentions, comments, or time spent on a page
     - ✓
     - ✓
     - ✓
   * - Impressions
     - Number of times an ad is displayed, regardless of user interaction
     -
     - ✓
     - ✓
   * - Exploration Rate
     - Measure of how often user choose to interact with results offered by the system for the purpose of exploration.
     - ✓
     - ✓
     - ✓
   * - Attribution Metrics
     - Determine the contribution of each marketing channel or touchpoint in driving conversions, aiding in budget allocation and strategy optimization
     - ✓
     - ✓
     - ✓

Proxy Metrics
~~~~~~~~~~~~~

Some business metrics (like revenue, retention, or lifetime value) cannot be directly optimized through standard loss functions for several key reasons:

* **Delayed Feedback**: Business metrics often materialize days or weeks after model predictions (e.g., earning for a conversion, lifetime value, retention).
* **Signal Sparsity**: Business events like purchases or subscriptions are rare compared to clicks or impressions.
* **Noise and Variability**: Business metrics fluctuate due to external factors beyond the model's control (seasonality, market conditions).
* **Attribution Challenges**: Difficult to attribute business outcomes to specific model decisions.
* **Normalization Issues**: Raw business metrics vary widely in scale and are not normalized.

Therefore, effective model training typically relies on carefully selected proxy metrics that:

1. Provide :ub:`immediate` feedback for optimization.
2. :ub:`Correlate` strongly with business outcomes.
3. Offer :ub:`consistent` and stable signals for modeling.
4. Are :ub:`normalized and comparable` across contexts.

.. list-table:: Common Business Metrics and Their Loss Function Mappings
   :header-rows: 1
   :widths: 20 30 30 20

   * - Business Metric
     - Proxy Metric
     - Loss Function
     - Prediction Head Type
   * - Revenue/ROAS
     - Click-Through Rate, Conversion Rate
     - Binary Cross-Entropy, Weighted BCE
     - Classification
   * - User Retention
     - Session Time, Revisit Rate
     - MSE, Huber Loss, Quantile Regression
     - Regression
   * - Content Discovery
     - Click Diversity, Exploration Rate
     - ListNet, ApproxNDCG
     - Ranking
   * - User Satisfaction
     - Explicit Ratings, Session Completion
     - Ordinal Cross-Entropy
     - Ordinal Classification
   * - Engagement
     - Scroll Depth, Playback Time, Synthetic Engagement Score
     - MSE, Huber Loss
     - Regression
   * - Conversion Funnel
     - Session Milestone Completion
     - All-Threshold Loss
     - Ordinal Classification

Some business metrics like Revenue/Return Per Mile (RPM) are accumulated smoothed across hundreds to thousands of impressions/interactions, and hence can possibly be one optimization objective, especially if delayed reward is tolerated.

Promoting Diversity
~~~~~~~~~~~~~~~~~~~

promoting diversity is one business focus for search/recommendation/ads system, as user does not want to be

although adding a diversity penality term to loss to encourage diversity is one way, for example, if the business want items after position 5 be more diverse,  you can add a sum of intra embeddings similarities to the loss to encourage a higher inter embedding similarities before the output layer caculation. this would eliminate one benefict of approxndcg, need need for reranking. however, this it is not the full picture.

in practice, diversity is promoted through multiple practice, including session based data (prevent user search twice and get the same results), and exploration experiments, ...

