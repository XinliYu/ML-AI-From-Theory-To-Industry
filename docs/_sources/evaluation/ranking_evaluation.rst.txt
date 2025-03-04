Ranking Evaluation
==================

Many ML/AI systems must rank items in order of predicted relevance, such as search engines and recommendation systems. In this section, we assume that given a **query**, the ML/AI system returns an ordered list of items.

Unlike precision and recall which only measure how well the ML/AI system makes binary decisions about returning relevant items, :newconcept:`Ranking Evaluation` focuses on assessing how well a system orders items. The common approach is to assign higher weights to items appearing higher in the ranked results during metric calculation. Ranking evaluation is particularly valuable in applications where the position of an item in results directly impacts user experience.

Precision@k & Recall@k
----------------------

:newconcept:`Precision@k` (P@k) and :newconcept:`Recall@k` (R@k) are basic metrics for ranking evaluation. Given a query, they adapt the traditional :refconcept:`precision` and :refconcept:`recall` metrics to evaluate only the top-k ranked items:

.. math::

   \text{Precision@k} = \frac{\text{Number of relevant items in top-k results}}{k}

.. math::

   \text{Recall@k} = \frac{\text{Number of relevant items in top-k results}}{\text{Total number of relevant items}}

These metrics are useful for evaluating systems where users typically only examine a limited number of top results (e.g., first page of search results). To evaluation the ML/AI system ranking, we can avearge P@k and R@k across all queries in the ground truth.

Mean Average Precision (MAP)
----------------------------

Given a query, :newconcept:`Average Precision (AP)` averages P@k and R@k metrics across a range of $k$ from 1 to a chosen maximum rank number $n$ when the item at position $k$ is relevant.

.. math::
    AP(q) = \frac{\sum_{k=1}^{n} P(k) \cdot \text{rel}(k)}{\sum_{k=1}^{n} \text{rel}(k)}

Where:

* $q$ is a query.
* $P(k)$ is P@k for query $q$.
* $\\text{rel}(k)$ is the relevance score for the item ranked at $k$, and is typically an indicator function for MAP,
  
  .. math::
     \text{rel}(k) = \mathbf{1}[{\text{item at position $k$ is relevant}}] = 
     \begin{cases} 
     1 & \text{if item at position $k$ is relevant} \\
     0 & \text{otherwise}
     \end{cases}

* Given the formula, item at higher rank has more weight in AP calculation. For example, item at rank $1$ will be considered across $k=1, ..., n$, while the item at rank $n$ is only considered once.

.. note:: Why using indicator $\\text{rel}(k)$ to skip irrelevant items?

    AP's formula skips items that are not relevant, through applying the indicator $\\text{rel}$, for the following reasons.

    * **Intuitive interpretation**: The overall AP value represents the average precision a user would experience when finding each relevant document in the ranked list. This is more meaningful than averaging precision at every position regardless of relevance.
    * **Historical development**: The formula evolved from the :refconcept:`PR Curve` where precision is plotted against recall. By averaging precision at each recall point (which occurs exactly when a new relevant document is found), AP approximates the :refconcept:`area under curve`.

.. note:: Can $\\text{rel}(k)$ be a non-indicator function?

    The relevance metric in Mean Average Precision (MAP) doesn't necessarily have to be an indicator function, but using a binary indicator function ($\\text{rel}(k) \\in \\{0,1\\}$) has become the standard approach. MAP was originally developed for information retrieval scenarios with binary relevance judgments (relevant/not relevant). The indicator function directly captures this binary nature. With binary relevance, AP has a clear interpretation: "the average precision a user experiences when finding each relevant document.", and has an AUC based interpretation. 
    
    There have been extensions of MAP that incorporate non-indicator relevance scores rather than just binary judgments to allow higher weights to more relevant items. The extended version is typically considered a separate metric with its own names (i.e., :newconcept:`Graded Average Precision (GAP)`) rather than standard MAP.
  
.. admonition:: AP's Sensitivity to Ranking Order
   :class: example-green

   Consider two different rankings for the same query, each with 3 relevant items (R) and 2 irrelevant items (N):

   **Ranking A:** [R, R, R, N, N]
   
   Precision calculations:

   - P@1 = 1/1 = 1.0 (rel=1)
   - P@2 = 2/2 = 1.0 (rel=1)
   - P@3 = 3/3 = 1.0 (rel=1)
   - P@4 = 3/4 = 0.75 (rel=0)
   - P@5 = 3/5 = 0.6 (rel=0)
   
   AP = (1.0×1 + 1.0×1 + 1.0×1 + 0.75×0 + 0.6×0)/3 = 3.0/3 = 1.0

   **Ranking B:** [R, N, R, N, R]
   
   Precision calculations:

   - P@1 = 1/1 = 1.0 (rel=1)
   - P@2 = 1/2 = 0.5 (rel=0)
   - P@3 = 2/3 = 0.67 (rel=1)
   - P@4 = 2/4 = 0.5 (rel=0)
   - P@5 = 3/5 = 0.6 (rel=1)
   
   AP = (1.0×1 + 0.5×0 + 0.67×1 + 0.5×0 + 0.6×1)/3 = 2.27/3 = 0.76

   Despite having the same relevant documents, Ranking B has a lower AP (0.76 vs 1.0) because the relevant documents are ranked lower.

:newconcept:`Mean Average Precision (MAP)` is simply the avearge of AP across all queries. 

.. math::
    MAP = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}(q)


MAP rewards methods that place relevant documents higher in the ranking and is particularly useful for comparing different ranking algorithms.

.. admonition:: Example: Step-by-Step MAP Calculation
   :class: example-green

   Let's work through a complete example to illustrate MAP calculation with multiple queries.

   Consider a search engine evaluation with three queries: "machine learning frameworks", "neural networks", and "data visualization". For each query, we have an ordered list of search results with relevance judgments (R = relevant, N = not relevant).

   **Query 1: "machine learning frameworks"**
   
   Top 10 ranked results:

   1. TensorFlow (R)
   2. PyTorch (R)
   3. Weather forecast (N)
   4. Scikit-learn (R)
   5. Keras (R)
   6. Random news article (N)
   7. Theano (R)
   8. E-commerce site (N)
   9. Caffe (R)
   10. Restaurant review (N)

   Total relevant items for this query: 6
   Relevant items are at positions: 1, 2, 4, 5, 7, 9

   Let's calculate precision at each relevant position:

   - P@1 = 1/1 = 1.000 (rel=1)
   - P@2 = 2/2 = 1.000 (rel=1)
   - P@4 = 3/4 = 0.750 (rel=1)
   - P@5 = 4/5 = 0.800 (rel=1)
   - P@7 = 5/7 = 0.714 (rel=1)
   - P@9 = 6/9 = 0.667 (rel=1)

   AP for Query 1 = (1.000 + 1.000 + 0.750 + 0.800 + 0.714 + 0.667) / 6 = 4.931 / 6 = 0.822

   **Query 2: "neural networks"**
   
   Top 8 ranked results:

   1. Deep learning article (R)
   2. Sports news (N)
   3. Convolutional networks paper (R)
   4. Online shop (N)
   5. Recurrent networks tutorial (R)
   6. Biography of a celebrity (N)
   7. Tech blog post (N)
   8. Backpropagation explanation (R)

   Total relevant items for this query: 4
   Relevant items are at positions: 1, 3, 5, 8

   Let's calculate precision at each relevant position:

   - P@1 = 1/1 = 1.000 (rel=1)
   - P@3 = 2/3 = 0.667 (rel=1)
   - P@5 = 3/5 = 0.600 (rel=1)
   - P@8 = 4/8 = 0.500 (rel=1)

   AP for Query 2 = (1.000 + 0.667 + 0.600 + 0.500) / 4 = 2.767 / 4 = 0.692

   **Query 3: "data visualization"**
   
   Top 6 ranked results:

   1. Movie review (N)
   2. Matplotlib tutorial (R)
   3. D3.js gallery (R)
   4. Social media post (N)
   5. Tableau guide (R)
   6. Visualization best practices (R)

   Total relevant items for this query: 4
   Relevant items are at positions: 2, 3, 5, 6

   Let's calculate precision at each relevant position:

   - P@2 = 1/2 = 0.500 (rel=1)
   - P@3 = 2/3 = 0.667 (rel=1)
   - P@5 = 3/5 = 0.600 (rel=1)
   - P@6 = 4/6 = 0.667 (rel=1)

   AP for Query 3 = (0.500 + 0.667 + 0.600 + 0.667) / 4 = 2.434 / 4 = 0.609

   **MAP Calculation**

   MAP = (AP_Query1 + AP_Query2 + AP_Query3) / 3
   MAP = (0.822 + 0.692 + 0.609) / 3 = 2.123 / 3 = 0.708

   **Analysis**

   The MAP score of 0.708 indicates good overall ranking performance across the three queries. Breaking down the results:

   * **Query 1 (AP = 0.822)**: Best performance, with relevant items clustered near the top and good precision throughout.
   * **Query 2 (AP = 0.692)**: Good performance but with some relevant items appearing lower in the ranking.
   * **Query 3 (AP = 0.609)**: Weakest performance, starting with an irrelevant result and having more inconsistent precision.

   This example illustrates how MAP rewards systems that rank relevant items higher while penalizing those that place irrelevant items at top positions. The use of macro-averaging gives equal weight to each query regardless of how many relevant items it contains, ensuring that performance on all queries contributes equally to the final metric.

MAP has its counterpart for recall. :newconcept:`Mean Average Recall (MAR)` is the metric for evaluating ranking performance but focusing on recall. While MAP emphasizes precision, MAR measures how well the system retrieves all relevant items across different ranks. MAR also shares a similar interpretation to MAP - related to the Recall-Precision curve and the AUC-RP metric.

.. math::
    AR(q) = \frac{\sum_{k=1}^{n} R(k) \cdot \text{rel}(k)}{\sum_{k=1}^{n} \text{rel}(k)}

Where:

* $q$ is a query.
* $R(k)$ is R@k for query $q$.
* $\\text{rel}(k)$ is the relevance score for the item ranked at $k$, typically an indicator function as in MAP.

:newconcept:`Mean Average Recall (MAR)` is then calculated as the average of AR across all queries:

.. math::
    MAR = \frac{1}{|Q|} \sum_{q \in Q} \text{AR}(q)


Normalized Discounted Cumulative Gain (NDCG)
--------------------------------------------

Zipf's Law
~~~~~~~~~~

:newconcept:`Zipf's Law` (`Wiki <https://en.wikipedia.org/wiki/Zipf%27s_law>`_) is an empirical law that states that when a list of items are sorted in decreasing order by certain value (e.g., words sorted by frequency in descending order), the value of the $i$-th ranked item from the ordered list is often approximately inversely proportional to $i$:

.. math::
    f(i) \propto \frac{1}{(i+p)^s}

where $p$ and $s$ are two parameters.

Zipf's Law is highly relevant to information retrieval, recommendation systems, and ranking evaluation. Research has shown that user attention to search results follows a similar distribution, with dramatically less attention given to items as their rank increases. While Zipf's Law suggests a power-law decline ($\\frac{1}{(i+p)^s}$), many ranking metrics use logarithmic discounting as a more balanced approximation of this attention drop-off. The :newconcept:`discount factor` used in DCG and NDCG draws inspiration from this relationship between item position and user attention.

DCG
~~~

:newconcept:`Normalized Discounted Cumulative Gain (NDCG)` is a widely used ranking metric that considers the position of relevant documents. It is based on :newconcept:`Discounted Cumulative Gain (DCG)`, which assigns higher importance to relevant documents appearing earlier in the ranked list. NDCG and DCG are typically calculated for the top-k items, and are noted as **NDCG@k** and **DCG@k**.

DCG is calculated for the top-k of the ranked list as:

.. math::
    \text{DCG}@k = \sum_{i=1}^{k} \frac{g(i)}{d(i)}

Where:

* $g(i)$ is the :newconcept:`gain function` of the item at position $i$ in the ranked list
* $d(i)$ is the :newconcept:`discount factor` that reduces the contribution of items at lower positions

In the canonical formulation, the gain function is simply the relevance score of the item $g(i) = \\text{rel}(i)$, and the discount factor is logarithmic $d(i) = \\log_2(i+1)$, giving us:

.. math::
    \text{DCG}@k = \sum_{i=1}^{k} \frac{\text{rel}(i)}{\log_2(i+1)}

.. note:: About Discount Factor $d(i)$

    The **discount factor** in Discounted Cumulative Gain (DCG) can use logarithms with bases other than 2. The choice of denominator affects how quickly relevance is discounted as position increases.
    * Higher base (e.g., the natural logarithm with base $e=2.71828$) create a more gradual discount, placing more importance on items deeper in the results.
    * Smaller base (e.g., 1.5) creates a steeper discount, severely penalizing lower positions.
    
    The discount factor need not necessarily be a logarithm. It can be **linear discount**, **exponential discount**, or **power discount** (as in the original Zipf's Law), etc.

NDCG
~~~~

DCG scores have interpretability issues that make it challenging to use on its own.

* **Query-specific interpretation**: A DCG of 10 might be excellent for a query with few relevant documents but poor for a query with many highly relevant documents.
* **Scale & upperbound issue**: Unlike metrics that are normalized between 0 and 1, DCG can grow unbounded depending on the number of relevant documents.

To address these limitations, NDCG normalizes DCG by the maximum possible DCG for that query:

The :newconcept:`Ideal DCG (IDCG)` is calculated by sorting all relevant items by their relevance scores (highest to lowest) and computing the DCG of this ideal ordering:

.. math::

   \text{IDCG}@k = \sum_{i=1}^{k} \frac{\text{rel}^*(i)}{\log_2(i+1)}

Where $\\text{rel}^*(i)$ is the relevance score at position $i$ in the ideally ranked list.

NDCG is then defined as:

.. math::

   \text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}

.. admonition:: Example: NDCG Calculation
   :class: example-green

   Consider a movie recommendation system that ranks films on a relevance scale:

   - 3: Highly relevant
   - 2: Relevant
   - 1: Somewhat relevant
   - 0: Irrelevant

   **User Query: "Sci-fi action movies"**
   
   System ranking with relevance scores:

   1. Star Wars: Episode V (rel=3)
   2. Blade Runner (rel=3)
   3. Romantic comedy (rel=0)
   4. The Matrix (rel=3)
   5. Avatar (rel=2)

   DCG@5 calculation:

   - DCG@5 = 3/log₂(1+1) + 3/log₂(2+1) + 0/log₂(3+1) + 3/log₂(4+1) + 2/log₂(5+1)
   - DCG@5 = 3/1 + 3/1.585 + 0/2 + 3/2.322 + 2/2.585
   - DCG@5 = 3 + 1.892 + 0 + 1.292 + 0.774 = 6.958

   The ideal ranking would place all highly relevant (3) items first, followed by relevant (2) items:

   1. Star Wars: Episode V (rel=3)
   2. Blade Runner (rel=3)
   3. The Matrix (rel=3)
   4. Avatar (rel=2)
   5. Romantic comedy (rel=0)

   IDCG@5 calculation:

   - IDCG@5 = 3/log₂(1+1) + 3/log₂(2+1) + 3/log₂(3+1) + 2/log₂(4+1) + 0/log₂(5+1)
   - IDCG@5 = 3/1 + 3/1.585 + 3/2 + 2/2.322 + 0/2.585
   - IDCG@5 = 3 + 1.892 + 1.5 + 0.861 + 0 = 7.253

   NDCG@5 = DCG@5/IDCG@5 = 6.958/7.253 = 0.959

   This high NDCG score of 0.959 indicates that despite ranking one irrelevant item at position 3, the system still performs very well overall, capturing most of the ideal ordering.

Ranking Metrics For Early-Relevance & Low-Precision Ranking Scenarios
---------------------------------------------------------------------

Some ranking applications inherently involve a low-precision scenario, which demands metrics that specifically evaluate a system's ability to present relevant content early in the ranking order.

1. Only one or very few items in a large candidate pool are relevant
2. The primary goal is to surface at least one relevant item within top positions

The following are some concrete scenarios.

  * **Voice assistants**: For smart devices like Alexa or Google Home, the screen may be small and low resolution, showing only three items at a time. On headless devices, voice recommendations can typically only present one item before user patience is exhausted.
  * **Sponsored Ads**: On search engines or social media platforms, there are typically only 2-3 available spots for sponsored results. Advertisers pay premiums for these positions, making the correct ranking critical for monetization.
  * **Mobile search**: On mobile screens, often only the top a few results appear without scrolling. Users frequently select from just these top results, rarely scrolling past the first few items.
  * **Fact-based queries**: For questions with a single correct answer (like "Who is the president of France?"), users primarily care about finding that one correct answer quickly rather than seeing multiple relevant documents.
  * **Featured snippets**: Search engines often display a single "featured snippet" at the top of results for certain queries. The system must correctly identify the most relevant result to feature in this high-visibility position.
  * **Autocomplete suggestions**: Search bars typically show 4-5 autocomplete suggestions, with users commonly selecting from only the top 1-2 options.

Mean Reciprocal Rank (MRR)
~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Mean Reciprocal Rank (MRR)` evaluates ranking quality by measuring how soon the first relevant document appears in the ranked list. It is calculated as the average reciprocal rank of the first relevant result across multiple queries.

.. math::
    MRR = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}(q)}

Where $\\text{rank}(q)$ is the position of the first relevant item for query $q$.

In comparison to the other metric :refconcept:`First Relevant Position (FRP)` introduced below, MRR is less affected by extreme outliers since the reciprocal transformation compresses very large rank values.

Mean Expected Reciprocal Rank (MERR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Mean Expected Reciprocal Rank (MERR)` extends the concept of Mean Reciprocal Rank (MRR) by incorporating relevance score and modeling user behavior (stopping at an ealier item) more realistically.

For a single query, :newconcept:`Expected Reciprocal Rank (ERR)` is calculated as:

.. math::
    \text{ERR}(q) = \sum_{i=1}^{n} \frac{1}{i} \times P(\text{stop at i})

where $P(\\text{stop at i})$ is the probability that the user stops at position $i$, which is a function of the document's relevance and the probability that the user didn't stop at earlier positions:

.. math::
    P(\text{stop at i}) = P(i) \times \prod_{j=1}^{i-1} (1 - P(j))

Here $P(i) \\in [0, 1]$ represents the probability that the user examines item at position $i$. It usually simply uses the relevacne scores, i.e., $P(i) = \\text{rel}(i)$. For relevance scores not in the range $[0, 1]$, normalization is needed. For example,

* :newconcept:`Exponential Utility Transformation`

    .. math::
        \text{rel}^{\text{exponential-utility-normalized}}(i) = \frac{2^{\text{rel}(i)} - 1}{2^R}

    where $\\text{rel}(i)$ is the assigned relevance grade for the document at position $i$, and $R$ is the maximum possible relevance score.

* :newconcept:`Sigmoid Transformation`
  
    .. math::
        \text{rel}^{\text{sigmoid-normalized}}(i) = \frac{1}{1 + \text{e}^{-\alpha(\text{rel}(i) - \beta)}}
    
    where $\\alpha$ controls the steepness of the curve and $\\beta$ shifts the midpoint. This transformation smoothly maps any range to $(0,1)$ and can better model gradual differences in relevance levels.


.. admonition:: Example: ERR Calculation
   :class: example-green

   Consider a search for "smartphone reviews" with the following ranked results and relevance grades (on a scale of 0-3):

   1. Comprehensive smartphone comparison guide (rel_score=3)
   2. Latest iPhone review (rel_score=2)
   3. Budget smartphones of 2024 (rel_score=3)
   4. Smartphone accessories (rel_score=1)
   5. Laptop reviews (rel_score=0)

   First, convert relevance scores to probabilities (using :refconcept:`Exponential Utility Transformation`):
   
   rel(1) = (2^3 - 1)/2^3 = 7/8 = 0.875
   rel(2) = (2^2 - 1)/2^3 = 3/8 = 0.375
   rel(3) = (2^3 - 1)/2^3 = 7/8 = 0.875
   rel(4) = (2^1 - 1)/2^3 = 1/8 = 0.125
   rel(5) = (2^0 - 1)/2^3 = 0

   ERR calculation:
   
   P(stop at 1) = rel(1) = 0.875
   P(stop at 2) = rel(2) × (1-rel(1)) = 0.375 × 0.125 = 0.047
   P(stop at 3) = rel(3) × (1-rel(1)) × (1-rel(2)) = 0.875 × 0.125 × 0.953 = 0.104
   P(stop at 4) = rel(4) × (1-rel(1)) × (1-rel(2)) × (1-rel(3)) = 0.125 × 0.125 × 0.953 × 0.896 = 0.013
   P(stop at 5) = rel(5) × ... = 0 (since rel(5)=0)

   ERR = (1/1 × 0.875) + (1/2 × 0.047) + (1/3 × 0.104) + (1/4 × 0.013) + (1/5 × 0)
   ERR = 0.875 + 0.023 + 0.035 + 0.003 = 0.936

   This high ERR score (0.936) indicates that users are likely to find a highly relevant result very early in the list, with most users stopping at the first position.


Mean ERR (MERR) is calculated by averaging ERR across all queries:

.. math::
    \text{MERR} = \frac{1}{|Q|} \sum_{q \in Q} \text{ERR}(q)

If we cut off a ranked list to position $k$, we have the **ERR@k** metric, and $MERR@k$ can be calculated accordingly.

Other Metrics
~~~~~~~~~~~~~

The following additional metrics are also suited for evaluating ranking performance in these contexts.

:newconcept:`Hit@k` is a binary metric that evaluates whether at least one relevant item appears in the top-k ranked results:

.. math::
   \text{Hit@k} = 
   \begin{cases} 
   1 & \text{if any relevant item appears in top-k results} \\
   0 & \text{otherwise}
   \end{cases}

:newconcept:`Mean Rank (MR)` measures the average position of relevant items in the ranked results.

.. math::
   \text{MR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{|R_q|} \sum_{r \in R_q} \text{rank}(r)

Where:
- $R_q$ is the set of relevant items for query $q$
- $\\text{rank}(r)$ is the position of relevant item $r$ in the ranked list

:newconcept:`First Relevant Position (FRP)` focuses solely on the rank of the first relevant item (similar to MRR):

.. math::
   \text{FRP} = \frac{1}{|Q|} \sum_{q \in Q} \min_{r \in R_q} \text{rank}(r)

Compared to :refconcept:`Mean Reciprocal Rank (MRR)` and `Mean Expected Reciprocal Rank (MERR)`, both :ul:`MR and FRP offer the benefit of being straightforward and immediately interpretable`, making them easier to understand for stakeholders without technical backgrounds. However, MR and FRP can be heavily skewed by a few queries where relevant items appear very late in the ranking.
A simple technique to address this limitation is to calculate **MR@k** and **FRP@k**, which only consider items within the top-k positions. For FRP@k specifically, a fixed constant "rank" value (e.g., k+1) can be assigned if no relevant item appears in the top-k results, creating a bounded metric while maintaining interpretability.

Summary
-------

This chapter examined ranking evaluation metrics for ML/AI systems that return ordered lists of items.

Position-Based Metrics
~~~~~~~~~~~~~~~~~~~~~~
  
  * **Precision@k**: Proportion of relevant items in the top-k results
  * **Recall@k**: Proportion of all relevant items found in the top-k results
  * **Mean Reciprocal Rank (MRR)** and **First Relevant Position (FRP)**: Focuses on the position of the first relevant item
  * **Mean Expected Reciprocal Rank (MERR)**: Focuses on early-item relevance, and incorporates user behavior modeling
  * **Hit@k**: Binary metric indicating if at least one relevant item appears in top-k

Comprehensive Ranking Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  * **Mean Average Precision (MAP)** and **Mean Average Recall (MAR)**: Averages precision/recall at each position where a relevant item appears
  * **Normalized Discounted Cumulative Gain (NDCG)**: Evaluates both relevance and position with discount factors
  * **Mean Rank**: Average position of relevant items in the results

Low-Precision Ranking Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  * Few relevant items in a large candidate pool
  * Primary goal is surfacing at least one relevant item in top positions
  * MRR, Hit@k, and First Relevant Position are particularly well-suited

Best Practices
~~~~~~~~~~~~~~
  
  * Choose metrics aligned with how users interact with results
  * Consider the constraints of the interface (e.g., visibility constraints on mobile/smarthome devices)