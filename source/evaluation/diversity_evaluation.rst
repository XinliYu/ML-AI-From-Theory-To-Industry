Diversity Evaluation
====================

Many ML/AI ranking systems aim not only to surface relevant items but also to provide diverse results that cover multiple aspects of a task. :newconcept:`Diversity Evaluation` assesses how well a system presents varied, non-redundant items that collectively satisfy different user needs or interpretations.

Unlike precision and recall which focus on relevance, diversity metrics evaluate the breadth of coverage across different subtopics, categories, or interpretations. Diversity is particularly valuable in:

1. **Ambiguous queries**: When a query has multiple valid interpretations (e.g., "apple" could refer to a fruit or a technology company)
2. **Exploratory search**: When users are exploring a topic without a specific information need
3. **Recommendation systems**: Where users benefit from varied suggestions rather than similar items
4. **Risk mitigation**: To reduce the chance of completely missing user intent
5. **Filter bubble reduction**: To avoid exposing users only to content similar to what they've previously consumed
6. **User satisfaction**: To provide a more engaging experience through content variety


Subtopic Diversity Metrics
--------------------------

Subtopic Recall (S-Recall)
~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Subtopic Recall` measures the proportion of subtopics covered in the top-k results (focusing solely on coverage breadth rather than relevance or position):

.. math::

   \text{S-Recall}@k = \frac{|\cup_{i=1}^{k} \text{subtopics}(i)|}{|S|}

where

* $\\text{subtopics}(i)$ is the set of subtopics covered by the item at position $i$
* $|S|$ is the total number of subtopics for the query


Subtopic Entropy
~~~~~~~~~~~~~~~~

:newconcept:`S-Entropy` measures how evenly items are distributed across subtopics:

.. math::

   \text{Entropy} = -\sum_{j=1}^{|S|} P(j) \log P(j)

where

* $S$ is the set of all subtopics
* $P(j)$ is the proportion of subtopic $j$ in the ranked results

Higher entropy indicates a more balanced distribution across categories. Maximum entropy occurs when all categories are equally represented, while minimum entropy (0) occurs when all items belong to a single category.


Gini Coefficient
~~~~~~~~~~~~~~~~

:newconcept:`Gini Coefficient` measures inequality in representation across subtopics:

.. math::

   \text{Gini} = \frac{1}{2} \times \frac{\sum_{i=1}^{|S|} \sum_{j=1}^{|S|} |s_i - s_j|}{|S|\sum_{i=1}^{|S|} s_i}

where 

* $s_i$ is the number of items from subtopic $i$ in the results.
* Lower values (approaching 0) indicate more equal representation. Imaging the items are evenly distributed to all subtopics, then every $s_i$ is equal to $s_j$, and above value is surely zero.
* Higher values (approaching 1) indicate dominance by fewer categories. In the extreme case, if one subtopic $x$ takes all items, then $\\sum_{i=1}^{\|S\|} s_i = 0+...+s_x+0+...=s_x$. Also,
  
   .. math::

      |s_i - s_j| = 
      \begin{cases} 
      0, & \text{if } i = j \\
      N, & \text{if } i = x \text{ and } j \neq x \\
      N, & \text{if } i \neq x \text{ and } j = x \\
      0, & \text{if } i \neq x \text{ and } j \neq x
      \end{cases}
  
  As a result,

   .. math::
      \sum_{j=1}^{|S|} |s_i - s_j| = 2 \times |S-1| \times s_x
  
  and therefore

   .. math::

      \text{Gini} = \frac{1}{2} \times \frac{2 \times (|S|-1) \times N}{|S| \times N} = \frac{(|S|-1)}{|S|} = 1 - \frac{1}{|S|}
  
  If there is only one subtopic, Gini Coefficient is still 0. With more subtopics $\|S\| = 2, 3, 4, ...$, the Gini Coefficient will go up to $0.5, 0.67, 0.75, ...$, indicating more inequality.


Proportionality (Total Variation Distance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Proportionality` measures how well the distribution of items across categories in the results matches a target distribution. It is based on :newconcept:`Total Variation Distance (TVD)` between two distributions,

.. math::

   \text{Proportionality}@k = 1 - \text{TVD}(P_\text{T}, P_\text{R})

where

.. math::

   \text{TVD}(P_\text{T}, P_\text{R}) = \frac{1}{2} \sum_{j=1}^{|S|} |P_\text{T}(j) - P_\text{R}(j)|

where

* $P_\\text{T}(j)$ is the target proportion for subtopic $j$
* $P_\\text{R}(j)$ is the actual proportion of subtopic $j$ in results
* The $\\frac{1}{2}$ coefficient is due to the max value of $∑\|P_{\\text{T}}(j) - P_{\\text{R}}(j)\|$ is 2, in extreme cases like $P_\\text{T} = (1, 0, 0)$ and $P_\\text{R} = (0, 1, 0)$.

A Proportionality value of 1 indicates perfect alignment with the target distribution, while 0 indicates maximum divergence.


Intent-Aware Ranking Metrics
----------------------------

:newconcept:`Intent-Aware` ranking metrics extend traditional relevance-based ranking metrics by incorporating subtopic or aspect coverage. 

* Traditionally, this requires a pre-defined set of subtopics $S$.
* Recently, LLM-driven AI systems can help dynamically generate varying subtopics for each query.


α-nDCG
~~~~~~

:newconcept:`α-nDCG` (Alpha-nDCG) extends :refconcept:`Normalized Discounted Cumulative Gain (NDCG)` by penalizing redundancy across subtopics. It is replacing the :refconcet:`gain function` $\\text{rel}(i)$ in the original DCG formula $\\text{DCG}(k) = \\sum_{i=1}^{k} \\frac{\\text{rel}(i)}{\\log_2(i+1)}$ by new topic-aware abd topic-coverage dependent score $\\sum_{j=1}^{\|S\|}g(i,j)$.

.. math::

   \alpha\text{-DCG}@k = \sum_{i=1}^{k} (\frac{1}{\log_2(i+1)} \times G(i, S))

where $G(i, S)$ is a :newconcept:`subtopics-aware gain function` considering the set of all subtopics $S$. One example of such function is 

.. math::

   G(i, S) = \frac{\sum_{j=1}^{|S|} g(i,j)}{|S|}

where $\|S\|$ is the number of topics, and $g(i,j)$ is the gain for subtopic $j$ at position $i$, typically defined as:

.. math::

   g(i,j) = \text{rel}(i,j) \cdot (1-\alpha)^{r(i-1,j)}

where

* $\\text{rel}(i,j)$ is the relevance of item at position $i$ to subtopic $j$.
* $r(i-1,j)$ is the number of items relevant to subtopic $j$ that appear before position $i$.
* $\\alpha$ is the :newconcept:`redundancy penality parameter` (typically 0.5).

.. note:: Understanding the Redundancy Penalty

   The gain function $g(i,j) = \\text{rel}(i,j) \\cdot (1-\\alpha)^{r(i-1,j)}$ is specifically designed to penalize redundancy across subtopics:
   
   * The first component, $\\text{rel}(i,j)$, represents the basic relevance value of the item at position $i$ to subtopic $j$.
   * The second component, $(1-\\alpha)^{r(i-1,j)}$, is a discount factor that decreases exponentially as more items covering the same subtopic appear earlier in the ranking.
   
   When the first item covering subtopic $j$ appears:
   
   * $r(i-1,j) = 0$ (no previous items cover this subtopic)
   * $(1-\\alpha)^0 = 1$ (no penalty applied)
   * $g(i,j) = \\text{rel}(i,j)$ (full relevance value)
   
   For subsequent items covering the same subtopic:
   
   * Each additional item receives an increasingly severe penalty
   * With $\\alpha = 0.5$, the second item relevant to subtopic $j$ gets a 50% discount, the third gets a 75% discount, and so on
   
   This mathematical formulation elegantly captures the diminishing returns property: :ub:`while the first result about a subtopic is highly valuable, subsequent results on the same subtopic provide decreasing additional value to users`.
   
   The $\\alpha$ parameter allows system designers to tune how strongly to enforce diversity:
   
   * Higher values of $\\alpha$ (closer to 1) create stronger penalties for redundancy, encouraging more diverse rankings
   * Lower values (closer to 0) apply milder penalties, allowing more items from the same subtopic when they're highly relevant

Similar to NDCG, α-nDCG is normalized by dividing by the ideal α-DCG value:

.. math::

   \alpha\text{-nDCG}@k = \frac{\alpha\text{-DCG}@k}{\alpha\text{-IDCG}@k}


Intent-Aware Expected Reciprocal Rank (ERR-IA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`ERR-IA` adapts :refconcept:`Expected Reciprocal Rank (ERR)` to account for multiple intents:

.. math::

   \text{ERR-IA}@k = \sum_{j=1}^{|S|} P(j) \times \text{ERR}(q, j)

where $\\text{ERR}(q, j)$ is the ERR metric of query $q$ with respect to subtopic $j$:

.. math::

   \sum_{i=1}^{n} \frac{1}{i} \cdot \text{rel}(i,j) \cdot \prod_{l=1}^{i-1} (1-\text{rel}(l,j))

where

* $P(j)$ is the probability or importance of subtopic $j$
* $\\text{rel}(i,j)$ is the relevance of item at position $i$ to subtopic $j$

Similar to :refconcept:`Expected Reciprocal Rank (ERR)`, we assume $\\text{rel}(i,j) \\in [0, 1]$ represents the probability that the user finds the document at position $i$ relevant to subtopic $j$. For relevance scores not in the range $[0, 1]$, normalization is needed.


Similarity-Based Diversity Metrics
----------------------------------

These metrics measure diversity based on item similarities without requiring intent or subtopic definitions and annotations.


Intra-List Diversity (ILD)
~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Intra-List Diversity` measures the average dissimilarity between all pairs of items in the ranked list:

.. math::

   \text{ILD}@k = \frac{1}{k(k-1)} \sum_{i=1}^{k} \sum_{j=1, j \neq i}^{k} d(i,j)

Where $d(i,j)$ is a distance or dissimilarity function between items at positions $i$ and $j$.


Expected Intra-List Diversity (EILD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Expected Intra-List Diversity` extends ILD by incorporating rank and relevance awareness:

.. math::

   \text{EILD}@k = \sum_{i=1}^{k} \sum_{j=1, j \neq i}^{k} P(i) \times P(j) \times d(i,j)

where:

* $P(i)$ is the probability of user examing item $i$, and $P(i) \\times P(j)$ can be interpreted as the probability user examining both item $i$ and item $j$. Similar to :refconcept:`Expected Reciprocal Rank (ERR)`, it can be simply $P(i) = \text{rel}(i)$ given that the relevance score is or can be normalized as probabilistic (i.e., in range $[0, 1]$).
* $d(i,j)$ is the same dissimilarity function used in ILD

EILD gives a more user-centric view of diversity by considering the probability user actually examing the items.


Summary
-------

This chapter examined diversity evaluation metrics for ML/AI systems that aim to present varied results covering multiple aspects of a query.

Subtopic Diversity Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~
  
* **Subtopic Recall (S-Recall)**: Measures the proportion of subtopics covered in the top-k results
* **Subtopic Entropy**: Quantifies how evenly items are distributed across subtopics
* **Gini Coefficient**: Measures inequality in representation across subtopics
* **Proportionality**: Assesses alignment between actual and target subtopic distributions

Intent-Aware Ranking Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
* **α-nDCG**: Extends NDCG by penalizing redundancy across subtopics using a diminishing returns model
* **ERR-IA**: Adapts Expected Reciprocal Rank to account for multiple user intents

Similarity-Based Diversity Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
* **Intra-List Diversity (ILD)**: Measures average dissimilarity between all pairs of items
* **Expected Intra-List Diversity (EILD)**: Extends ILD by incorporating user examination probabilities

Best Practices
~~~~~~~~~~~~~

* **Balance diversity with relevance**: Optimize for both metrics based on application context
* **Choose appropriate metrics**: Select diversity measures that align with specific diversity goals
* **Consider query ambiguity**: Apply higher diversity requirements for ambiguous or exploratory queries
* **Calibrate to user expectations**: Adapt diversity levels to match user needs and application context