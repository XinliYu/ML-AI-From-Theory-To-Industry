Novelty Metrics
---------------

Closely related to diversity is the concept of :newconcept:`novelty`, which focuses on exposing users to previously unseen or unusual items rather than just providing variety within a single result set.

Expected Popularity Complement (EPC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Expected Popularity Complement (EPC)` measures the average unpopularity of recommended items:

.. math::

   \text{EPC}@k = \frac{1}{k} \sum_{i=1}^{k} (1 - pop(i))

Where $pop(i)$ is the normalized popularity of the item at position $i$, typically defined as the number of users who have interacted with the item divided by the total number of users.

Expected Intra-List Novelty (EIN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Expected Intra-List Novelty` measures the novelty of items relative to what a user has already seen in the ranked list:

.. math::

   \text{EIN}@k = \frac{1}{k} \sum_{i=1}^{k} \frac{1}{i-1} \sum_{j=1}^{i-1} d(i,j)

Where $d(i,j)$ is a distance function between items at positions $i$ and $j$. This metric rewards lists where each new item is substantially different from previously seen items.

Calibrated Diversity
------------------

:newconcept:`Calibrated Diversity` aims to match the diversity level to user expectations or needs, rather than simply maximizing diversity:

.. math::

   \text{Calibrated Diversity}@k = 1 - |D_{observed}@k - D_{expected}@k|

Where:
* $D_{observed}@k$ is the observed diversity in the top-k results
* $D_{expected}@k$ is the expected or desired diversity level, which might be:
  * Based on user history
  * Derived from the query itself
  * Set according to application-specific requirements

Personalized Diversity
--------------------

:newconcept:`Personalized Diversity` metrics consider individual user preferences when evaluating diversity:

User-Specific Topic Coverage (USTC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`User-Specific Topic Coverage` measures how well the results cover topics of interest to a specific user:

.. math::

   \text{USTC}@k = \frac{|\cup_{i=1}^{k} topics(i) \cap topics_{user}|}{|topics_{user}|}

Where:
* $topics(i)$ is the set of topics covered by the item at position $i$
* $topics_{user}$ is the set of topics the user is interested in

Unexpectedness
~~~~~~~~~~~~

:newconcept:`Unexpectedness` measures the proportion of recommended items that deviate from a user's expected items:

.. math::

   \text{Unexpectedness}@k = \frac{|\{i \in R_k: i \notin E\}|}{k}

Where:
* $R_k$ is the set of top-k recommendations
* $E$ is the set of expected items (often derived from a baseline recommender)

.. admonition:: Example: Personalized Diversity in Music Recommendations
   :class: example-green

   Consider a music streaming service that recommends songs to a user with known preferences for rock, indie, and electronic music. The system evaluates two recommendation strategies:

   **User Profile:**
   - Preferred genres: Rock (60% of listening history), Indie (30%), Electronic (10%)
   - Recently played artists: The Strokes, Arctic Monkeys, Tame Impala

   **Recommendation Set A:**
   1. "Last Nite" by The Strokes (Rock)
   2. "Do I Wanna Know?" by Arctic Monkeys (Rock)
   3. "Reptilia" by The Strokes (Rock)
   4. "Why'd You Only Call Me When You're High?" by Arctic Monkeys (Rock)
   5. "Someday" by The Strokes (Rock)

   **Recommendation Set B:**
   6. "Fluorescent Adolescent" by Arctic Monkeys (Rock)
   7. "Let It Happen" by Tame Impala (Psychedelic/Indie)
   8. "Midnight City" by M83 (Electronic)
   9. "Little Dark Age" by MGMT (Indie/Electronic)
   10. "The Less I Know The Better" by Tame Impala (Psychedelic/Indie)

   **Calibrated Diversity Calculation:**
   
   Expected genre distribution based on user history: Rock (60%), Indie (30%), Electronic (10%)
   
   Observed genre distribution in Set A: Rock (100%), Indie (0%), Electronic (0%)
   Calibrated Diversity(A) = 1 - (|1.0-0.6| + |0.0-0.3| + |0.0-0.1|)/2 = 1 - 0.4 = 0.6
   
   Observed genre distribution in Set B: Rock (20%), Indie/Psychedelic (60%), Electronic (20%)
   Calibrated Diversity(B) = 1 - (|0.2-0.6| + |0.6-0.3| + |0.2-0.1|)/2 = 1 - 0.3 = 0.7
   
   **Unexpectedness Calculation:**
   
   Assuming the expected items are songs by The Strokes, Arctic Monkeys, and Tame Impala:
   
   Unexpectedness(A) = 0/5 = 0 (all recommendations are from expected artists)
   
   Unexpectedness(B) = 2/5 = 0.4 (2 recommendations from unexpected artists: M83 and MGMT)
   
   **Analysis:**
   
   Set B provides better personalized diversity than Set A:
   - It more closely matches the user's historical genre distribution (better calibrated diversity)
   - It introduces unexpected artists while maintaining connection to the user's preferences
   - It balances familiarity (artists the user knows) with discovery (new artists in genres the user enjoys)
   
   This example illustrates how personalized diversity metrics can capture the quality of recommendations beyond simply maximizing variety, focusing instead on meaningful diversity that aligns with user preferences.

Trade-offs Between Relevance and Diversity
----------------------------------------

Optimizing for diversity often involves trade-offs with relevance metrics:

1. **Relevance-Diversity Balance**: Increasing diversity may require including less relevant items for underrepresented aspects
2. **Application-Specific Priorities**: News platforms may prioritize diversity more than specialized technical search engines
3. **User Intent Clarity**: Diversity matters more for ambiguous queries and less for highly specific ones

Many systems employ a hybrid approach where top positions prioritize relevance while ensuring reasonable diversity across the complete result set.

Evaluation Challenges
-------------------

Evaluating diversity presents several unique challenges:

Subjective Nature
~~~~~~~~~~~~~~~

Diversity is inherently subjective and context-dependent. What constitutes appropriate diversity varies by:
* Query type (navigational vs. exploratory)
* User experience level (novices may prefer more diversity than experts)
* Domain (news benefits from diverse viewpoints, technical documentation less so)

Annotation Complexity
~~~~~~~~~~~~~~~~~~

Creating ground truth for diversity evaluation requires:
* Identifying all possible subtopics or interpretations
* Annotating items with multiple subtopic relevance judgments
* Determining appropriate weights or importance for each subtopic

Metric Selection
~~~~~~~~~~~~~~

Different metrics capture different aspects of diversity:
* Coverage metrics (S-Recall) focus on breadth
* Intent-aware metrics (α-nDCG) balance relevance and diversity
* Distance-based metrics (ILD) focus on dissimilarity between items

Online vs. Offline Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Offline diversity metrics may not fully capture user satisfaction:
* Users might prefer less diverse but more relevant results in specific contexts
* The perceived value of diversity can vary based on task and user intent
* A/B testing with explicit diversity interventions may be necessary to validate metric improvements

Selection of an Appropriate Diversity Metric
-----------------------------------------

The choice of diversity metric should align with application requirements:

* **Intent-aware metrics** (α-nDCG, ERR-IA): Best when subtopic annotations are available and interpretations are well-defined
* **Explicit diversity metrics** (ILD, S-Recall): Suitable when item similarity or categorization is available
* **Balance metrics** (Proportionality, Gini): Appropriate when category distribution targets are known

A common practice is to evaluate both relevance and diversity metrics in parallel, looking for optimal configurations that maintain strong relevance while improving diversity.

.. admonition:: Code: Implementing Basic Diversity Metrics
   :class: code-grey

   .. code-block:: python

      import numpy as np
      from sklearn.metrics.pairwise import cosine_similarity
      
      def subtopic_recall(results, subtopics, k=10):
          """
          Calculate Subtopic Recall@k
          
          Parameters:
          -----------
          results : list of dicts
              Ranked results with 'id' and 'subtopics' fields
          subtopics : set
              Set of all possible subtopics for the query
          k : int
              Number of top results to consider
          
          Returns:
          --------
          float
              S-Recall@k value
          """
          covered_subtopics = set()
          for i in range(min(k, len(results))):
              covered_subtopics.update(results[i]['subtopics'])
          
          return len(covered_subtopics) / len(subtopics)
      
      def intra_list_diversity(results, embeddings, k=10):
          """
          Calculate Intra-List Diversity@k using cosine distance
          
          Parameters:
          -----------
          results : list
              Ranked result IDs
          embeddings : dict
              Mapping from item ID to vector representation
          k : int
              Number of top results to consider
          
          Returns:
          --------
          float
              ILD@k value
          """
          k = min(k, len(results))
          if k <= 1:
              return 0.0
              
          # Get embeddings for top-k results
          vectors = np.array([embeddings[results[i]] for i in range(k)])
          
          # Calculate similarity matrix
          sim_matrix = cosine_similarity(vectors)
          
          # Convert to distance and sum off-diagonal elements
          dist_sum = np.sum(1 - sim_matrix) - k  # Subtract diagonal elements (self-similarity)
          
          # Normalize by number of pairs
          return dist_sum / (k * (k - 1))

Practical Applications
--------------------

Different applications prioritize diversity in distinct ways:

Search Engines
~~~~~~~~~~~~~

* **Web search**: Diverse results for ambiguous queries (e.g., "jaguar" → car, animal, sports team)
* **E-commerce**: Category diversity to showcase product range
* **Academic search**: Viewpoint diversity to present multiple scholarly perspectives

Recommendation Systems
~~~~~~~~~~~~~~~~~~~

* **Media streaming**: Genre and mood diversity to prevent fatigue
* **News aggregators**: Source and viewpoint diversity to reduce filter bubbles
* **E-commerce**: Price point and brand diversity to provide comparison options

Feed Ranking
~~~~~~~~~~

* **Social media**: Topic diversity to maintain engagement
* **Content platforms**: Creator diversity to support broader ecosystem
* **News feeds**: Temporal diversity to balance breaking news with evergreen content

Summary
------

Diversity evaluation extends traditional relevance-based assessment to measure how well ML/AI systems present varied, comprehensive results.

Key Diversity Metrics
~~~~~~~~~~~~~~~~~~~~

* **Intent-Aware Metrics**: α-nDCG and ERR-IA incorporate subtopic coverage
* **Explicit Diversity Metrics**: Intra-List Diversity and Subtopic Recall directly measure variety
* **Balance Metrics**: Proportionality and Gini Coefficient evaluate categorical representation
* **Novelty Metrics**: Expected Popularity Complement and Intra-List Novelty measure exposure to unusual items
* **Personalized Metrics**: User-Specific Topic Coverage and Unexpectedness adapt to individual preferences

Diversity-Relevance Balance
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Trade-offs**: Often necessary to sacrifice some relevance for improved diversity
* **Calibration**: Matching diversity levels to user expectations rather than maximizing
* **Application-Specific**: Different domains require different diversity approaches

Best Practices
~~~~~~~~~~~~~

* **Hybrid Evaluation**: Assess both relevance and diversity metrics
* **User Studies**: Validate diversity metrics with explicit user feedback
* **Contextual Approach**: Adjust diversity expectations based on query type and user intent
* **Appropriate Metrics**: Select diversity measures aligned with application goals

Effective diversity evaluation requires understanding the specific dimensions of variety that matter most to users in a given context, then developing metrics that accurately capture those dimensions while maintaining strong overall result quality.