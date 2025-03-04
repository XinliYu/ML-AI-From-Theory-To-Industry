Basic Evaluation
================

In evaluating the performance of Machine Learning (ML) and Artificial Intelligence (AI) systems, :newconcept:`precision` and :newconcept:`recall` are foundational metrics that quantify how well a system retrieves or recommends relevant items. These metrics originated in :newconcept:`information retrieval (IR)` and :newconcept:`recommender/ranking systems (RecSys)`, but are now widely applied across various ML and AI tasks, including LLM (Large Language Model)-driven AI tasks. 

For simplicity without loss of generality, we adopt standard terminology from IR and RecSys, focusing on :newconcept:`ground truth based evaluation` in this section — where labeled data provides the reference standard for assessing model predictions.

Dataset & Ground Truth
----------------------
A :newconcept:`dataset` consists of three essential components, :newconcept:`queries` (denoted by $Q$) representing the inputs into the ML/AI system, :newconcept:`documents` (denoted by $D$) representing the pool of all candidate items the ML/AI system can choose from to respond to a query, and the :newconcept:`annotations` $A$ where $A \\subset Q \\times D \\times \\{0, 1\\}$. 

* The terms "**query**" and "**document**" are generic concepts in IR. They can take many concrete forms in practice. For example, a query can be a user profie, and a document can be a Song, Movie, or Ad. In practice, we often refer to a "document" as a "**candidate item**" to be recommended by the ML/AI system.
* Each annotation $a=(q, d, l)\\in A$ is a triplet consiting of a query $q$, a document $d$ and a :newconcept:`label` $l\\in\\{0, 1\\}$ where the :newconcept:`positive label` "$1$" means $d$ is relevant to $q$ and :newconcept:`negative label` "$0$" means otherwise. Each annotation is sometimes also referred to as a :newconcept:`data sample`.

The annotations provide information known to be factually accurate because it has been verified or labeled by humans or reliable processes. It :ul:`serves as the reference` standard against which predictions or measurements are compared. It is :ul:`context-specific` and represents the "correct answer" for a particular machine learning or AI application, :ul:`may still contain some mistakes or subjectivity`, but is treated as correct for evaluation purposes.

In this chapter we assume annotations with :newconcept:`binary relevance scores` (i.e., the label is either 0 or 1). In practice, :newconcept:`graded relevance scores` are also common, such as Amazon product review ratings (1-5 stars). It is usually difficult to ask humans to assign a continuous and normalized :newconcept:`probabilistic relevance score` from 0 to 1, but it is more common with model-based ground truth (e.g., using a ecoder model to compute embeddings between the query and candidate item, and treat their embeeding similarity as the ground-truth relevance score). LLMs are also frequently used to simulate human annotations and produce various relevance scores. The graded/probabilistic relevance scores can be mapped to binary relevance scores through a threshold or more complex rules. The graded/probabilistic relevance scores will be more involved in :refconcept:`Ranking Evaluation` and :refconcept:`Diversity Evaluation`.

The concept of "query" and "documents" even generalize to scenarios beyond classic IR an RecSys. For example, in modern NLP tasks like generative summarization or translation, the "query" is the original text pieces, and the "documents" is an infinite set of possible candidates. :ul:`The annotations can be provided "on-the-fly" using a model`.

Groud Truth
~~~~~~~~~~~

Given a query $q$, a :newconcept:`model` $M(q, d)$ is a function generating a :newconcept:`predicted relevance score` for every candidate $d \\in D$. We choose a :newconcept:`threshold` $t$ and all items $d$ such that $M(q, d) \> t$ are considered relevant or recommended.

Given a query $q$, its :newconcept:`Ground Truth`, denoted by $A_q$ where $A_q \\subset \\{q\\} \\times D \\times \\{0, 1\\} \\subset A$, is all annotations associated with $q$. Ground truth in practice often only provides positive labels for a subset of $D$, with the remaining items from $D$ assumed to be negatively labeled. Ground truth can be broken down in two ways, as shown in :numref:`fig-ground-truth`.

.. figure:: ../_static/images/evaluation/ground_truth_breakdown.png
   :alt: Break Down of Ground Truth showing TP, FP, TN, and FN categories
   :width: 100%
   :name: fig-ground-truth
   
   Break Down of Ground Truth showing TP, FP, TN, and FN categories

Depending on if an item is relevant:

* :newconcept:`!Positive/Relevant Items (P)`: Items that match the user needs or preferences according to the ground truth. These are the items that should ideally be recommended or retrieved. Also called the :newconcept:`Positive Class`. Positive/Relevant Items include :refconcept:`True Positive` and :refconcept:`False Negative` (i.e., P=TP+FN).
* :newconcept:`!Negative/Irrelevant Items (N)`: Items that do not match the user's needs or preferences according to the ground truth. These items should ideally not be recommended or retrieved. Also called the :newconcept:`Negative Class`. Negative/Irrelevant Items include :refconcept:`False Positive` and :refconcept:`True Negative` (i.e., N=FP+TN).
* Whichever of these two classes is significantly smaller is called the :newconcept:`minority class`. By convention, we assume the positive class is the minority class (without loss of generality), because if the negative class is the minority, we can simply flip the labels. While many real-world ML/AI applications involve imbalanced classes (e.g., retrieval, recommendation, anomaly detection), certain problems can have relatively balanced classes (for example, sentiment analysis where positive and negative sentiments are comparably frequent, or stock market prediction with similar rates of upward and downward movements).

Depending on if an item is recommended:

* **Recommended Items**: Items that are suggested to the user as potentially relevant or interesting. Examples include documents retrieved by a search engine, products recommended by an e-commerce platform, or advertisements displayed to a user. "Recommended Items" consists of:

  * :newconcept:`!True Positive (TP)`: Items that are both recommended by the system and genuinely relevant to the user. These represent successful recommendations that match user needs.
  * :newconcept:`!False Positive (FP)`: Items that are recommended by the system but are not actually relevant to the user. These are also called :newconcept:`Type I errors` (or :newconcept:`errors of commission`) where the system incorrectly includes irrelevant items.

* **Not-Recommended Items**: All other items in the ground truth that are not suggested to the user.

  * :newconcept:`!True Negative (TN)`: Items neither recommended by the system nor relevant to the user. These represent correct decisions to exclude irrelevant items from recommendations.
  * :newconcept:`!False Negative (FN)`: Items that are not recommended by the system but would have been relevant to the user. These represent :newconcept:`Type II errors` (:newconcept:`errors of omission`) where the system fails to identify relevant items.

Confusion Matrix
~~~~~~~~~~~~~~~~

A confusion matrix is a structured way to evaluate a model's performance by categorizing its predictions into four possible outcomes:

.. list-table:: Confusion Matrix
   :header-rows: 1
   :widths: 20 20 20

   * - **Actual / Predicted**
     - **Relevant (Predicted Positive)**
     - **Irrelevant (Predicted Negative)**
   * - **Relevant (Actual Positive)**
     - True Positive (TP)
     - False Negative (FN)
   * - **Irrelevant (Actual Negative)**
     - False Positive (FP)
     - True Negative (TN)

The confusion matrix provides a structured way to analyze errors and trade-offs.

Precision
---------

:newconcept:`!Precision` metric measures the proportion of recommended items that are relevant to the user.

.. math::

   \text{Precision} = \frac{\text{TP}}{\text{Recommended Items}} = \frac{\text{TP}}{\text{TP} + \text{FP}}

:newconcept:`Minimum Baseline Precision` refers to the precision when the system recommends everything.

.. math::

   \text{Minimum Baseline Precision} = \frac{\text{P}}{\text{All Items}} = \frac{\text{P}}{\text{P} + \text{N}}

.. note::
   The "Minimum Baseline Precision" metric :ul:`serves great guardrail purpose when the candidate pool is small and already consists mostly of positive items`. For example, in multi-phase recommendations, the last phase could be choosing the top-3 from 10 candidates where most candidates are already relevant, and then we need to compare Precision with this metric.

Recall & FPR
------------

:newconcept:`!Recall` metric (also known as :newconcept:`sensitivity`, :newconcept:`True Positive Rate (TPR)`) measures the proportion of relevant items that were successfully recommended. It answers the question: "Of all relevant items, how many did we recommend?"

.. math::

   \text{Recall} = \frac{\text{TP}}{\text{P}} = \frac{\text{TP}}{\text{Relevant Items}} = \frac{\text{TP}}{\text{TP} + \text{FN}}

While recall focuses on the positive items, the :newconcept:`False Positive Rate (FPR)` focuses on the negative itesms. It answers the question: "Of all irrelevant items, how many did we recommend?". 

.. math::

   \text{FPR} = \frac{\text{FP}}{\text{N}} = \frac{\text{FP}}{\text{Irrelevant Items}} = \frac{\text{FP}}{\text{FP} + \text{TN}}

F1 Score
--------

Given :math:`n` quantities :math:`x_1, ..., x_n`, their :newconcept:`harmonic mean` :math:`H` is:

.. math::

   H(x_1, ..., x_n) = \frac{n}{\sum_{i=1}^{n}{\frac{1}{x_i}}}

where a key property is any one of :math:`x_i, i=1, ..., n` approaches :math:`0`, the mean approaches :math:`0`. To obtain a reasonably high harmonic mean, none of the individual values should approach :math:`0`, as shown in :numref:`fig-harmonic-mean`. Intuitively, :ul:`this metric is giving more weights to low values`.

.. figure:: ../_static/images/evaluation/harmonnic_mean.png
   :alt: A plot of 2D harmonic mean between range 0 and 1
   :width: 100%
   :name: fig-harmonic-mean
   
   A plot of 2D harmonic mean between range 0 and 1. The mean approaches 0 if either x or y approaches 0.

The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both considerations. The harmonic mean is used instead of the arithmetic mean because it gives more weight to low values. Intuitively, :ul:`a reasonable F1-score performance needs to have reasonable performance in precision and recall`.

.. math::

   \text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}


Precision-Recall Trade-off
--------------------------

In most machine learning and AI systems, there exists an inherent trade-off between precision and recall. This trade-off arises because:

1. **Increasing recall** typically requires recommending more items, which often leads to including more irrelevant items, thus decreasing precision.
2. **Increasing precision** typically requires being more selective about recommendations, which often means missing some relevant items, thus decreasing recall.

This trade-off is particularly evident when adjusting the threshold for making recommendations:

* A **higher threshold** (recommending fewer items) tends to increase precision but decrease recall.
* A **lower threshold** (recommending more items) tends to increase recall but decrease precision.

The F1 score addresses this trade-off by finding a balance point where both precision and recall are reasonably high. Since the F1 score uses the harmonic mean, it penalizes systems that achieve high performance in one metric at the expense of poor performance in the other. For example:

* A system with precision = 1.0 and recall = 0.1 would have F1 = 0.18
* A system with precision = 0.5 and recall = 0.5 would have F1 = 0.5

This demonstrates how the F1 score favors balanced performance over excellence in just one dimension. In practical applications, the choice between optimizing for precision, recall, or F1 depends on the specific requirements of the task:

* **Precision-focused applications**: :ul:`High risk at missing false positives`. For example, medical diagnosis systems where false positives can lead to unnecessary treatments.
* **Recall-focused applications**: :ul:`High risk at missing false negatives`. For example, fraud detection systems where missing fraudulent cases is costly.
* **F1-focused applications**: General recommendation systems where overall user experience depends on both relevance and coverage.


Threshold Dependency & PR Curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In practice, :ul:`precision-recall trade-off is often tweaked by adjusting a threshold on the relevance score` (assuming model and its hyperparameters are fixed) on the training data. For example, for a "F1-focused application", we could tweak a final relevance score threshold to achieve the best F1 score on the training data. When given multiple models, or multiple sets of hyperparameters, such threshold dependent evaluation has limitation.

1. **Threshold Dependency Challenge**: Precision, recall, and F1 score at a single threshold may not provide a complete performance picture.
   
   * Different business contexts may require different operating thresholds.
   * Threshold selection introduces subjectivity into evaluation.
   * Optimal thresholds on training data often don't generalize well to production environments.

2. **Dynamic Requirements**: In many real-world applications, the optimal threshold may not be reflected in the test set, and may change over time due to various business dynamics, such as shifting data distributions, evolving business priorities, changing cost structures and seasonal variations.

3. **Comparison Difficulties**: When comparing multiple models/systems:
   
   * Each model might perform optimally at a different threshold. A model that performs well at one threshold may perform poorly at others
   * While each model can theoretically be tuned to its own "best" threshold using training or validation data, but even this approach has limitations:

     * Varying threshold per model instead of a single standardized operating point might complicate science, production and business operations.
     * The thresholds are still subjective and limited to offline data, and doesn't capture a model's robustness to threshold adjustments, which is important in dynamic environments.

These limitations motivate the need for a :newconcept:`threshold-agnostic evaluation` approach that evaluates overall performance across :ul:`all possible threshold values`. A typical technique is to plot a graph for the trade-off metrics, and here it is the :newconcept:`PR Curve`, and :newconcept:`Area Under Curve (AUC)` (here it is :newconcept:`Area Under the PR Curve (AUC-PR)`) is a single, threshold-agnostic and comprehensive metric measuring the overall performance.

In comparison to the :refconcept:`ROC curve`, PR curve typically applies when
   1. The dataset has :ul:`strong class imbalance`, and the ML/AI system is :ul:`constrained to output a maximum number of relevant items` (e.g., in search results where the first page only shows 20 items, or fraud detection use cases where human judgment is required and the maxium number of cases to review is capped by huamn resource), then the :refconcept:`False Positive Rate` would naturally be low nonetheless, making it harder to compare different models/systems.
   2. :ul:`Precision is more critical` (to minimize false positive while managing recall), and the evaluation is required to be :ul:`sensitive to the minority class` (and therefore AUC-PR is considered a precision-focused metric).
   3. Evaluation requires :ul:`directly highlight the precision-recall trade-off`.

Analagous to PR Curve, and AUC-PR, there is :newconcept:`Recall-Precision Curve` (RP Curve) and :newconcept:`AUC-RP`, which is more recall-focused while managing the precision.

.. admonition:: Example : Email Spam Detection Model Comparison
   :class: example-green

   **Scenario:**
   Our precision-recall curves compare two spam detection models (Model A and Model B) that assign scores between 0 and 1 to emails, with higher scores indicating higher likelihood of being spam. Both models are evaluated on 1000 emails, of which 100 (10%) are actually spam. This is a scenario with high class imbalance.

   **Score Distribution Comparison:**

   .. list-table:: Score Distribution by Model
      :header-rows: 1
      :widths: 20 20 20 20 20

      * - Score Range
        - Model A Spam Caught
        - Model A Precision
        - Model B Spam Caught
        - Model B Precision
      * - 0.8-1.0
        - 58
        - 88.3%
        - 52
        - 96.2%
      * - 0.6-0.8
        - 22
        - 68.8%
        - 23
        - 79.3%
      * - 0.4-0.6
        - 12
        - 54.5%
        - 13
        - 62.4%
      * - 0.2-0.4
        - 6
        - 16.7%
        - 8
        - 28.6%
      * - 0.0-0.2
        - 2
        - 0.9%
        - 4
        - 1.6%

   **Precision-Recall Curve Analysis:**
   Our visualization compares both models across the precision-recall space. Key thresholds are highlighted:

   .. list-table:: Key Operating Points
      :header-rows: 1
      :widths: 15 15 20 20 30

      * - Model
        - Threshold
        - Precision
        - Recall
        - Interpretation
      * - A
        - 0.8
        - 88.3%
        - 58.0%
        - High-precision point
      * - B
        - 0.8
        - 96.2%
        - 52.0%
        - High-precision point
      * - A
        - 0.4
        - 54.5%
        - 92.0%
        - High-recall point
      * - B
        - 0.4
        - 62.4%
        - 88.0%
        - High-recall point

   The area under the curve (AUC-PR) for Model A is 0.838 and for Model B is 0.852, indicating that Model B has slightly better overall performance across different threshold settings.

   .. figure:: ../_static/images/evaluation/spam_model_comparison_pr.svg
      :alt: Precision-Recall curve comparing two spam detection models
      :width: 100%
      :name: fig-spam-model-comparison
      
      Precision-Recall curve comparing two spam detection models. Model A (blue) and Model B (orange) show different performance characteristics. Model B maintains higher precision at low to medium recall levels, while Model A performs slightly better at very high recall values.

   **Business Implications:**
   The comparison between the two models reveals important performance differences:

   * **Model B excels in high-precision scenarios**: At the 0.8 threshold, Model B significantly outperforms Model A in precision (96.2% vs 88.3%) although with slightly lower recall (52.0% vs 58.0%). This makes Model B substantially more suitable for contexts where false positives are particularly costly.
   * **Model B maintains precision advantage at medium recall levels**: In the mid-range of recall values (around 0.4-0.7), Model B maintains a precision advantage, indicating better discrimination ability in this common operational range.
   * **Model A performs better at very high recall**: At the highest recall levels (above 0.9), Model A begins to outperform Model B in precision, which might be preferred in scenarios where catching every possible spam email is the absolute priority.

   **Operational Considerations:**
   The choice between models depends on business priorities:

   * **Business email services** would strongly prefer Model B due to its superior precision at standard operating thresholds, minimizing the risk of legitimate business communications being marked as spam
   * **Security-focused environments** would benefit from Model B's higher precision in the typical working range, reducing false alarms
   * **Consumer email services** might still consider Model B because its precision significantly outperforms at a reasonable 80% recall level.
   * **Tiered filtering systems** might use Model B for primary classification and Model A as a secondary review mechanism for maximum recovery of more potential spam emails.

.. admonition:: Code: AUC-PR Calculation
   :class: code-grey

   In practice, **Area Under the Precision-Recall Curve (AUC-PR)** is typically computed by:

   1. **Sorting predictions by their scores** (from highest to lowest).  
   2. **Iterating over possible thresholds** defined by each unique score to calculate the corresponding precision and recall values.  
   3. **Plotting precision (y-axis) against recall (x-axis)**, then using numerical integration to compute the area under the curve.  

   Most machine learning libraries handle these steps internally. For instance, in Python’s scikit-learn, you can use:

   .. code-block:: python

      from sklearn.metrics import average_precision_score

      y_true = [...]       # Ground truth (0/1) labels
      y_scores = [...]     # Model outputs (continuous probabilities or scores)

      auc_pr = average_precision_score(y_true, y_scores)
      print(auc_pr)

   Under the hood, ``average_precision_score``:
   
   * Sorts examples by their predicted score.
   * Computes a “precision-recall” pair at each threshold.
   * Integrates (using the `trapezoidal rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`_) to approximate the area under the PR curve.


Receiver Operating Characteristic (ROC)
---------------------------------------

True Negative Sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~

Although PR curve offers a nuanced threshold-agnostic view in imbalanced datasets, it ignores true negatives entirely. This is generally acceptable or even desirable for highly imbalanced problems, where the focus is on performance for the minority (positive) class. However, this could be a problem when

1. :ub:`Classes are balanced or moderately imbalanced`.
   If your positive and negative classes are roughly of similar size — or at least not extremely skewed — ignoring true negatives can cause you to lose insight into how well the model rejects negative cases.
2. :ub:`Costs of false negatives are similar to or even higher than false positives`.
   When missing a positive (false negative, Type II error) carries a cost comparable to incorrectly flagging a negative (false positive, type I error), you need a measure that balances both Type I/II errors. Because the PR curve disregards true negatives (TN), it cannot show how many false positives arise from all negative examples.
3. :ub:`You want a global view of how the model/system separates classes`.
   The PR curve focuses on performance within the minority class and may obscure information about how the model treats the majority (negative) class — particularly relevant when class distribution is not extremely skewed or both classes matter equally.
4. :ub:`The problem iself is inherently low precision, such as the click-through rate in Ads campaigns`.
   In certain settings—like advertising campaigns that must target a large fraction of users—precision remains low across all thresholds and models. This “crushes” the PR curve near the bottom, offering limited insight into performance trade-offs.


.. admonition:: Example 1: Fraud Detection (High-Stake False Negatives)
   :class: example-green

   Consider a fraud detection system where 1% of 1,000 transactions are fraudulent (i.e., 10 positive cases). Two models are compared:

   * **Model A**: Flags 50 transactions, correctly identifying 8 frauds (TP = 8, FP = 42). This model misses 2 fraudulent transactions (FN = 2).
   * **Model B**: Flags 20 transactions, correctly identifying 6 frauds (TP = 6, FP = 14). This model misses 4 fraudulent transactions (FN = 4).

   Their F1 scores are:
   * Model A: Precision = 8/50 = 0.16, Recall = 8/10 = 0.80, F1 ≈ 0.27
   * Model B: Precision = 6/20 = 0.30, Recall = 6/10 = 0.60, F1 ≈ 0.40

   Although Model B achieves a higher F1 score (0.40 versus 0.27) due to its improved precision, it comes at the cost of a lower recall — meaning it misses more fraudulent transactions (4 missed cases instead of 2). In high-stakes environments where missing a fraudulent transaction carries significant financial or legal consequences, this trade-off is critical. This example demonstrates that while F1 or PR curves might suggest better performance by Model B, they fail to fully capture the impact of high-stake false negatives, underscoring the need for evaluation metrics that consider the overall balance of errors, including true negatives.

.. admonition:: Example 2: Online Advertising
   :class: example-green

   Consider an online advertising system where typical click-through rates are around 2%. In a pool of 1,000,000 users, suppose 20,000 are potential clickers. Two advertising models are evaluated:

   * **Model A**: Shows ads to 10,000 users, resulting in 500 clicks (TP = 500, FN = 1,500).  
     - Precision = 500/10,000 = 0.05  
     - Recall = 500/2,000 = 0.25  
     - F1 ≈ 0.083
   * **Model B**: Shows ads to 100,000 users, resulting in 1,800 clicks (TP = 1,800, FN = 200).  
     - Precision = 1,800/100,000 = 0.018  
     - Recall = 1,800/2,000 = 0.90  
     - F1 ≈ 0.035

   Despite Model A achieving a much higher recall (90%) by reaching more potential clickers, its precision remains extremely low because it shows ads to a vast number of uninterested users. This scenario—where precision is inherently low across all thresholds—illustrates point (4): the PR curve is “crushed” near the bottom and offers limited insight into performance trade-offs.

.. admonition:: Example 3: Sentiment Analysis (Balanced Scenario)
   :class: example-green

   Consider a sentiment analysis system with a balanced dataset of 5,000 positive and 5,000 negative reviews. Two models, **Model A** and **Model B**, are evaluated based on their classification of positive reviews:

   * **Model A**:
     - True Positives (TP): 4,500 (Recall = 90%)
     - False Positives (FP): 500, so Precision = 4,500 / (4,500 + 500) = 90%
     - True Negatives (TN): 4,500
   * **Model B**:
     - True Positives (TP): 4,500 (Recall = 90%)
     - False Positives (FP): 800, so Precision = 4,500 / (4,500 + 800) ≈ 84.9%
     - True Negatives (TN): 4,200

   The PR curve would indeed show that Model B has lower precision compared to Model A, reflecting its higher FP count. However, the PR curve only tells you that precision is lower without indicating the broader impact: in Model B, those extra 300 false positives come from misclassifying a significant fraction of the negative reviews.


In all these examples, metrics that account for true negatives (like AUC-ROC) provide critical additional insights by considering the false positive rate across different thresholds. This enables a more comprehensive evaluation of model performance and supports more informed business decisions in scenarios where class balance, cost trade-offs, global separation, or inherently low precision are important considerations.


Threshold Dependency
~~~~~~~~~~~~~~~~~~~~

Similar to what motivated :refconcept:`Precision-Recall Curve`, we need for a threshold-agonostic evaluation metric to peform overall evaluation of multiple models/systems, and need to considers :ul:`both positive and negative classes` in its calculation.

The ROC Curve
~~~~~~~~~~~~~

:newconcept:`ROC curve` (Receiver Operating Characteristic) is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold varies. The ROC curve was first developed during World War II for radar signal detection before finding applications now in ML/AI systems.

.. note::
   The term "**Receiver Operating Characteristic**" originates from its early development during World War II, where it was used to evaluate the performance of radar receivers in detecting enemy aircraft, specifically how well a "receiver operator" could distinguish between actual signals and background noise; hence, the "receiver" part of the name refers to the radar receiver, and "operating characteristic" describes how well it functioned under different conditions. 

The ROC curve plots two parameters:
* :refconcept:`True Positive Rate (TPR)` or :refconcept:`Recall` on the y-axis
* :refconcept:`False Positive Rate (FPR)` on the x-axis

While both ROC and PR curves provide threshold-agnostic evaluation of ML/AI systems, they emphasize different aspects of model performance. Understanding the key differences helps select the appropriate curve for specific use cases:

.. list-table:: Key Differences Between ROC and PR Curves
   :header-rows: 1

   * - Aspect
     - ROC Curve
     - PR Curve
   * - Axes
     - TPR (y-axis) vs FPR (x-axis)
     - Precision (y-axis) vs Recall (x-axis)
   * - Incorporates TN
     - Yes (in FPR denominator)
     - No
   * - Class Imbalance Sensitivity
     - Less sensitive; can be misleadingly optimistic with severe imbalance
     - Highly sensitive; directly affected by class imbalance
   * - Visualization Focus
     - Overall discriminative ability
     - Focus on positive class performance
   * - Baseline
     - Diagonal line (y=x) represents random classifier
     - Horizontal line at y=P/(P+N) represents random classifier
   * - Preferred Use Cases
     - Balanced datasets; both classes matter equally; FPR not crushed
     - Imbalanced datasets; focus on minority class; Precision not crushed

.. admonition:: Example : Credit Scoring Model
   :class: example-green

   **Scenario:**
   Our ROC curve illustrates a credit scoring model that assigns a score between 0 and 1 to customers, with higher scores indicating lower likelihood of default. The model is evaluated on 200 customers, of which 50 (25%) actually defaulted.

   **Data Distribution:**
   The distribution of scores shows the model's ability to separate defaulting and non-defaulting customers:

   .. list-table:: Score Distribution by Customer Type
      :header-rows: 1
      :widths: 25 25 25 25

      * - Score Range
        - Defaulted Customers
        - Good Customers
        - Precision
      * - 0.8-1.0
        - 0
        - 39
        - 0.0%
      * - 0.6-0.8
        - 13
        - 49
        - 21.0%
      * - 0.4-0.6
        - 11
        - 40
        - 21.6%
      * - 0.2-0.4
        - 18
        - 22
        - 45.0%
      * - 0.0-0.2
        - 8
        - 0
        - 100.0%

   **ROC Curve Analysis:**
   Our visualization shows how varying the threshold creates different points on the ROC curve. Three key thresholds are highlighted:

   .. list-table:: Selected Points on the ROC Curve
      :header-rows: 1
      :widths: 20 20 20 20 20

      * - Score Threshold
        - TPR (Recall)
        - FPR
        - Precision
        - Interpretation
      * - 0.7
        - 0.21
        - 0.20
        - 25.0%
        - Conservative (red point)
      * - 0.5
        - 0.50
        - 0.40
        - 28.6%
        - Balanced (green point)
      * - 0.3
        - 0.85
        - 0.80
        - 26.6%
        - Aggressive (orange point)

   The area under the curve (AUC) is 0.767, indicating good discrimination ability. The shaded blue region in our visualization represents this area.

   .. figure:: ../_static/images/evaluation/credit_scoring_roc.svg
      :alt: ROC curve for a credit scoring model showing different threshold points
      :width: 100%
      :name: fig-credit-scoring-roc
      
      ROC curve for a credit scoring model. The curve shows how TPR and FPR change as the classification threshold varies from high (0.7, conservative) to low (0.3, aggressive). The diagonal dashed line represents random guessing performance. The "High Precision Region" generally contains points with better precision but lower recall, while the "Low Precision Region" offers higher recall at the cost of precision.

   **Business Implications:**
   The three threshold points on our curve represent different business strategies:
   
   * **Conservative approach (t=0.7):** The red point shows a TPR of 0.2 and FPR of 0.2. The bank would approve most loans but miss detecting 80% of defaulters. This maximizes loan volume but increases default-related losses.
   * **Balanced approach (t=0.5):** The green point shows a TPR of 0.5 and FPR of 0.4. This middle-ground catches half the defaulters while maintaining reasonable precision.
   * **Aggressive approach (t=0.3):** The orange point shows a TPR of 0.85 and FPR of 0.8. This conservative lending strategy catches most defaulters but also rejects 80% of good customers. This minimizes default losses but significantly reduces loan volume.

   **Operational Decisions:**
   The optimal threshold depends on business considerations not visible in the ROC curve itself:
   
   * The cost ratio between false negatives (approving defaults) and false positives (rejecting good loans)
   * Regulatory requirements for risk management
   * Overall lending volume targets and risk appetite

   Our visualization demonstrates why threshold-independent metrics like AUC-ROC are valuable for model evaluation, while the specific operating point selection remains a critical business decision requiring additional context beyond model performance.


AUC-ROC
~~~~~~~

:newconcept:`AUC-ROC` (Area Under the ROC Curve) has a nice probabilistic interpretation. It represents the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance. It is a single scalar value between 0 and 1 that quantifies the overall discriminative ability of a ML/AI system independent of any specific threshold.

.. math::

   \text{AUC-ROC} = P(S_{\text{positive}} > S_{\text{negative}})

Where $S_{\\text{positive}}$ is the score for a positive instance, and $S_{\\text{negative}}$ is the score for a negative instance.

.. admonition:: Mathematical Proof: AUC-ROC as a Probability
   :class: example-green

   **Theorem:**
   The Area Under the ROC Curve (AUC-ROC) equals the probability that a randomly chosen positive instance receives a higher score than a randomly chosen negative instance.
   
   **Mathematical Statement:**
   
   .. math::
      \text{AUC-ROC} = P(S_{\text{positive}} > S_{\text{negative}})
   
   Where:
   - :math:`S_{\text{positive}}` is the score assigned by the model to a randomly selected positive instance
   - :math:`S_{\text{negative}}` is the score assigned by the model to a randomly selected negative instance
   
   **Proof:**
   
   Let's define:
   - :math:`X_1, X_2, \ldots, X_m` as the scores for the :math:`m` positive instances
   - :math:`Y_1, Y_2, \ldots, Y_n` as the scores for the :math:`n` negative instances
   
   The ROC curve is constructed by varying a threshold :math:`t` and plotting the resulting (FPR, TPR) pairs:
   
   .. math::
      \text{TPR}(t) = \frac{1}{m}\sum_{i=1}^{m} \mathbf{1}(X_i > t)
   
   .. math::
      \text{FPR}(t) = \frac{1}{n}\sum_{j=1}^{n} \mathbf{1}(Y_j > t)
   
   Where :math:`\mathbf{1}(\cdot)` is the indicator function that equals 1 when its argument is true and 0 otherwise.
   
   The AUC-ROC is the integral of the TPR with respect to the FPR:
   
   .. math::
      \text{AUC-ROC} = \int_{0}^{1} \text{TPR}(t) \, d(\text{FPR}(t))
   
   To compute this integral in practice, we don't use a continuous set of thresholds. Instead, we use the unique score values in our dataset as thresholds, because TPR and FPR only change when the threshold crosses an actual data point. We construct the ROC curve by sorting all instances by their scores and moving the threshold from highest to lowest.
   
   :ul:`Key Step: Examining Threshold Movement`
   
   Let's examine what happens when we move the threshold past a specific negative instance with score :math:`Y_j` (assuming :math:`Y_j \ne X_i, i=1, ..., m`):
   
   1. We're essentially setting the threshold to be just below :math:`Y_j` (i.e., :math:`Y_j - \epsilon`, where :math:`\epsilon` is infinitesimally small).
   
   2. All instances with scores :math:`> Y_j` were already classified as positive, and now we're additionally classifying the instance with score :math:`Y_j` as positive.
   
   3. Since :math:`Y_j` is the score of a negative instance, this movement increases our false positive count by 1, which increases the FPR by :math:`\frac{1}{n}`.
   
   4. The TPR at this threshold represents the fraction of positive instances with scores greater than :math:`Y_j`:
   
      .. math::
         \text{TPR}(Y_j) = \frac{1}{m}\sum_{i=1}^{m} \mathbf{1}(X_i > Y_j)
   
   5. The area added to our AUC calculation is a small rectangle with:
      
      * Width = :math:`\frac{1}{n}` (the change in FPR)
      * Height = TPR(:math:`Y_j`) (the current TPR value)
   
   Thus, the area added is:
   
   .. math::
      \Delta A_{\text{neg}} = \text{TPR}(Y_j) \cdot \frac{1}{n}
   
   :ul:`Handling Ties`
   
   For the case where multiple instances (both positive and negative) have the same score, we need to be more careful. If :math:`Y_j = X_i` for some instances, then a most general approach is to assign a weight of 0.5 to tied positive-negative pairs.
   
   Using the third approach, we modify our calculation to:
   
   .. math::
      \text{TPR}(Y_j) = \frac{1}{m}\sum_{i=1}^{m} \left[ \mathbf{1}(X_i > Y_j) + \frac{1}{2}\mathbf{1}(X_i = Y_j) \right]
   
   :ul:`Summing Over All Instances`
   
   Summing over all negative instances, the total area becomes:
   
   .. math::
      \text{AUC-ROC} = \sum_{j=1}^{n} \left( \frac{1}{m}\sum_{i=1}^{m} \left[ \mathbf{1}(X_i > Y_j) + \frac{1}{2}\mathbf{1}(X_i = Y_j) \right] \right) \cdot \frac{1}{n}
   
   This simplifies to:
   
   .. math::
      \text{AUC-ROC} = \frac{1}{mn}\sum_{j=1}^{n}\sum_{i=1}^{m} \left[ \mathbf{1}(X_i > Y_j) + \frac{1}{2}\mathbf{1}(X_i = Y_j) \right]
   
   This expression counts:
   
   * 1 for each pair where the positive instance has a higher score than the negative instance
   * 0.5 for each pair where the positive and negative instances have equal scores
   * Divided by the total number of positive-negative pairs (:math:`mn`)
   
   Therefore:
   
   .. math::
      \text{AUC-ROC} = P(S_{\text{positive}} > S_{\text{negative}}) + \frac{1}{2}P(S_{\text{positive}} = S_{\text{negative}})
   
   When there are no ties (as is often the case with continuous scores), this simplifies to:
   
   .. math::
      \text{AUC-ROC} = P(S_{\text{positive}} > S_{\text{negative}})

.. note::
   Interpreting AUC-ROC:

   * AUC = 1.0: Perfect classification (ideal model)
   * 0.9 ≤ AUC < 1.0: Excellent classification
   * 0.8 ≤ AUC < 0.9: Good classification
   * 0.7 ≤ AUC < 0.8: Fair classification
   * 0.6 ≤ AUC < 0.7: Poor classification
   * 0.5 ≤ AUC < 0.6: Failed classification (little better than random)
   * AUC = 0.5: Random classification (no discriminative power)
   * AUC < 0.5: Worse than random guessing (suggests inverted predictions)

.. admonition:: Code: AUC-ROC Calculation
   :class: code-grey

   In practice, **Area Under the ROC Curve (AUC-ROC)** is typically computed using standard machine learning libraries. In Python's scikit-learn, you can use:

   .. code-block:: python

      from sklearn.metrics import roc_auc_score

      y_true = [...]       # Ground truth (0/1) labels
      y_scores = [...]     # Model outputs (continuous probabilities or scores)

      auc_roc = roc_auc_score(y_true, y_scores)
      print(auc_roc)

   Under the hood, ``roc_auc_score``:
   
   * Sorts examples by their predicted score.
   * Computes TPR and FPR pairs at each possible threshold.
   * Integrates (using the `trapezoidal rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`_) to approximate the area under the ROC curve.

Micro and Macro Averaging
-------------------------

A :refconcept:`dataset` might be divided into multiple groups (e.g., by classes in multi-class classification; by users, categories, etc.), then each group can have its own :refconcept:`precision` and :refconcept:`recall` metrics, and we need to aggregate these metrics to measure overall system performance. There are two common approaches: micro-averaging and macro-averaging.

Macro-Precision and Macro-Recall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Macro-averaging` calculates metrics individually for each group and then simply takes the average of these metrics.

.. math::

   \text{Macro-Precision} = \frac{1}{n} \sum_{i=1}^{n} \text{Precision}_i

.. math::

   \text{Macro-Recall} = \frac{1}{n} \sum_{i=1}^{n} \text{Recall}_i

.. math::

   \text{Macro-F1} = \frac{1}{n} \sum_{i=1}^{n} \text{F1}_i

Alternatively, Macro-F1 can be calculated as the harmonic mean of Macro-Precision and Macro-Recall:

.. math::

   \text{Macro-F1} = 2 \cdot \frac{\text{Macro-Precision} \cdot \text{Macro-Recall}}{\text{Macro-Precision} + \text{Macro-Recall}}

Micro-Precision and Micro-Recall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Micro-averaging` calculates metrics by aggregating the individual true positives, false positives, and false negatives across all groups and then calculating the metrics.

.. math::

   \text{Micro-Precision} = \frac{\sum_{i=1}^{n} \text{TP}_i}{\sum_{i=1}^{n} \text{TP}_i + \sum_{i=1}^{n} \text{FP}_i}

.. math::

   \text{Micro-Recall} = \frac{\sum_{i=1}^{n} \text{TP}_i}{\sum_{i=1}^{n} \text{TP}_i + \sum_{i=1}^{n} \text{FN}_i}

.. math::

   \text{Micro-F1} = 2 \cdot \frac{\text{Micro-Precision} \cdot \text{Micro-Recall}}{\text{Micro-Precision} + \text{Micro-Recall}}

Micro-averaging gives equal weight to each item, which means a group with more items (e.g., active users, popular queries) has a greater influence on the final metric.

Micro and Macro AUC
~~~~~~~~~~~~~~~~~~~

When dealing with curve-based metrics like AUC-ROC or AUC-PR across multiple groups, we need to extend our averaging approaches. There are several strategies for computing AUC in multi-group scenarios:

:newconcept:`Macro-AUC` computes the AUC for each group independently and then averages the results:

.. math::

   \text{Macro-AUC} = \frac{1}{n} \sum_{i=1}^{n} \text{AUC}_i

This gives equal importance to each group regardless of size, which can be desirable when each group is equally important from a business perspective.

:newconcept:`Micro-AUC` approaches, on the other hand, can be implemented in two main ways:

1. **Pooling Method**: Combine all instances from all groups into a single pool, then compute a single AUC on this pooled data:

   .. math::
   
      \text{Micro-AUC (Pooled)} = \text{AUC}(\text{all pooled instances})

   This method effectively weights each instance equally, giving more influence to larger groups.

2. **Threshold-wise Method**: For each threshold value, compute micro-averaged TPR and FPR (or precision and recall for AUC-PR) across all groups, then compute the AUC from these averaged curves:

   .. math::
   
      \text{Micro-TPR}(t) = \frac{\sum_{i=1}^{n} \text{TP}_i(t)}{\sum_{i=1}^{n} \text{TP}_i(t) + \sum_{i=1}^{n} \text{FN}_i(t)}
   
   .. math::
   
      \text{Micro-FPR}(t) = \frac{\sum_{i=1}^{n} \text{FP}_i(t)}{\sum_{i=1}^{n} \text{FP}_i(t) + \sum_{i=1}^{n} \text{TN}_i(t)}

   The AUC is then calculated using these micro-averaged curves:

   .. math::
   
      \text{Micro-AUC (Threshold-wise)} = \int_{0}^{1} \text{Micro-TPR}(\text{Micro-FPR}^{-1}(x)) \, dx

.. admonition:: Example: Micro vs. Macro AUC in Multi-Class Classification
   :class: example-green

   Consider a multi-class classification problem with 3 classes (A, B, C) where we've trained a one-vs-rest classifier. We evaluate the model for each class:
   
   +-------+------------+---------+
   | Class | Class Size | AUC-ROC |
   +=======+============+=========+
   | A     | 1000       | 0.95    |
   +-------+------------+---------+
   | B     | 200        | 0.85    |
   +-------+------------+---------+
   | C     | 50         | 0.75    |
   +-------+------------+---------+
   
   **Macro-AUC-ROC** calculation:
   
   .. math::
      
      \text{Macro-AUC-ROC} = \frac{0.95 + 0.85 + 0.75}{3} = 0.85
   
   This gives equal weight to each class's performance, regardless of its frequency in the dataset.
   
   **Micro-AUC-ROC** (Pooled) would consider all 1250 instances together, essentially weighting by class frequency. This would heavily favor performance on class A which represents 80% of the data.

Multi-Class Extension
---------------------

Strategies
~~~~~~~~~~

When extending binary classification metrics to multi-class scenarios, :newconcept:`One-vs-Rest (OVR)` is the most common approach for extending binary metrics to multi-class scenarios:

* For each class :math:`c`, a separate binary classification problem is created
* Class :math:`c` is treated as the "positive" class
* All other classes are combined and treated as the "negative" class
* Metrics are calculated for each binary problem, then averaged using micro or macro approaches
* This approach is specified with ``multi_class='ovr'`` in scikit-learn

The other strategy :newconcept:`One-vs-One (OVO)` creates a binary classifier for each pair of classes:

* For :math:`n` classes, :math:`\frac{n(n-1)}{2}` binary classifiers are trained, and each classifier can be more specialized
* Each classifier distinguishes between just two classes
* The final classification is typically determined by "voting"
* This approach is specified with ``multi_class='ovo'`` in scikit-learn
* Generally more computationally expensive than OVR, but can perform better on some problems because each classifier is more specialized

.. admonition:: Code: Computing Micro and Macro Metrics For Multi-Class Classification
   :class: code-grey

   Here's how to compute micro and macro metrics using scikit-learn for multi-class classification:

   .. code-block:: python

      from sklearn.metrics import roc_auc_score, average_precision_score
      from sklearn.metrics import precision_score, recall_score, f1_score
      from sklearn.preprocessing import label_binarize
      import numpy as np
      
      # Assume y_true contains class labels (0, 1, 2, ...) 
      # y_score is of shape (n_samples, n_classes) containing probability scores
      # y_pred is of shape (n_samples,) containing predicted class labels
      
      # Binarize the labels for one-vs-rest evaluation
      classes = np.unique(y_true)
      y_true_bin = label_binarize(y_true, classes=classes)
      
      # 1. AUC calculations
      # ------------------
      
      # Macro AUC-ROC (average AUC for each class)
      # "OVR" stands for "One-vs-Rest" (sometimes also called "One-vs-All" or OVA), which is a strategy 
      # for extending binary classification metrics to multi-class scenarios.
      macro_auc_roc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')
      
      # Micro AUC-ROC (compute TPR/FPR across all classes, then calculate AUC)
      micro_auc_roc = roc_auc_score(y_true_bin, y_score, average='micro', multi_class='ovr')
      
      print(f"Macro-averaged AUC-ROC: {macro_auc_roc:.3f}")
      print(f"Micro-averaged AUC-ROC: {micro_auc_roc:.3f}")
      
      # For PR AUC, we can use average_precision_score
      macro_auc_pr = average_precision_score(y_true_bin, y_score, average='macro')
      micro_auc_pr = average_precision_score(y_true_bin, y_score, average='micro')
      
      print(f"Macro-averaged AUC-PR: {macro_auc_pr:.3f}")
      print(f"Micro-averaged AUC-PR: {micro_auc_pr:.3f}")
      
      # 2. Precision, Recall, and F1 Score calculations
      # ----------------------------------------------
      
      # For precision, recall, and F1, we need predicted labels (not scores)
      # If you only have scores, you need to convert them to predictions first:
      # y_pred = np.argmax(y_score, axis=1)
      
      # Macro-averaging (compute metrics for each label, then average)
      macro_precision = precision_score(y_true, y_pred, average='macro')
      macro_recall = recall_score(y_true, y_pred, average='macro')
      macro_f1 = f1_score(y_true, y_pred, average='macro')
      
      print(f"Macro-averaged Precision: {macro_precision:.3f}")
      print(f"Macro-averaged Recall: {macro_recall:.3f}")
      print(f"Macro-averaged F1: {macro_f1:.3f}")
      
      # Micro-averaging (aggregate TP, FP, FN across all classes, then calculate metrics)
      micro_precision = precision_score(y_true, y_pred, average='micro')
      micro_recall = recall_score(y_true, y_pred, average='micro')
      micro_f1 = f1_score(y_true, y_pred, average='micro')
      
      print(f"Micro-averaged Precision: {micro_precision:.3f}")
      print(f"Micro-averaged Recall: {micro_recall:.3f}")
      print(f"Micro-averaged F1: {micro_f1:.3f}")
      
      # 3. Class-specific metrics
      # -----------------------
      
      # Sometimes it's useful to see metrics for each class individually
      class_precision = precision_score(y_true, y_pred, average=None)
      class_recall = recall_score(y_true, y_pred, average=None)
      class_f1 = f1_score(y_true, y_pred, average=None)
      
      # Print metrics for each class
      for i, cls in enumerate(classes):
          print(f"Class {cls}:")
          print(f"  Precision: {class_precision[i]:.3f}")
          print(f"  Recall: {class_recall[i]:.3f}")
          print(f"  F1 Score: {class_f1[i]:.3f}")
      
      # 4. Weighted averaging (alternative that accounts for class imbalance)
      # ------------------------------------------------------------------
      
      # Weighted averaging - weights metrics by class frequency
      weighted_precision = precision_score(y_true, y_pred, average='weighted')
      weighted_recall = recall_score(y_true, y_pred, average='weighted')
      weighted_f1 = f1_score(y_true, y_pred, average='weighted')
      
      print(f"Weighted-averaged Precision: {weighted_precision:.3f}")
      print(f"Weighted-averaged Recall: {weighted_recall:.3f}")
      print(f"Weighted-averaged F1: {weighted_f1:.3f}")
      
      # 5. Computing a confusion matrix for additional insights
      # ----------------------------------------------------
      
      from sklearn.metrics import confusion_matrix
      import matplotlib.pyplot as plt
      import seaborn as sns
      
      # Generate the confusion matrix
      cm = confusion_matrix(y_true, y_pred)
      
      # Optional: Normalize by row (true labels) to show recall
      cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      
      # Visualize the confusion matrix
      plt.figure(figsize=(10, 8))
      sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                  xticklabels=classes, yticklabels=classes)
      plt.xlabel('Predicted Label')
      plt.ylabel('True Label')
      plt.title('Normalized Confusion Matrix')
      plt.show()

Multi-Class Confusion Matrix
----------------------------

In multi-class classification, instead of just positive and negative classes, we have multiple classes (e.g., Class A, Class B, Class C). The confusion matrix extends to an :math:`n \times n` structure, where each row represents the actual class and each column represents the predicted class.

.. list-table:: Example Multi-Class Confusion Matrix (3 Classes)
   :header-rows: 1
   :widths: 15 20 20 20 20

   * - **Actual / Predicted**
     - **Class A**
     - **Class B**
     - **Class C**
     - **Total (Actual)**
   * - **Class A**
     - True Positives (TP_A)
     - False Negative (FN_A→B)
     - False Negative (FN_A→C)
     - P_A = TP_A + FN_A→B + FN_A→C
   * - **Class B**
     - False Negative (FN_B→A)
     - True Positives (TP_B)
     - False Negative (FN_B→C)
     - P_B = TP_B + FN_B→A + FN_B→C
   * - **Class C**
     - False Negative (FN_C→A)
     - False Negative (FN_C→B)
     - True Positives (TP_C)
     - P_C = TP_C + FN_C→A + FN_C→B
   * - **Total (Predicted)**
     - PP_A = TP_A + FP_A
     - PP_B = TP_B + FP_B
     - PP_C = TP_C + FP_C
     - Total = P_A + P_B + P_C

Where:

- **True Positives (TP_x)**: Correct predictions for Class X.
- **False Negatives (FN_x→y)**: Instances of Class X misclassified as Class Y.
- **False Positives (FP_x)**: Sum of Class Y and Class Z instances misclassified as Class X.
- **P_x**: Total actual instances of Class X.
- **PP_x**: Total predicted instances of Class X.
- **Total**: Total number of samples

.. admonition:: Example: Concrete Multi-Class Confusion Matrix Example
   :class: example-green

   Suppose we have a weather classification system that predicts three possible weather conditions based on satellite imagery:
   
   - **Class A: Sunny**
   - **Class B: Cloudy**
   - **Class C: Raining/Snowing**

   The model makes predictions based on historical weather data, and we evaluate its performance using a confusion matrix.

   .. list-table:: Example Multi-Class Confusion Matrix (Weather Classification)
      :header-rows: 1
      :widths: 15 20 20 20 20

      * - **Actual / Predicted**
        - **Sunny (Class A)**
        - **Cloudy (Class B)**
        - **Raining/Snowing (Class C)**
        - **Total (Actual)**
      * - **Sunny (Class A)**
        - True Positives (TP_A) = 45
        - False Negative (FN_A→B) = 8
        - False Negative (FN_A→C) = 2
        - P_A = 45 + 8 + 2 = 55
      * - **Cloudy (Class B)**
        - False Negative (FN_B→A) = 7
        - True Positives (TP_B) = 50
        - False Negative (FN_B→C) = 10
        - P_B = 7 + 50 + 10 = 67
      * - **Raining/Snowing (Class C)**
        - False Negative (FN_C→A) = 3
        - False Negative (FN_C→B) = 9
        - True Positives (TP_C) = 40
        - P_C = 3 + 9 + 40 = 52
      * - **Total (Predicted)**
        - PP_A = TP_A + FP_A = 45 + 10 = 55
        - PP_B = TP_B + FP_B = 50 + 17 = 67
        - PP_C = TP_C + FP_C = 40 + 12 = 52
        - Total = P_A + P_B + P_C = 55 + 67 + 52 = 174

   **Observations:**
   - The system performs well for **Cloudy** weather, but it misclassifies **Sunny** conditions as **Cloudy** 8 times.
   - **Raining/Snowing** has the highest number of false negatives (misclassified as Cloudy 9 times), which could be problematic in applications like weather forecasting for travel safety.
   - The system confuses **Sunny and Cloudy** more often than **Sunny and Raining/Snowing**, likely due to the visual similarities in cloud cover.

   This confusion matrix provides a structured way to evaluate model errors and potential areas for improvement, such as **better feature differentiation between Cloudy and Raining/Snowing conditions**.

Summary
-------

This chapter explored essential evaluation metrics for machine learning and AI systems, focusing on their applications, trade-offs, and interpretations. Here's a comprehensive summary of the key concepts:

Fundamental Metrics
~~~~~~~~~~~~~~~~~~

* **Basic Components**: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).

* **Core Metrics**:

  * **Precision** (TP/(TP+FP)): Measures the proportion of recommended items that are actually relevant
  * **Recall** (TP/(TP+FN)): Measures the proportion of relevant items that were successfully recommended
  * **F1 Score**: Harmonic mean of precision and recall, balancing both considerations
  * **False Positive Rate (FPR)** (FP/(FP+TN)): Proportion of irrelevant items incorrectly recommended

Threshold Dependency
~~~~~~~~~~~~~~~~~~~~

* **Trade-offs**:

  * Higher thresholds → Increased precision, decreased recall
  * Lower thresholds → Increased recall, decreased precision
  * Different applications prioritize different sides of this trade-off

* **Challenges with Single-Threshold Evaluation**:

  * Business contexts require different operating thresholds
  * Optimal thresholds on training data may not generalize
  * Different models perform optimally at different thresholds

Threshold-Agnostic Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Precision-Recall Curves**:

  * Plot precision vs. recall across all thresholds
  * AUC-PR: Area Under the Precision-Recall Curve
  * Best for imbalanced datasets and when precision is critical
  * Focuses on performance on positive class

* **ROC Curves**:

  * Plot TPR vs. FPR across all thresholds
  * AUC-ROC: Area Under the ROC Curve
  * Probabilistic interpretation: P(S_positive > S_negative)
  * Best for balanced datasets and when both classes matter equally

* **Comparative Analysis**:

  * PR curves are more sensitive to imbalanced datasets
  * ROC curves incorporate true negatives through FPR
  * PR curves have variable baseline depending on class distribution
  * ROC curves have fixed baseline (diagonal line)

Multi-Group Evaluation
~~~~~~~~~~~~~~~~~~~~~~

* **Macro-Averaging**:

  * Calculate metrics individually for each group, then average
  * Gives equal weight to each group regardless of size
  * Macro-F1: Either average of individual F1 scores or harmonic mean of macro-precision and macro-recall

* **Micro-Averaging**:

  * Aggregate TP, FP, FN across all groups, then calculate metrics
  * Gives more weight to groups with more samples
  * Particularly useful for imbalanced group distributions

Application-Specific Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Precision-Focused Applications**:

  * Medical diagnosis, content filtering
  * High cost of false positives (Type I error)
  * Metrics: Precision, AUC-PR

* **Recall-Focused Applications**:

  * Fraud detection, disease screening
  * High cost of false negatives (Type II error)
  * Metrics: Recall, AUC-RP

* **Balance-Focused Applications**:

  * General recommendation systems
  * Similar costs for both types of errors
  * Metrics: F1 score, AUC-ROC

Best Practices
~~~~~~~~~~~~~~

* **Metric Selection**:

  * Choose metrics aligned with business objectives
  * Consider class distribution (balanced vs. imbalanced)
  * Evaluate across multiple operating thresholds

* **Comprehensive Evaluation**:

  * Use both threshold-specific and threshold-agnostic metrics
  * Consider both PR and ROC curves for complete picture
  * Incorporate business context and error costs in interpretation

* **Implementation**:

  * Leverage standard libraries for consistent calculation
  * Be cautious of ties in score values when computing AUC
  * Use both micro and macro averaging when evaluating across groups