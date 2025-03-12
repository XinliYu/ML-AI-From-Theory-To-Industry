Data Preparation
================

In ML/AI systems for search, recommendation, and advertising, effective data preparation for offline model development is crucial. The retrieval system relies on various inputs, including user queries, user data, contextual signals and runtime analytics. Properly collecting, labeling, and balancing this data is critical model performance and user satisfaction.

The runtime system will typically log the following information for offline development (see also `Recommendation ML/AI System Design <../../system_design/recommendation_and_ads_system_design/01_recommendation_system_design.html>`_).

* **User Queries**: In search systems and reactive recommendation or advertising scenarios, user queries are primary inputs. While typically textual, queries can also be in the form of audio, images, or videos (e.g., the Amazon Shopping app allows users to search for products by taking pictures).
* **User Data**: Proactive recommendations and ads heavily depend on user data, such as :refconcept:`User Profiles` and :refconcept:`Historical User Activities & Analytics`. This data is also valuable in search and reactive scenarios to personalize results.
* **Runtime Signals** provide supplementary information that enhances input understanding, including :refconcept:`Context Signals` and :refconcept:`Real-Time User Activities & Analytics`.

In additional to regular runtime user impressions and interactions, runtime exploration experiments can be conducted with intention to collect data.

  * :newconcept:`Exploration Sampling`: Occasionally presenting non-personalized or diverse items to gather feedback outside the current model's preferences.
  * :refconcept:`A/B Testing`: A/B testing to collect datapoint regarding different system variations.
  * :refconcept:`Multi-Armed Bandit` Exploration: Using multi-armed bandit approaches to collect data on unexplored items/users.

Constructing datasets from offline logs involves extracting relevant features and accurately labeling data for effective model development. Specifically, it's essential to annotate whether a candidate is relevant (binary label) or assign graded relevance (categorical/numeric label). Accurate labeling ensures that models learn to distinguish between relevant and irrelevant items effectively.

The process starts with **collecting implicit feedback**: :newconcept:`Implicit Feedback` refers to data gathered from :refconcept:`User Activities & Analytics`. Examples include page views, click-through rates, time spent on contents (a.k.a. dwell time, or playback time for audio/video contents), navigation patterns (e.g., click path), and strong signals such as cart/purchase/order history.

Session Modeling
----------------

A :newconcept:`Session` is an ordered (typically by time) sequence of user interactions with an ML/AI system, where the interactions are potentially related to each other.

:newconcept:`Session Identification` is a fundamental preprocessing step that sequentially groups user interactions that likely share context or purpose. This grouping provides a consistent framework for downstream models to analyze user behavior and engagement patterns. Several rule-based approaches exist for identifying and defining user sessions:

* :newconcept:`Event-Based Boundaries`: Rough session boundaries can be delineated by strong events.
  
  * :newconcept:`Entry Events`: Actions like beginning website/app visits, initial referral/campaign link clicks (including search referrals), etc., often signal a new session.
  * :newconcept:`Exit Events`: Activities such as logging out, or closing the website/app strongly mark session endings. For robustness against user accidentally logging out, closing website/app, etc.
    
    * A time-based rule can be applied, such as user not re-login or re-opening within a pre-defined time gap (e.g., 1 min)
    * Downstream models can consider previous few sessions from a time window as historical context to maintain continuity when user tasks are interrupted.

* :newconcept:`Time-Based Windowing`: The most widely used approach defines sessions based on a period of inactivity, combined with above event signals.
  
  * Interactions between an entry event and exit event can be divided into sessions if user inactivity (idle time) exceeds a threshold, for example e-commerce (30 minutes), streaming content (1 hour), professional tools (2+ hours), voice assistant (1 min).
  * This approach is simple to implement and works well for many applications, especially when explicitly defined start/end boundaries (like login/logout) are unavailable between sessions.

In reality, a session obtained from above rule-based approaches are usually complex and not streamlined.

* **Parallel Tasks**: Users may conduct multiple tasks simultaneously, blurring session boundaries. For example:
  
  * In e-commerce scenario, user might go shopping in multiple browser tabs, viewing multiple items, adding multiple items to carts, and then checkout altogether.
  * In travel booking scenario, a user researches flights in one tab, hotels in another, and local activities in a third. They might switch between these tasks over hours or days before completing reservations.
  * In social network scenario, user might maintain multiple conversation threads while simultaneously scrolling through feeds, checking notifications, and possibly creating content—all activities that appear interwoven but represent distinct engagement modes.

* **Interrupted Sessions**: External interruptions may cause artificial session breaks, e.g., accidental loging out, closing website/app due to computer slowdown, interruptions by real-life events, etc.
* **Cross-Device Usage**: Users switching between devices (e.g., mobile to desktop) may appear as separate sessions despite continuing the same task.

Rule-based techniques still provide a foundation for handling these complexities:

* Interactions from multiple tabs or devices can be merged into a single session when they occur within a pre-defined time window.
* Downstream search/recommendation/ads models can incorporate previous few sessions from a longer time window as historical context (e.g., up to three sessions from the past 24 hours) to maintain continuity when user tasks are interrupted.

In contemporary systems with sophisticated downstream search/recommendation/ads models, rule-based session identification is typically sufficient despite its limitations. The underlying assumption is that :ub:`tasks performed within a single session (e.g., within a pre-defined time window) share latent relationships`, even if not immediately apparent. To facilitate consitent feature representation and downstream development,

* Session data and analytics should be available for both offline and runtime for downstream use.
* Optionally, an :newconcept:`interaction encoder` can be developed to integrate information for an interaction, potentially: 
  
  * Multi-modal (integrating various information types; see also `Multi-Modal Fusion <../../system_design/recommendation_and_ads_system_design/01_recommendation_system_design.html#multi-modal-fusion>`_)
  * Contextual (considering the whole session for offline, or consdering the interactions up to the last completed one for runtime)
  * Multi-task (learning from major downstream tasks)
  
  The ecodings can serve as input for classic downstream ML, or serve as context tokens in downstream LLM (if the LLM is post-trained to support it).

* Optionally, for LLM, a session information prompt template can be recommended for downstream LLMs to leverage (post-training not required).


Basic Interaction Intent Labeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although sophisticated downstream models could often already discern patterns within noisy and interleaved sessions and effectively utilize this information, :newconcept:`Interaction Analytics` can still offer valuable additional signals for downstream use, not just modeling, but also business analytics, facilitating more structured understanding of user behavior within sessions.

:newconcept:`Basic Interaction Intent Labeling` assigns specific intent or task identifiers to individual interactions, helping to signal more coherent "sub-sessions" within a larger time-based session. Business analytics can leverage these signals to categorize interaction. Downstream models can leverage these additional intent/task labels to improve their modeling of user behavior patterns. While there's no universal requirement for intent/task identifier format, following a consistent structured template or schema is recommended. For example:

* **Template format**: ``{domain}/{intent}/{action}/{item_type1: item1 \| item_type2: item2 \| ...}``. 
  
  * The :newconcept:`domain`, :newconcept:`intent`, :newconcept:`action` components form a complete :newconcept:`intent type` label
  * For single-purpose website/app, "domain" component can be dropped and "intent" component is promoted as the "domain". Sometimes "intent" and "action" are merged as a single label component. Either way, the template can be simplified as ``{domain}/{intent}/{item_type1: item1 \| item_type2: item2 \| ...}``

* **Concrete examples**:

  * ``shopping/exploring/view_promotion/product_id: ABC123|promotion_id: SALE50``
  * ``shopping/subscribe_and_save/cancel/product_id: XYZ789``
  * ``instagram/content_discovery/browse_feed/hashtag: travel``

In most cases, the website/app's internal state information can deterministically provide most of the intent/task label. The currently active page and the user's interaction with UI elements often contain sufficient information to determine the domain, intent, action, product IDs, and other relevant attributes. 

Such labeling provides basic categorical labels for business/modeling applications.

* Downstream search/recommendation/ads models might start simple with only a pre-defined subset of intents highly related to the downstream tasks, or at least exclude certain types of interactions (such as account management activities) from the session context to reduce context size and computational cost
* User Experience Analytics might leverage the intent labels to identify at which point user frictions often occur as feedback to development teams


Interaction Intent Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

However, the basic intent labeling derived from pure rule-based and website/app state-based approaches are limited. There are multiple difficulties, 

* A session might have multiple parallel semantic tasks, and different semantic tasks often have their interactions interleaved in a session. 
  
  * Sometimes one semantic task is conducted on two devices (e.g., mobile and desktop), and time-based merging interaction sequences from different devices make such interleaving issue more serious. This problem significantly limits the application of intent labels in downstream business analytics and model development. 
  * As a concrete example, downstream models can train on a joint task to predict the next intent type to illustrate the model understands user behavior patterns. However, interleaved interaction sequence makes it difficult, as it not making sense to frequently ask the model to predict an ad-hoc next intent type.

* One semantic task might consist of interactions of multiple basic intent types. For example, 
  
  * In ecommerce case, a user might be exploring multiple relevant items before concluding and purchasing a subset of them.
  * In social network case, a user might perform multiple relevant visit_user_home, view_post and comment interactions as part of one exploration.
  * However, because of the interaction interleaving issue, it is difficult to judge relevance of the next interaction based on the simple intent types.

:newconcept:`Interaction Intent Clustering` is a valuable interaction analytics approach to identify and group related interactions into coherent flat or hierarchical clusters, even when activities appear fragmented across the session timeline. For example,

* A sequence of interactions might be clustered as task "buy:kid_clothing".
* This sequence might be interleaved with a parallel purchase task for "buy:kid_drinks".
* Both might be occasionally interrupted by subscription management interactions or other account management activities.

Data Creation
^^^^^^^^^^^^^

The main challenge for learning intent clustering is session data labeling. There are two aspects for this labeling, we need to group interactions, and assign a semantic task label (such as "buy:kid_cloth"). The grouping aspect is more critical. The task label is usually also structured following certain pre-defined schema.

* :ub:`Human Annotation` on real user runtime sessions can provide high-quality labels but is resource-intensive and difficult to scale. This approach is often used to create:

  * Gold standard evaluation datasets to measure model performance
  * Initial training data to bootstrap the modeling process
  * Targeted examples for edge cases and complex interaction patterns
  
  Modern approaches now :ub:`leverage LLMs to significantly reduce human annotation effort`. LLMs can annotate according to instructions and few-shot examples, with multiple LLMs used for cross-validation. Sessions with low annotation confidence can be routed to human annotators for final determination. Additionally, human reviewers can verify a sample of LLM annotations to ensure quality. This hybrid approach enables the creation of substantially larger well-annotated datasets while maintaining quality standards.

* :ub:`Rule-Based` approaches apply domain-specific heuristics to group interactions, providing a more scalable but less flexible alternative to manual annotation. Rules are usually strict to only apply to high-confidence behavior patterns. For example,
  
  * In ecommerce case, a sequence consisting of exploring and purchasing interactions associated with the same product ID can be extracted as a task, with a label following schema "buy:{product_type}" attached. However, relevant in-between exploring interactions on other related products might be dropped.
  * In social network case, grouping post-viewing and commenting interactions targetting the same topic as one task, with a label following schema "explore:{topic}" attached. However, relevant in-between exploring interactions on other related topics might be dropped.
  * Rule-based grouping can help :ub:`estimate interleaving probabilities`, such as how often a user buy products from two categories together, or how often user explore two topics together on social nework.

* :ub:`Synthetic Data Generation` aims at creating artificial but realistic user sessions with intent labels. For example, to synthesize a sequence ending in purchasing kid cloth and drinks:
  
  1. **Item Selection**: Sample two items from "kid cloth" and "kid drinks" product categories, corresponding to "buy:kid_clothing" and "buy:kid_drinks" tasks.

  2. **Query and Interaction Simulation**: For each item:

     * Sample an associated query from historical data, optionally applying a rephrasing model for diversity.
     * Either use historical interaction sequences that led to purchase of that item, or trigger actual product search API and navigate through results, applying :ub:`historical transition probabilities to simulate realistic interaction patterns`.
  
  3. **Sample or Simulate Noisy Interactions**: Simulate common interactions that can be considered noisy to the "buy:kid_clothing" and "buy:kid_drinks" tasks, such as account management interactions. Again, either sample from real historical data, or simulate according to historical transition probabilities.

  4. **Multi-Task Simulation**: Interleaving the interaction sequences for both "kid cloth", "kid drinks", as well as other noisy interactions. Interleaving them according to a historical traffic-based probability to simulate realistic multi-task behavior.

  Here is another example for synthesizing a session involving content discovery and creator engagement on social network:

  1. **Topic Selection**: Sample realistic hashtags or search terms from historical data that lead to different topics.
  2. **Engagement and Navigation Simulation**: For the selected content:

     * Either replay historical navigation patterns from users with similar interests, or use the platform's prodcut recommendation pipeline to generate realistic content recommendations.
     * Apply known engagement probability distributions (e.g., 60% view, 25% like, 10% comment, 5% share) based on content type and user segment.

  3. **Social Interaction Simulation**: Incorporate realistic social behaviors:

     * Follow a creator after engaging with multiple pieces of their content.
     * Simulate direct message interactions after certain engagement thresholds.

  4. **Multi-Task Simulation**: Interleaving the interaction sequences according to a historical traffic-based probability to simulate realistic multi-task behavior.

Modeling
^^^^^^^^

Given a session $S$ of $N$ interactions, where each interaction $i$ has a feature representation $\\mathbf{x}_i$ produced by an interaction encoder (where multi-modal signals such as the interaction metadata, interaction results, basic intent labels, etc., have been integrated). This is fundamentally a :newconcept:`Semantic Grouping Problem For Sequence` (similar to sentence grouping by topics in a paragraph). As a common practice, they can be passed into a transformer architecture with pre-defined max session size, and eventually linked a :ub:`pairwise cross-entropy loss` for every pair of items in the sequence. 

* We are unable to pre-define the number of groups in a session due to its dynamic nature, therefore pairwise loss is more suitable than multi-class loss.

.. math:: 

    \mathcal{L}_{\text{pairwise}} = -\frac{1}{N \times (N-1)} \sum_{i=1}^{N}\sum_{j=1,j\neq i}^{N} \left[ y_{i,j} \log(p_{i,j}) + (1-y_{i,j}) \log(1-p_{i,j}) \right]

where $p\_{i,j}$ is the probability of interaction $i$ be in the same semantic task group as interaction $j$ after the transformers layer.

If we want to consider the factor of :ub:`temporal proximity`, we can add a :newconcept:`temporal coherence regularization` term. For example, interactions within the same task often occur in close proximity even with interleaving, and we want to be conservative and ensure correctness of grouping of items near each other.

.. math::

    \mathcal{L}_{\text{temporal}} = \sum_{i=1}^{N}\sum_{j=1,j\neq i}^{N} w_{i,j}^{\text{time}} \cdot \text{temporal_penalty}(p_{i,j}, y_{i,j})

where $w_{i,j}^{\\text{time}}$ is a :newconcept:`temporal weight`, for example, $w_{i,j}^{\\text{time}} = e^{-\\frac{\|t_i - t_j\|}{\\tau}}$, which decreases the weight as the time difference between interactions increases, and has a hyperparameter $\\tau$ to adjust decreasing rate. One example of the :newconcept:`temporal penality` can be $\\text{temporal\_penality}(p\_{i,j}, y\_{i,j}) = p\_{i,j} - y\_{i,j}$, which is a linear penality. This encourages the model to prioritize correctly classifying temporally close interactions. Both the "temporal weight" and "temporal penality" can be tweaked according to experiment results. For example, we can increase the penalty for failing to identify related interactions beyond certain temporal distances to ensure the model doesn't focus exclusively on proximate interactions.

For the temporal penalty function that penalizes failures to identify related interactions more heavily as temporal distance increases:

.. math::

    \text{temporal_penalty}(p_{i,j}, y_{i,j}) = 
    \begin{cases}
    |p_{i,j} - y_{i,j}| & \text{if } |t_i - t_j| \leq \delta \\
    |p_{i,j} - y_{i,j}| \cdot (1 + \gamma \cdot \frac{|t_i - t_j| - \delta}{\delta_{\text{max}} - \delta}) & \text{if } |t_i - t_j| > \delta \text{ and } y_{i,j} = 1 \\
    |p_{i,j} - y_{i,j}| & \text{if } |t_i - t_j| > \delta \text{ and } y_{i,j} = 0
    \end{cases}

where:

* $\\delta$ is the temporal distance threshold beyond which we apply the increased penalty for missed connections
* $\\gamma$ is the scaling factor that determines how much extra penalty to apply for distant but related interactions
* $\\delta_{\\text{max}}$ is the maximum temporal distance to consider (can be set to the session length or another appropriate value)

This formulation increases the penalty for failing to identify related interactions that are temporally distant, helping the model learn to recognize semantic connections even when they span across significant time gaps in the session.

The final loss function combines the pairwise binary loss with the temporal coherence regularization:

.. math::
    
    \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pairwise}} + \lambda_{\text{temporal}} \cdot \mathcal{L}_{\text{temporal}}

where $\\lambda\_{\\text{temporal}}$ is a hyperparameter controlling the influence of temporal coherence.



Reward Function (Ruled-Based Label Creation)
--------------------------------------------

The next step is creating a reward function. At the beginning, it is usually a rule based function resulting from intuitive business requirements.

* Even with recent advancements `Deep Reward (Labeling) Models`_, the rule-based reward functions with `Meta Learning`_ approache is still a common practice, especially helping swiftly establish labeling and evaluation tools especially when a product is starting up and there is not yet sufficient data. The downsides for this approache include the manual efforts for feature engineering, creating and maintaining the complex reward strucuture, various thresholds and weights.
* After sufficient data have been collected, `Deep Reward (Labeling) Models`_ is an advanced techinque to reduce business complexity, ingest ever-growing complex data and signals, and better facility automated continuous model improvement.

Consider an e-commerce platform that recommends products to users. The platform collects various implicit feedback signals with increasing strength of user interest:

* **Impression (IMP)**: User sees the recommendation
* **Click (CLK)**: User clicks on the recommended product
* **Add-to-Cart (ATC)**: User adds the product to their shopping cart
* **Purchase (PUR)**: User completes the purchase

There might be more than one reward functions, depending on the business scenario. For example, in case of shopping search, there can be an initial reward function assigning reward to interacted items at the shopping session. Later, when the user submit reviews, the initial reward can be corrected with a follow-up correction. The following is a comprehensive example.

* ``calculate_interaction_reward`` - This is calculating rewards based on user interactions up to and including the purchase. It runs at the time of the shopping session, using immediate signals like clicks, cart actions, and purchases.
* ``adjust_reward_with_rating`` - This takes the initial reward value and adjusts it when a customer later submits a rating and review. This function can be called asynchronously whenever a review comes in, even if it's days or weeks after the session.
* The following is still a simplified synthetic example, but **shows how complex a manul reward function could be for a realistic business scenario**. 

.. _code-example-ecommerce-reward-function:

.. code-block:: python
   :class: folding
   :name: calculate_interaction_reward

    def calculate_interaction_reward(user_signals, time_thresholds):
        """
        Calculate the initial reward value based on user interaction signals
        (This is the original reward function without ratings)
        
        Parameters:
        - user_signals: Dictionary containing signal flags and timestamps
        {
            'impression': {'occurred': bool, 'timestamp': datetime},
            'click': {'occurred': bool, 'timestamp': datetime},
            'add_to_cart': {'occurred': bool, 'timestamp': datetime},
            'remove_from_cart': {'occurred': bool, 'timestamp': datetime},
            'save_for_later': {'occurred': bool, 'timestamp': datetime},
            'purchase': {'occurred': bool, 'timestamp': datetime},
            'dwell_time': float  # Time spent on product page in seconds
        }
        - time_thresholds: Dictionary of time thresholds
        {
            'quick_bounce': float,  # seconds threshold for negative signal
            'good_engagement': float  # seconds threshold for positive signal
        }
        
        Returns:
        - float: Calculated reward value
        """
        # Base reward initialization
        reward = 0.0
        
        # No impression or impression without click (negative signal)
        if not user_signals['impression']['occurred'] or not user_signals['click']['occurred']:
            reward = -0.1
            return reward
        
        # Calculate time between impression and click (if both occurred)
        if user_signals['impression']['occurred'] and user_signals['click']['occurred']:
            imp_to_click_time = (user_signals['click']['timestamp'] - 
                                user_signals['impression']['timestamp']).total_seconds()
            
            # Quick clicks might indicate accidental clicks or misleading thumbnails
            if imp_to_click_time < time_thresholds['quick_bounce']:
                reward -= 0.05
        
        # Add reward components based on user engagement levels
        if user_signals['click']['occurred']:
            reward += 0.2
            
            # Penalize very short dwell times (likely bounce/irrelevant content)
            if user_signals['dwell_time'] < time_thresholds['quick_bounce']:
                reward -= 0.1
            # Reward good engagement with content
            elif user_signals['dwell_time'] > time_thresholds['good_engagement']:
                reward += 0.2
        
        # Handle add-to-cart, save-for-later, and remove-from-cart scenarios
        if user_signals['add_to_cart']['occurred']:
            # Case 1: Item was later removed from cart (strong negative signal)
            if user_signals['remove_from_cart']['occurred']:
                atc_to_rfc_time = (user_signals['remove_from_cart']['timestamp'] - 
                                user_signals['add_to_cart']['timestamp']).total_seconds()
                
                # Check if removal was to save for later (more positive than pure removal)
                if user_signals['save_for_later']['occurred']:
                    # Timing check to ensure save-for-later happened around the same time as removal
                    rfc_to_sfl_time = abs((user_signals['save_for_later']['timestamp'] - 
                                        user_signals['remove_from_cart']['timestamp']).total_seconds())
                    
                    if rfc_to_sfl_time < 10:  # Save-for-later immediately after removal (likely same action)
                        reward += 0.2  # Moderate positive signal - still interested, just not right now
                        
                        # Additional reward if they spent significant time viewing the product first
                        if user_signals['dwell_time'] > time_thresholds['good_engagement']:
                            reward += 0.1  # Additional signal of genuine interest
                else:
                    # Regular removal without saving for later (negative signal)
                    # Immediate removal is a stronger negative signal than delayed removal
                    if atc_to_rfc_time < 60:  # Quick removal (within 1 minute)
                        reward -= 0.8  # Strong negative signal - likely misclick or immediate regret
                    elif atc_to_rfc_time < 300:  # Removal after some consideration (within 5 minutes)
                        reward -= 0.5  # Moderate negative signal - considered but rejected
                    else:  # Delayed removal
                        reward -= 0.3  # Milder negative signal - may be due to budget or other constraints
            
            # Case 2: Item was directly saved for later without removal (meaning it was never in cart)
            elif user_signals['save_for_later']['occurred'] and not user_signals['remove_from_cart']['occurred']:
                # Direct save-for-later is a positive signal, but weaker than add-to-cart
                reward += 0.3
                
                # Consider timing of save-for-later decision
                if user_signals['click']['occurred']:
                    click_to_sfl_time = (user_signals['save_for_later']['timestamp'] - 
                                        user_signals['click']['timestamp']).total_seconds()
                    
                    if click_to_sfl_time < 45:  # Quick decision indicates stronger interest
                        reward += 0.05
                        
                    # High dwell time before save-for-later suggests genuine interest
                    if user_signals['dwell_time'] > time_thresholds['good_engagement']:
                        reward += 0.1
            
            # Case 3: Item remained in cart (most positive signal short of purchase)
            else:
                # Item remained in cart (positive signal)
                reward += 0.5
                
                # Add time-based component: faster add-to-cart after click might
                # indicate stronger interest
                if user_signals['click']['occurred']:
                    click_to_atc_time = (user_signals['add_to_cart']['timestamp'] - 
                                    user_signals['click']['timestamp']).total_seconds()
                    if click_to_atc_time < 30:  # Quick decision to add to cart
                        reward += 0.1
        
        if user_signals['purchase']['occurred']:
            # Strongest signal of recommendation success
            reward += 1.0
            
            # If purchase happens in same session as impression
            if (user_signals['purchase']['timestamp'] - 
                user_signals['impression']['timestamp']).total_seconds() < 1800:  # 30 minutes
                reward += 0.2
        
        return reward


.. code-block:: python
   :class: folding
   :name: adjust_reward_with_rating

    def adjust_reward_with_rating(initial_reward, purchase_timestamp, rating_info):
        """
        Adjust the initial reward value based on customer ratings and reviews that arrive later
        
        Parameters:
        - initial_reward: float, the reward value calculated from the interaction function
        - purchase_timestamp: datetime, when the purchase occurred
        - rating_info: Dictionary containing rating information
        {
            'rating_value': float,  # Customer rating (e.g., 1-5 stars)
            'rating_timestamp': datetime,  # When the rating was submitted
            'review_length': int,  # Length of review text (if applicable)
            'has_photo': bool,  # Whether the review includes photos
            'has_video': bool,  # Whether the review includes videos
            'verified_purchase': bool,  # Whether the review is from a verified purchase
            'helpful_votes': int  # Optional: number of helpful votes (if applicable)
        }
        
        Returns:
        - float: Adjusted reward value incorporating rating information
        """
        # Start with the initial reward
        adjusted_reward = initial_reward
        
        # Scale rating to a range from -1.0 to 1.0 (for a 1-5 star system)
        # This makes 3-star neutral, 1-star a significant penalty, and 5-star a significant boost
        normalized_rating = (rating_info['rating_value'] - 3) / 2
        
        # Base rating impact on reward
        rating_adjustment = normalized_rating * 1.0  # Scaling factor (adjust as needed)
        
        # Add extra weight to ratings with more detailed feedback
        if rating_info.get('review_length', 0) > 50:  # Substantial text review
            rating_adjustment *= 1.2
            
        # Reviews with media are typically more informative and valuable
        if rating_info.get('has_photo', False):
            rating_adjustment *= 1.1
        if rating_info.get('has_video', False):
            rating_adjustment *= 1.2
            
        # Verified purchase reviews are more reliable
        if rating_info.get('verified_purchase', False):
            rating_adjustment *= 1.3
            
        # Consider community validation if available
        if 'helpful_votes' in rating_info and rating_info['helpful_votes'] > 0:
            # Logarithmic scaling for helpful votes to prevent outliers from dominating
            vote_factor = min(1 + 0.2 * math.log1p(rating_info['helpful_votes']), 2.0)
            rating_adjustment *= vote_factor
        
        # Timing of rating after purchase can provide insight
        if 'rating_timestamp' in rating_info:
            purchase_to_rating_time = (rating_info['rating_timestamp'] - purchase_timestamp).total_seconds()
            
            # Very quick ratings (< 1 day) might be less reliable/thoughtful
            if purchase_to_rating_time < 86400:  # 24 hours
                rating_adjustment *= 0.8
                
            # Ratings after significant product use time (e.g., 1 week+) might be more informative
            elif purchase_to_rating_time > 604800:  # 7 days
                rating_adjustment *= 1.2
                
            # Extremely delayed ratings (e.g., 30+ days) might indicate strong sentiment
            if purchase_to_rating_time > 2592000:  # 30 days
                # For positive ratings, this is a strong positive signal
                if rating_info['rating_value'] >= 4:
                    rating_adjustment *= 1.3
                # For negative ratings, this is a strong negative signal
                elif rating_info['rating_value'] <= 2:
                    rating_adjustment *= 1.3
        
        # Add rating adjustment to the initial reward
        adjusted_reward += rating_adjustment
        
        return adjusted_reward


In practice, reward functions also often incorporate business priorities beyond user engagement/ratings:

.. code-block:: python
   :class: folding
   :name: business_adjusted_reward

    def business_adjusted_reward(reward, product_data):
        """
        Adjust reward based on business priorities
        
        Parameters:
        - reward: Float value from normalize_reward function
        - product_data: Dictionary with product business information
          {
            'margin': float,  # Profit margin percentage
            'inventory_status': str,  # 'overstocked', 'normal', 'limited'
            'strategic_category': bool,  # Whether product is in a strategic category
            'new_product': bool  # Whether product is newly launched
          }
        
        Returns:
        - float: Business-adjusted reward
        """
        adjusted_reward = reward
        
        # Boost high-margin products
        if product_data['margin'] > 0.3:  # 30% margin
            adjusted_reward *= 1.1
        
        # Prioritize overstocked items
        if product_data['inventory_status'] == 'overstocked':
            adjusted_reward *= 1.15
        
        # Boost strategic category products
        if product_data['strategic_category']:
            adjusted_reward *= 1.2
        
        # Promote new products to gain market insights
        if product_data['new_product']:
            adjusted_reward *= 1.1
        
        # Ensure the final reward is still within [0,1]
        return min(adjusted_reward, 1.0)

Above reward functions incorporate several important principles to keep it :ul:`clear and straightforward`:

1. **Signal Strength Hierarchy**: Stronger signals of intent (purchase > add-to-cart > save-for-later > click > impression) receive higher reward components, while negative signals (remove-from-cart) receive appropriate penalties.
2. **Temporal Factors**: The timing between signals affects the reward value (e.g., quick bounces are penalized, quick add-to-cart after click is rewarded, immediate cart removal is penalized more heavily than delayed removal).
3. **Engagement Quality**: Dwell time is used to distinguish between genuine engagement and accidental or unsatisfied interactions, with additional bonuses for high engagement before save-for-later actions.
4. **Session Continuity**: Completing the full funnel from impression to purchase in a single session receives additional reward.
5. **Intent Classification**: The function distinguishes between different user intents by analyzing action sequences (e.g., add-to-cart followed by remove-from-cart vs. add-to-cart followed by save-for-later), providing more nuanced feedback signals.


Reward Normalization
~~~~~~~~~~~~~~~~~~~~

For many machine learning algorithms, it's beneficial to normalize the reward values to a standard range, (e.g., :math:`[0,1]`). Different normalization techniques offer various advantages depending on your specific recommendation system needs. **Min-Max Normalization** and **Sigmoid Normalization** are two most common normalization methods for rewards.

:newconcept:`Min-Max Normalization` linearly scales values to a range ($[0, 1]$ in the following) and is straightforward to implement and interpret. Cutoffs are usually set for extreme quantiles (i.e., :newconcept:`Quantile-Capped Min-Max Normalization`) to ensure the scaling more robust while maintaining the interpretability of linear scaling:

.. code-block:: python
   :class: folding
   :name: quantile_capped_normalize

    def quantile_capped_normalize(raw_reward, reward_distribution=None, 
                                  lower_quantile=0.05, upper_quantile=0.95):
        """
        Normalize using min-max scaling with quantile cutoffs to handle outliers
        
        Parameters:
        - raw_reward: Float value from calculate_reward function
        - reward_distribution: Optional list of representative reward values 
                              to calculate quantiles from
        - lower_quantile: Percentile below which values are capped to 0 (default: 5%)
        - upper_quantile: Percentile above which values are capped to 1 (default: 95%)
        
        Returns:
        - float: Normalized reward between 0 and 1
        """
        import numpy as np
        
        # If no distribution is provided, use default bounds
        if reward_distribution is None:
            # Default bounds based on the reward function design
            min_val = 0.0
            max_val = 1.0
        else:
            # Calculate quantile bounds from the empirical distribution
            min_val = np.percentile(reward_distribution, lower_quantile * 100)
            max_val = np.percentile(reward_distribution, upper_quantile * 100)
        
        # Cap the raw_reward to the quantile bounds
        capped_reward = max(min_val, min(raw_reward, max_val))
        
        # Apply min-max normalization on the capped value
        if max_val > min_val:  # Avoid division by zero
            normalized = (capped_reward - min_val) / (max_val - min_val)
        else:
            normalized = 0.5  # Default if min and max are the same
        
        return normalized

This quantile-capped min-max approach offers several benefits for recommendation systems:

* :ub:`Outlier Resistance`: By using percentile-based boundaries instead of absolute min/max values, the normalization becomes significantly more robust to extreme values in the reward distribution.
* :ub:`Adaptability`: The cutoff points can adjust when the reward distribution changes as you collect more data or modify your reward function.
* :ub:`Focus on Relevant Range`: By concentrating the normalization on the middle 90% (above code by default exclude top 5% and bottom 5%, or whatever range you choose) of reward values, you improve the resolution where it matters most.
* :ub:`Interpretability`: Unlike some other outlier-resistant methods, it preserves the linear relationship between values within the accepted range.

One major weakness of min-max normalization is the inability to control the "midpoint". In practice, there is a :newconcept:`Meaningful Interaction Midpoint` (a.k.a. :newconcept:`Neutral Point`) representing a raw reward score threshold when high-value user interaction starts to happen (e.g., long dwell time, being added to cart). The definition of this "meaningful midpoint" itself is a business decision. A possible example could be 5% quantile of reward scores associated with items at least being added to the cart. Intutively, :ub:`we should boost reward from this "Meaningful Interaction Midpoint" on`. However, 

* It is :ub:`not straightforward for min-max normalization to capture the "Meaningful Interaction Midpoint"`. This requires careful engineering of the reward function to match the raw score distribution. In the following Figure :numref:`fig-reward-normalization`, this midpoint is mapped to reward 0.56, although not unreasonable, it is at least hard to interpret.
* Min-max normalization uses linear scale, and there is :ub:`no reward boost after "Meaningful Interaction Midpoint"`.
* The extreme quantile cutoffs are rigid, and :ub:`deciding the quantile thresholds adds additional business complexity`. This quantile thresholds might need a revisit when data distribution shifts.


:newconcept:`Sigmoid Normalization` uses the logistic function to map values to the :math:`[0,1]` range. It offers several advantages:

* :ub:`Naturally handles Meaningful Interaction Midpoint and easy to control reward boost around the midpoint`: 
   
   * The ``midpoint`` parameter determines which raw reward value maps to 0.5, essentially setting the "Meaningful Interaction Midpoint" of your reward system.
   * The ``steepness`` parameter controls how quickly values transition from low to high around the "Meaningful Interaction Midpoint", allowing you to emphasize differences and create sharper classifications.
  
* :ub:`Handles outliers gracefully`: Unlike min-max normalization using quantile cutoffs, sigmoid is less affected by extreme values or outliers in the raw reward distribution.

Although not always, Sigmoid Normalization is indeed more commonly used than min-max normalization. In practice, you might experiment with both approaches to see which produces better results for your specific search/recommendation/Ads task and model architecture.

* **Use quantile-capped min-max normalization when**:

  * You need linear scaling for interpretability
  * You want to handle outliers while maintaining linear properties within the relevant range
  * Your recommendation model performs better with a linear transformations
  * Computational efficiency is critically important (simpler calculation than sigmoid)


* **Use sigmoid normalization when**:

  * Your reward function produces a wide or unpredictable range of values
  * You want to emphasize differences around a certain threshold (controlled by midpoint)
  * You need to flexibly adjust the "Meaningful Interaction Midpoint" (by adjusting logistic function midpoint parameter)
  * You need to flexibly adjust the sensitivity of the normalization (by adjusting logistic function steepness parameter)
  * Your recommendation model performs better with smooth non-linear transformations
  * You prefer a probabilistic interpretation of reward values


.. code-block:: python
   :class: folding
   :name: sigmoid_normalize_reward

    def sigmoid_normalize_reward(raw_reward, steepness=1.0, midpoint=0.5):
        """
        Normalize the raw reward value using a sigmoid function to a [0,1] range
        
        Parameters:
        - raw_reward: Float value from calculate_reward function
        - steepness: Controls how steep the sigmoid curve is (higher = steeper transition)
        - midpoint: The raw_reward value that should map to 0.5 after normalization
        
        Returns:
        - float: Normalized reward between 0 and 1
        """
        import math
        
        # Apply sigmoid normalization: 1 / (1 + e^(-steepness * (x - midpoint)))
        normalized = 1 / (1 + math.exp(-steepness * (raw_reward - midpoint)))
        
        return normalized


.. _fig-reward-normalization:

.. figure:: /_static/images/modeling/classic_modeling/data_preparation/reward_normalization_comparison.png
   :alt: E-Commerce Reward Normalization: Min-Max vs. Sigmoid
   :width: 100%
   :align: left
   
   The e-commerce reward distribution in the visualization shows four peaks because it's modeling distinct clusters of user behaviors that commonly occur:
   **1. First Peak** (left, around -2.0): This represents negative interactions like cart removals and quick bounces. These are cases where users had some initial interest but then explicitly rejected items, perhaps by removing them from cart or clicking and then immediately leaving the page.
   **2. Second Peak** (middle, around -0.5): This largest peak represents the most common scenario - impressions without meaningful engagement. These are search results shown to users who either didn't click at all or clicked but quickly left without significant engagement. This is typically the most frequent outcome in a search system.
   **3. Third Peak** (right, between 0.5-1.0): This represents moderate positive engagement, including clicks with longer dwell time and some interest shown. These are users who engaged with search results but didn't take high-value actions like adding to cart or purchasing.
   **4. Fourth Peak** (furthest right, around 2.0-2.5): This represents the most valuable user interactions - completed purchases, especially those followed by positive ratings or reviews. These are the rarest but most valuable outcomes in an e-commerce recommendation system.
   **In contrast**, sigmoid normalization explicitly maps the "Meaningful Interaction Point" to 0.5, and it naturally boosts the reward after the midpoint. You may also adjust another **steepness** parameter to control the boost.


.. note:: Do reward sign and range matter?

    The sign of the raw reward (positive vs. negative values) is indeed important in most learning algorithms. For example:

    * In reinforcement learning, a positive reward encourages behavior while a negative reward discourages it
    * In supervised learning with reward weighting, positive vs. negative rewards can determine whether to strengthen or weaken certain patterns

    When using normalization, a common practice is to map all values to the range :math:`[0,1]`, which means there are no negative numbers in the output. This creates a potential disconnect. The solution is to establish a :ub:`reference point` within the :math:`[0,1]` range that represents "neutral". Typically, this is 0.5 (for sigmoid normalization, this is by setting the logistic function's midpoint to "Meaningful Interaction Midpoint"):

    * Values above 0.5 represent positive rewards (positive, encouraging behaviors)
    * Values below 0.5 represent negative rewards (negative, discouraging behaviors)


Meta Learning
~~~~~~~~~~~~~

Meta-learning approaches for reward function optimization represent a way to systematically improve the reward functions beyond manual tuning. Meta-learning in this context means creating a system that "learns how to learn" - specifically, it learns how to optimize the parameters of your reward function based on business outcomes. Instead of manually adjusting weights and thresholds through trial and error, you're automatically discovers the optimal parameters.

The first step is identifying all tunable parameters in the reward function (taking above :ref:`ecommerce reward function <code-example-ecommerce-reward-function>` for examle):

- **Base Signal Weights**: The fundamental values assigned to each user action
  .. code-block::

    impression_weight = 0.1
    click_weight = 0.2
    add_to_cart_weight = 0.5
    purchase_weight = 1.0


- **Time Thresholds**: The timing boundaries that categorize user behaviors
  .. code-block::
    quick_bounce_threshold = 5.0  # seconds
    good_engagement_threshold = 30.0  # seconds

- **Temporal Modifiers**: Adjustments based on timing between actions
  .. code-block::
    quick_click_penalty = -0.05
    long_dwell_bonus = 0.2
    quick_atc_bonus = 0.1
    same_session_purchase_bonus = 0.2


- **Normalization Parameters**: Controls for reward scaling
  .. code-block::
    sigmoid_midpoint = 0.5
    sigmoid_steepness = 1.0


The second step is having clear business metrics to optimize against. These serve as the "ground truth" for evaluating reward function effectiveness:

* **Revenue or Profit**: Direct financial impact
* **Conversion Rate**: Percentage of users completing desired actions
* **User Retention**: Return rate and engagement over time
* **Customer Lifetime Value**: Long-term user value
* **User Satisfaction**: Ratings, reviews, and feedback scores

The goal in reward function optimization is typically to maximize the correlation between the calculated rewards and business metrics (like revenue or conversions). A high positive correlation indicates that the reward function is effectively predicting business outcomes. For multiple business objects, we can have a weighted combination of them. Although there is still manual weights in this combination, the number of manual parameters are significantly less than the reward function, significantly reducing business complexity. 

* Traditionally, meta learning is performed using a representative medium-sized subsample of the whole dataset. A typical method is :newconcept:`L-BFGS-B (Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bounds)` for several specific reasons,

    * **Constrained optimization**: The "B" in L-BFGS-B indicates it handles parameter bounds, which is critical when :ub:`optimizing reward function parameters that should remain within specific ranges` (for example, we want to prevent reward/penality for user actions like adding to or removing from cart from going too extreme).
    * **Memory efficiency**: L-BFGS-B is :ub:`memoery-efficient for the meta-learning case` without excessive memory requirements.
    * **Second-order information**: L-BFGS-B (a `Quasi-Newton Method <https://en.wikipedia.org/wiki/Quasi-Newton_method>`) approximates second derivatives, potentially leading to faster convergence compared to first-order methods.

* For an objective function that is hard to calculate the gradient formula, it
  
  1. Compute the objective function value at the current point.
  2. Perturb each parameter slightly to approximate partial derivatives. Store the last $m$ pairs of position changes and gradient changes. Use these numerical approximations to estimate the gradient and how its gradient change (2nd order approximation).
  3. Apply the L-BFGS-B algorithm using these estimated gradients

* This traditional method has a stochastic version that can handle dataset of a large size for robustness and reduce the complexity and workload on data sampling. However, if a more sophisticated approach is required to naturally integrate same level of diverse and complex information as in the runtime search/recommendation/Ads models, the a modern day approach is `Deep Reward Models`_.

.. code-block:: python
   :class: folding
   :name: meta_learning_example

    def meta_learning_multi_objective_function(params, interactions, revenue_metrics, retention_metrics, satisfaction_metrics):
        rewards = [calculate_reward(interaction, params) for interaction in interactions]
        
        # Calculate correlations with different business metrics
        revenue_corr = np.corrcoef(rewards, revenue_metrics)[0, 1]
        retention_corr = np.corrcoef(rewards, retention_metrics)[0, 1]
        satisfaction_corr = np.corrcoef(rewards, satisfaction_metrics)[0, 1]
        
        # Weighted combination based on business priorities
        combined_score = (0.5 * revenue_corr + 
                        0.3 * retention_corr + 
                        0.2 * satisfaction_corr)
        
        return -combined_score  # Negative for minimization

    from scipy.optimize import minimize
    result = minimize(
        meta_learning_multi_objective_function,
        initial_params,
        args=(user_interactions, business_metrics),
        method='L-BFGS-B',
        bounds=parameter_bounds
    )


Deep Reward (Labeling) Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Deep Reward Models` is now a widely-used advanced approach to build a powerful labeling tool. It is the most critical module for a :newconcept:`Self-Improving ML/AI System` (meaning that the ML/AI system can continuously improve itself using real-time user feedback data with a :newconcept:`Closed Feedback Learning Loop`).

* They :ub:`directly learn the mapping from user interaction signals to reward values`, bypassing the need for explicit feature engineering, rule-based reward function development, and manual parameter tuning.
* They still need :newconcept:`Meta Rules` (or :newconcept:`Meta Labels`), which only require much simpler :refconcept:`Preference Rules` distinguishing a simple scenario - given interaction sequences $A$ and $B$, which one is prefered. For example, if interaction session $A$ leads to purchase, but interaction session $B$ only leads to adding item to cart, then obviously $A$ is more preferred for the user. In comparison the the complex manual reward structure shown in :ref:`ecommerce reward function <code-example-ecommerce-reward-function>`, such :ub:`preference rules are much more intuitive to develop and maintain` for business. Annotation on the data points with such preference rules are called :newconcept:`Preference Annotations`.
* They are able to :ub:`ingest same level of complex data` as the runtime search/recommendation/Ads models and incorporating multi-modal signals, because it is a deep model.


In comparison to :newconcept:`Runtime Models`, despite their similarities in data inputs, they serve fundamentally different purposes:

* :ub:`Input Difference`: 
  
  * **Reward Model**: is solely based on :refconcept:`Historical User Activities & Analytics`, has the advantage of information asymmetry of hindsight, able to access all post-interaction signals (how users actually responded to search results, recommendations or Ads, various analytics, etc.), and :ub:`has complete data for every session it evaluates`.
  * **Runtime Model**: The runtime model must make predictions on the next action before signals for the next action exist. It has access to :refconcept:`Historical User Activities & Analytics` for historical sessions, and :refconcept:`Real-Time User Activities & Analytics` for the current session, which is :ub:`partial in comparison to what the reward model has`. 
    
    * During runtime model training, the model must simulate the partial information for the current session.
  
  * Think of it as a :ub:`teacher-student relationship`. The reward model (teacher) evaluates performance with full historical information, creating learning signals (labels) to guide runtime model to improve, while the runtime model (student) learns from the teacher, but must also learn to make good predictions in real scenarios in the future.

* :ub:`Different Objectives`:
  
  * **Reward Model**: Focuses on :ub:`offline` evaluating the quality or relevance of recommendations after the fact without latency constraint :ub: `without latency constraint`. Given the sequence of actions happened before an action being evaluated, it learns to predict how valuable that user interaction was (through clicks, purchases, dwell time, etc.).
  * **Runtime Model**: Focuses on :ub:`real-time` evaluating the quality or relevance and making the actual search/recommendations/Ads :ub:`with latency constraint`.  Given the sequence of actions already happend during runtime, it predicts user's next action (e.g., interaction with an item, clicking an ad).

* :ub:`Training Method Difference`: Due to the difference in objectives,
  
  * **Reward Model** is usually :ub:`built with supervised training using preference annotations as the training targets`.
  * **Runtime Model** is usually :ub:`built with reinforcement learning`
  * Both reward model and runtime model updates can leverage continuous training (maintaining model training states, incorporating new data, dropping outdated data, and continuing the training with a small number of epoches).

.. figure:: /_static/images/modeling/classic_modeling/data_preparation/closed_learning_loop.png
   :alt: Self-Improving ML/AI System with Closed Feedback Learning Loop
   :width: 100%
   :align: left
   
   Self-Improving ML/AI System with Closed Feedback Learning Loop: The deep reward model (teacher) creates learning signals using complete historical data and operates offline without latency constraints. The runtime model (student) learns from these signals to make real-time predictions for the future under strict latency requirements with partial information from a runtime seesion. The system features two distinct training pipelines: runtime model training using reinforcement learning (updated more frequently, e.g., daily) and reward model training using supervised preference learning (updated less frequently, e.g., weekly). Both implicit interactions and explicit feedback are captured to continuously improve the system.


.. note::

    Traditionally, :newconcept:`Inverse Reinforcement Learning (IRL)` was a widely-used approach for building reward models in specific domains. IRL infers reward functions from :newconcept:`Expert Demonstrations` - sequences of states and actions that demonstrate optimal behavior.
    
    This approach has become less common in modern search/recommendation/ads systems for several key reasons:
    
    * **Data Collection Challenges**: As systems become more versatile, collecting comprehensive expert demonstrations across diverse scenarios becomes prohibitively expensive and often infeasible.
    * **Unrealistic Assumptions**: IRL assumes experts act optimally according to a consistent reward function. Human behavior frequently violates this assumption - people are inconsistent, make mistakes, and have mixed motives.
    * **Computational Complexity**: IRL requires repeatedly solving the forward RL problem during training, which becomes computationally prohibitive for large-scale models and complex state spaces.
    
    Today, :newconcept:`Preference Learning` has become the predominant method for building reward models, as it requires only comparative judgments between alternatives rather than complete optimal demonstrations, making data collection more practical and scalable.


Preference Learning
^^^^^^^^^^^^^^^^^^^

The fundamental unit in preference learning is the :newconcept:`Preference Pair`, derived from the pre-defined :newconcept:`Preference Rules` - "given two interaction sequences, which one is preferred over the other". For example, a preference rule can be "any interaction sequence ending in purchase has highest user preference". As a result,

* Interaction Sequence A: View → Dwell → Add to Cart → Purchase
* Interaction Sequence B: View → Dwell → Saved For Later → View Different Product
* Preference: A > B (A is preferred over B because it ends with a purchase interaction)

In general, these preference pairs can be obtained by

* **Automatically generated** using pre-defined preference rules based on implicit user behavior signals (purchases, cart additions, etc.)
* **Explicitly labeled** by user feedback, or human annotators

