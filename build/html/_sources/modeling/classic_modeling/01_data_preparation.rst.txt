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


Reward Function
---------------

The next step is creating a reward function. At the beginning, it is usually a rule based function resulting from intuitive business requirements.

Consider an e-commerce platform that recommends products to users. The platform collects various implicit feedback signals with increasing strength of user interest:

* **Impression (IMP)**: User sees the recommendation
* **Click (CLK)**: User clicks on the recommended product
* **Add-to-Cart (ATC)**: User adds the product to their shopping cart
* **Purchase (PUR)**: User completes the purchase

There might be more than one reward functions, depending on the business scenario. For example, in case of shopping search, there can be an initial reward function assigning reward to interacted items at the shopping session. Later, when the user submit reviews, the initial reward can be corrected with a follow-up correction. The following is a comprehensive example.

* ``calculate_interaction_reward`` - This is calculating rewards based on user interactions up to and including the purchase. It runs at the time of the shopping session, using immediate signals like clicks, cart actions, and purchases.
* ``adjust_reward_with_rating`` - This takes the initial reward value and adjusts it when a customer later submits a rating and review. This function can be called asynchronously whenever a review comes in, even if it's days or weeks after the session.

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


Reward Normalization (Label Creation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. figure:: /_static/images/modeling/data_preparation/reward_normalization_comparison.png
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

