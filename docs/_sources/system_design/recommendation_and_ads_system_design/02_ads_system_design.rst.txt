Advertising Systems
===================




Ad Engagement Prediction Systems
============================

This document outlines comprehensive approaches to building machine learning systems for ad engagement prediction across different platforms. These systems are designed to optimize ad relevance, user engagement, and business objectives in a scalable, efficient manner.

Ad Engagement Prediction System Architecture
--------------------------------------------

Ad engagement prediction shares many characteristics with recommendation systems but focuses specifically on predicting user interactions with advertisements. Like recommendation systems, ad engagement prediction can be categorized into three types:

- **Reactive**: Responding to explicit user actions or queries (e.g., search ads)
- **Proactive**: Anticipating user needs without explicit prompting (e.g., personalized ad suggestions)
- **Feed**: Combining elements of both in a curated content stream (e.g., social media ads)

All ad engagement prediction systems, regardless of implementation context, consist of four major components:

* **Task Understanding**: Interpreting user intent and needs through explicit queries, implicit signals, or predicted interests to determine what ads would be most relevant in the current context.
* **Contextual Awareness**: Incorporating session data, location, time, device information, and other situational factors to enhance relevance and personalization of ad recommendations.
* **Delivery**: Creating a ranked list of ads that best match the interpreted task and context, optimizing for user engagement, satisfaction, and business objectives.
* **Evaluation & Feedback**: Measuring ad system performance through metrics, A/B testing, and user feedback to continuously improve the system and adapt to changing preferences and behaviors.

Staged Ad Recommendation System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to general recommendation systems, ad engagement prediction employs a staged filtering approach that progressively narrows down candidate items to achieve both computational efficiency and recommendation quality.

1. **Sourcing Layer** 
   
   The sourcing layer acquires potential ad candidates from diverse sources, making proper storage and indexing before formal retrieval begins.
   
   * **Integrated Data Sources**:
     * **Advertiser Campaign Data**: Ad creatives, targeting parameters, bids, budgets
     * **Ad Inventory**: Available ad slots, formats, placement contexts
     * **Platform Data**: User profiles, interaction history, contextual information
   
   * **Implementation Example**:
     ```python
     def source_ad_candidates(user_context, available_slots):
         """
         Source initial ad candidates from various data sources.
         
         Parameters:
         - user_context: Dict containing user information and context
         - available_slots: List of ad slot configurations
         
         Returns:
         - List of candidate ads with basic metadata
         """
         candidates = []
         
         # Source from advertiser campaigns
         active_campaigns = ad_campaign_db.get_active_campaigns(
             region=user_context.get('region'),
             language=user_context.get('language')
         )
         
         # Apply basic filtering rules
         for campaign in active_campaigns:
             if campaign_matches_basic_criteria(campaign, user_context, available_slots):
                 candidates.extend(campaign.get_ad_creatives())
         
         # Add real-time bid ads if applicable
         if rtb_enabled(available_slots):
             rtb_candidates = fetch_rtb_candidates(user_context, available_slots)
             candidates.extend(rtb_candidates)
         
         return candidates
     ```

2. **Recall Layer**
   
   The recall layer retrieves a manageable subset of potentially relevant ads from the vast sourced candidate pool.
   
   * **Implementation Approaches**:
     * **Vector Search**: Embedding-based retrieval using approximate nearest neighbor search
     * **Rule-Based Filtering**: Targeting criteria, frequency capping, budget constraints
     * **Collaborative Filtering**: Leveraging similar user preferences
   
   * **Example Implementation**:
     ```python
     def retrieve_relevant_ads(user_id, query_context, candidates, max_candidates=1000):
         """
         Retrieve most relevant ad candidates using vector search and filtering.
         
         Parameters:
         - user_id: User identifier
         - query_context: Search query or contextual information
         - candidates: List of candidate ads from sourcing layer
         - max_candidates: Maximum number of candidates to return
         
         Returns:
         - Filtered list of most relevant ad candidates
         """
         # Get user embedding
         user_embedding = user_embedding_store.get(user_id)
         
         # Get query/context embedding
         if query_context:
             context_embedding = embedding_model.encode(query_context)
         else:
             context_embedding = None
         
         # Combine embeddings for retrieval
         search_embedding = combine_embeddings(user_embedding, context_embedding)
         
         # Retrieve ads using ANN search
         ad_ids = vector_index.search(
             query_vector=search_embedding,
             filter_conditions={
                 "eligible_for_user": user_eligibility_filter(user_id),
                 "within_budget": active_budget_filter()
             },
             limit=max_candidates
         )
         
         return [ad for ad in candidates if ad.id in ad_ids]
     ```

3. **Integrity Layer**
   
   The integrity layer ensures candidates passing to the precision layer meet business rules, quality standards, and policy requirements.
   
   * **Key Components**:
     * **Policy Enforcement**: Ensuring ads comply with platform policies and legal requirements
     * **Quality Checks**: Ad creative quality, landing page experience
     * **Advertiser Reputation**: Historical performance, user feedback
     * **Safety Filters**: Blocking harmful or inappropriate content
   
   * **Example Implementation**:
     ```python
     def apply_integrity_filters(user_context, ad_candidates):
         """
         Apply integrity filters to ensure ads meet platform standards.
         
         Parameters:
         - user_context: Dict containing user information and context
         - ad_candidates: List of ad candidates from recall layer
         
         Returns:
         - Filtered list of ads that pass integrity checks
         """
         filtered_candidates = []
         
         for ad in ad_candidates:
             # Check policy compliance
             if not policy_service.is_compliant(ad, user_context):
                 continue
             
             # Check ad quality
             quality_score = quality_model.predict(ad, user_context)
             if quality_score < MIN_QUALITY_THRESHOLD:
                 continue
             
             # Check advertiser reputation
             advertiser_score = reputation_service.get_score(ad.advertiser_id)
             if advertiser_score < MIN_ADVERTISER_SCORE:
                 continue
             
             # Check safety filters
             if safety_service.contains_unsafe_content(ad, user_context):
                 continue
             
             # Ad passed all integrity checks
             filtered_candidates.append(ad)
         
         return filtered_candidates
     ```

4. **Precision Layer**
   
   The precision layer applies sophisticated ranking to the filtered candidate set to identify the most engaging ads.
   
   * **Model Types**:
     * **Deep Neural Networks**: Multi-layer perceptrons, transformers for complex feature interactions
     * **Gradient Boosting Decision Trees**: XGBoost, LightGBM, CatBoost for efficient prediction
     * **Multi-task Learning Models**: Optimizing for multiple objectives simultaneously
   
   * **Feature Categories**:
     * **User Features**: Demographics, interests, behavior patterns
     * **Ad Features**: Creative elements, targeting parameters, historical performance
     * **Contextual Features**: Time, device, location, surrounding content
     * **Interaction Features**: Cross-features capturing user-ad relationships
   
   * **Example Implementation**:
     ```python
     def rank_ad_candidates(user_context, ad_candidates):
         """
         Rank ad candidates based on predicted engagement probability.
         
         Parameters:
         - user_context: Dict containing user information and context
         - ad_candidates: List of ad candidates that passed integrity checks
         
         Returns:
         - Ranked list of ads with engagement probability scores
         """
         # Extract features for each candidate
         features_batch = []
         for ad in ad_candidates:
             features = extract_features(user_context, ad)
             features_batch.append(features)
         
         # Get engagement predictions from model
         engagement_scores = engagement_model.predict_batch(features_batch)
         
         # Apply advertiser bid adjustments
         adjusted_scores = []
         for i, (ad, score) in enumerate(zip(ad_candidates, engagement_scores)):
             bid_adjustment = calculate_bid_adjustment(ad, score)
             final_score = score * bid_adjustment
             adjusted_scores.append((ad, final_score))
         
         # Rank ads by adjusted score
         ranked_ads = sorted(adjusted_scores, key=lambda x: x[1], reverse=True)
         
         return ranked_ads
     ```

Ad Relevance System for Search Engines
--------------------------------------

Search advertising systems display ads in response to user queries, making them primarily reactive recommendation systems. The core challenge is matching ad content to search intent while ensuring commercial viability.

### System Architecture

1. **Query Understanding**
   
   * **Query Analysis**: Parse, normalize, and understand search intent
   * **Query Expansion**: Identify related terms and concepts
   * **Commercial Intent Classification**: Determine if the query has commercial intent
   
   * **Example Implementation**:
     ```python
     def analyze_search_query(query_text):
         """
         Analyze search query to understand intent and commercial potential.
         
         Parameters:
         - query_text: The raw search query text
         
         Returns:
         - Dict containing query analysis results
         """
         # Normalize and tokenize query
         normalized_query = normalize_text(query_text)
         tokens = tokenize(normalized_query)
         
         # Extract entities and concepts
         entities = entity_recognition_model.extract(tokens)
         
         # Expand query with related terms
         expanded_terms = query_expansion_model.expand(normalized_query)
         
         # Classify commercial intent
         commercial_score = commercial_intent_model.predict(
             query=normalized_query,
             entities=entities
         )
         
         # Categorize query
         categories = query_categorization_model.predict(normalized_query)
         
         return {
             'original_query': query_text,
             'normalized_query': normalized_query,
             'entities': entities,
             'expanded_terms': expanded_terms,
             'commercial_score': commercial_score,
             'categories': categories
         }
     ```

2. **Ad Selection & Ranking**
   
   * **Query-Ad Matching**: Match ads to query through keyword and semantic matching
   * **Quality Score Calculation**: Assess expected ad performance (CTR, relevance)
   * **Auction Mechanism**: Combine bid and quality score for final ranking
   
   * **Example Implementation**:
     ```python
     def select_and_rank_search_ads(query_analysis, user_context):
         """
         Select and rank ads for a search query.
         
         Parameters:
         - query_analysis: Output from query analysis step
         - user_context: User and session information
         
         Returns:
         - Ranked list of ads
         """
         # Retrieve candidate ads based on query terms
         candidates = []
         
         # Keyword-based matching
         keyword_candidates = keyword_index.match(
             query_analysis['normalized_query'],
             query_analysis['expanded_terms']
         )
         candidates.extend(keyword_candidates)
         
         # Semantic matching
         if len(candidates) < MIN_CANDIDATE_THRESHOLD:
             query_embedding = embedding_model.encode(query_analysis['normalized_query'])
             semantic_candidates = vector_index.search(
                 query_vector=query_embedding,
                 filter_conditions={
                     "categories": query_analysis['categories']
                 }
             )
             candidates.extend(semantic_candidates)
         
         # Apply integrity filters
         filtered_candidates = apply_integrity_filters(user_context, candidates)
         
         # Calculate quality scores
         scored_candidates = []
         for ad in filtered_candidates:
             quality_score = calculate_quality_score(ad, query_analysis, user_context)
             predicted_ctr = predict_ctr(ad, query_analysis, user_context)
             
             # Get advertiser bid
             bid = get_bid_for_query(ad, query_analysis)
             
             # Calculate ad rank
             ad_rank = bid * quality_score
             
             scored_candidates.append({
                 'ad': ad,
                 'quality_score': quality_score,
                 'predicted_ctr': predicted_ctr,
                 'bid': bid,
                 'ad_rank': ad_rank
             })
         
         # Rank by ad rank (bid * quality score)
         ranked_ads = sorted(scored_candidates, key=lambda x: x['ad_rank'], reverse=True)
         
         return ranked_ads
     ```

3. **Special Considerations for Search Ads**
   
   * **Query Intent Alignment**: Ensuring ads match search intent types (navigational, informational, transactional)
   * **Dynamic Keyword Insertion**: Customizing ad copy based on search terms
   * **Landing Page Relevance**: Evaluating how well landing pages address the query intent
   
   * **Example Implementation**:
     ```python
     def enhance_search_ad_relevance(query, ad):
         """
         Enhance ad relevance through query-specific optimizations.
         
         Parameters:
         - query: The search query
         - ad: The ad to enhance
         
         Returns:
         - Enhanced ad with improved relevance
         """
         enhanced_ad = ad.copy()
         
         # Apply dynamic keyword insertion if enabled
         if ad.supports_dynamic_insertion:
             enhanced_ad.headline = insert_keywords(ad.headline_template, query)
             enhanced_ad.description = insert_keywords(ad.description_template, query)
         
         # Calculate landing page relevance
         landing_page_relevance = calculate_landing_page_relevance(
             landing_page=ad.landing_page,
             query=query
         )
         enhanced_ad.landing_page_relevance = landing_page_relevance
         
         # Adjust quality score based on landing page relevance
         enhanced_ad.quality_score *= (0.7 + 0.3 * landing_page_relevance)
         
         return enhanced_ad
     ```

Ad Relevance System for Social Networks
---------------------------------------

Social media advertising systems blend proactive and feed-based recommendation approaches, leveraging rich user data to predict relevant ads without explicit queries.

### System Architecture

1. **User Understanding and Profiling**
   
   * **Interest and Behavior Modeling**: Analyzing user activities, content interactions, connections
   * **Temporal Patterns**: Capturing evolving interests and behaviors over time
   * **Social Graph Utilization**: Leveraging connections and social influence patterns
   
   * **Example Implementation**:
     ```python
     def build_user_profile(user_id, timestamp):
         """
         Build comprehensive user profile for ad targeting.
         
         Parameters:
         - user_id: User identifier
         - timestamp: Current timestamp for temporal relevance
         
         Returns:
         - User profile with interests, behaviors, and social data
         """
         # Retrieve base user information
         user_info = user_db.get_user(user_id)
         
         # Get recent user activity (past 30 days)
         recent_activity = activity_store.get_user_activity(
             user_id=user_id,
             start_time=timestamp - 30*DAY_IN_SECONDS,
             end_time=timestamp
         )
         
         # Extract interests from activity
         interest_model = InterestModel()
         interests = interest_model.extract_interests(recent_activity)
         
         # Get temporal patterns
         temporal_model = TemporalModel()
         temporal_patterns = temporal_model.extract_patterns(recent_activity)
         
         # Get social graph information
         social_model = SocialGraphModel()
         social_data = social_model.get_user_social_context(user_id)
         
         # Combine into comprehensive profile
         user_profile = {
             'user_info': user_info,
             'interests': interests,
             'temporal_patterns': temporal_patterns,
             'social_context': social_data,
             'recent_activity': summarize_activity(recent_activity)
         }
         
         return user_profile
     ```

2. **Feed Integration and Native Ad Placement**
   
   * **Content-Ad Similarity**: Ensuring ads match surrounding content style and themes
   * **Engagement Prediction**: Forecasting specific engagement types (likes, comments, clicks)
   * **Optimal Placement**: Determining when and where to show ads in the feed
   
   * **Example Implementation**:
     ```python
     def integrate_ads_into_feed(user_profile, organic_feed_items, available_ad_slots):
         """
         Integrate ads into social feed for native appearance and relevance.
         
         Parameters:
         - user_profile: Comprehensive user profile
         - organic_feed_items: List of organic content in the feed
         - available_ad_slots: Available positions for ad placement
         
         Returns:
         - Integrated feed with organic content and ads
         """
         # Source candidate ads
         ad_candidates = source_ad_candidates(user_profile, len(available_ad_slots))
         
         # Retrieve relevant ads
         relevant_ads = retrieve_relevant_ads(
             user_id=user_profile['user_info']['id'],
             candidates=ad_candidates
         )
         
         # Filter for integrity
         filtered_ads = apply_integrity_filters(user_profile, relevant_ads)
         
         # For each potential ad slot
         integrated_feed = organic_feed_items.copy()
         used_ad_count = 0
         
         for slot_position in available_ad_slots:
             if used_ad_count >= len(filtered_ads):
                 break
                 
             # Get surrounding content for context
             surrounding_content = get_surrounding_content(
                 feed=integrated_feed,
                 position=slot_position,
                 window_size=3
             )
             
             # Find best matching ad for this context
             best_ad_index = find_best_contextual_match(
                 ads=filtered_ads,
                 context=surrounding_content,
                 user_profile=user_profile
             )
             
             if best_ad_index != -1:
                 # Insert ad into feed
                 ad_to_insert = filtered_ads.pop(best_ad_index)
                 integrated_feed.insert(slot_position + used_ad_count, ad_to_insert)
                 used_ad_count += 1
         
         return integrated_feed
     ```

3. **Multi-Modal Content Understanding**
   
   * **Image and Video Analysis**: Extracting concepts and themes from visual content
   * **Cross-Modal Matching**: Aligning ad creatives with user-preferred content styles
   * **Creative Optimization**: Selecting optimal creative elements based on user preferences
   
   * **Example Implementation**:
     ```python
     def analyze_ad_creative(ad):
         """
         Perform multi-modal analysis of ad creative for better matching.
         
         Parameters:
         - ad: Ad with creative elements to analyze
         
         Returns:
         - Creative analysis results
         """
         results = {}
         
         # Analyze text elements
         if ad.has_text:
             text_analysis = text_model.analyze(ad.text_elements)
             results['text_concepts'] = text_analysis['concepts']
             results['text_sentiment'] = text_analysis['sentiment']
             results['text_style'] = text_analysis['style']
         
         # Analyze image elements
         if ad.has_images:
             image_analysis = image_model.analyze(ad.images)
             results['image_objects'] = image_analysis['objects']
             results['image_scenes'] = image_analysis['scenes']
             results['image_style'] = image_analysis['style']
             results['image_colors'] = image_analysis['dominant_colors']
         
         # Analyze video elements
         if ad.has_video:
             video_analysis = video_model.analyze(ad.video)
             results['video_concepts'] = video_analysis['concepts']
             results['video_pacing'] = video_analysis['pacing']
             results['video_engagement_curve'] = video_analysis['engagement_curve']
         
         # Create unified creative embedding
         creative_embedding = multimodal_embedding_model.encode(ad)
         results['creative_embedding'] = creative_embedding
         
         return results
     ```

4. **Social Context and Targeting**
   
   * **Social Influence Modeling**: Identifying influential users and relationships
   * **Lookalike Audience Expansion**: Finding similar users to known engagers
   * **Social Proof Integration**: Incorporating social signals into ad presentation
   
   * **Example Implementation**:
     ```python
     def enhance_social_targeting(ad, user_profile):
         """
         Enhance ad targeting with social context.
         
         Parameters:
         - ad: The ad to enhance
         - user_profile: User profile with social information
         
         Returns:
         - Enhanced ad with social context
         """
         enhanced_ad = ad.copy()
         
         # Find social connections who engaged with this ad
         engaged_connections = find_engaged_connections(
             ad_id=ad.id,
             user_connections=user_profile['social_context']['connections']
         )
         
         if engaged_connections:
             # Add social proof elements
             enhanced_ad.social_proof = format_social_proof(engaged_connections)
             
             # Boost relevance score based on social proof strength
             social_boost_factor = calculate_social_boost(
                 engaged_connections,
                 user_profile['social_context']['connection_strengths']
             )
             enhanced_ad.relevance_score *= (1 + social_boost_factor)
         
         # Check if user is in lookalike audience
         lookalike_match = check_lookalike_audience_match(
             user_profile=user_profile,
             ad_lookalike_segments=ad.lookalike_segments
         )
         
         if lookalike_match:
             enhanced_ad.lookalike_score = lookalike_match['score']
             enhanced_ad.relevance_score *= (1 + 0.2 * lookalike_match['score'])
         
         return enhanced_ad
     ```

Multi-Modal Learning for Ad Engagement
--------------------------------------

Modern ad systems leverage multiple data modalities to create a comprehensive understanding of users, ads, and contexts. These multi-modal approaches enhance prediction accuracy and enable more nuanced targeting.

### Key Modalities for Ad Engagement Prediction

1. **User Modality**
   
   * **Explicit Profile Data**: Demographics, declared interests, account settings
   * **Behavioral Data**: Click patterns, purchase history, content consumption
   * **Temporal Patterns**: Time-of-day preferences, seasonal behaviors

2. **Ad Creative Modality**
   
   * **Visual Elements**: Images, videos, animations, colors, layouts
   * **Textual Elements**: Headlines, descriptions, calls-to-action
   * **Interactive Elements**: Formats, interactive features

3. **Contextual Modality**
   
   * **Platform Context**: Placement location, surrounding content
   * **Device Context**: Device type, screen size, connection speed
   * **Situational Context**: Location, time, weather, events

### Fusion Strategies

The choice of fusion strategy impacts both technical performance and operational feasibility:

1. **Early Fusion Implementation**
   
   ```python
   def early_fusion_model(user_features, ad_features, context_features):
       """
       Implement early fusion by concatenating features before model processing.
       
       Parameters:
       - user_features: Features from user modality
       - ad_features: Features from ad creative modality
       - context_features: Features from contextual modality
       
       Returns:
       - Engagement prediction score
       """
       # Concatenate all features
       combined_features = np.concatenate([
           user_features,
           ad_features,
           context_features
       ])
       
       # Pass through neural network
       hidden1 = dense_layer(combined_features, units=256, activation='relu')
       hidden2 = dense_layer(hidden1, units=128, activation='relu')
       hidden3 = dense_layer(hidden2, units=64, activation='relu')
       
       # Output layer
       engagement_score = dense_layer(hidden3, units=1, activation='sigmoid')
       
       return engagement_score
   ```

2. **Intermediate Fusion Implementation**
   
   ```python
   def intermediate_fusion_model(user_features, ad_features, context_features):
       """
       Implement intermediate fusion with modality-specific processing.
       
       Parameters:
       - user_features: Features from user modality
       - ad_features: Features from ad creative modality
       - context_features: Features from contextual modality
       
       Returns:
       - Engagement prediction score
       """
       # Modality-specific subnets
       user_hidden = dense_layer(user_features, units=64, activation='relu')
       
       ad_hidden1 = dense_layer(ad_features, units=128, activation='relu')
       ad_hidden2 = dense_layer(ad_hidden1, units=64, activation='relu')
       
       context_hidden = dense_layer(context_features, units=32, activation='relu')
       
       # Fusion layer - concatenate processed features
       fusion = np.concatenate([user_hidden, ad_hidden2, context_hidden])
       
       # Joint processing after fusion
       joint_hidden1 = dense_layer(fusion, units=128, activation='relu')
       joint_hidden2 = dense_layer(joint_hidden1, units=64, activation='relu')
       
       # Multi-task outputs
       click_prob = dense_layer(joint_hidden2, units=1, activation='sigmoid')
       conversion_prob = dense_layer(joint_hidden2, units=1, activation='sigmoid')
       
       return {
           'click_probability': click_prob,
           'conversion_probability': conversion_prob
       }
   ```

3. **Late Fusion Implementation**
   
   ```python
   def late_fusion_model(user_id, ad_id, context):
       """
       Implement late fusion by combining outputs from separate models.
       
       Parameters:
       - user_id: User identifier
       - ad_id: Ad identifier
       - context: Contextual information
       
       Returns:
       - Combined engagement prediction
       """
       # Get user features
       user_features = user_feature_service.get_features(user_id)
       
       # Get ad features
       ad_features = ad_feature_service.get_features(ad_id)
       
       # Get context features
       context_features = context_feature_service.extract_features(context)
       
       # User-focused model
       user_model_prediction = user_model.predict(user_features, ad_id, context)
       
       # Ad-focused model
       ad_model_prediction = ad_model.predict(ad_features, user_id, context)
       
       # Context-focused model
       context_model_prediction = context_model.predict(context_features, user_id, ad_id)
       
       # Late fusion through weighted average
       combined_score = (
           0.4 * user_model_prediction +
           0.4 * ad_model_prediction +
           0.2 * context_model_prediction
       )
       
       # Alternative: Use a meta-model for fusion
       fusion_features = [
           user_model_prediction,
           ad_model_prediction,
           context_model_prediction,
           # Additional meta-features
           user_model_confidence,
           ad_model_confidence,
           context_model_confidence
       ]
       
       meta_model_score = fusion_model.predict(fusion_features)
       
       return meta_model_score
   ```

Scaling to Billions of Users
----------------------------

Building an ad engagement prediction system that can operate at scale requires careful architecture decisions:

### Distributed Data Storage

```python
def design_data_sharding_strategy(user_base_size, regional_distribution):
    """
    Design a sharding strategy for user data.
    
    Parameters:
    - user_base_size: Total number of users
    - regional_distribution: Dictionary mapping regions to percentage of users
    
    Returns:
    - Sharding configuration
    """
    # Calculate optimal shard count based on user size
    base_shard_count = user_base_size // OPTIMAL_USERS_PER_SHARD
    
    # Adjust for growth
    shard_count = int(base_shard_count * 1.5)  # 50% growth margin
    
    # Regional shard allocation
    regional_shards = {}
    for region, percentage in regional_distribution.items():
        region_shard_count = max(1, int(shard_count * percentage / 100))
        regional_shards[region] = region_shard_count
    
    # Consistent hashing configuration
    hash_ring_config = {
        'algorithm': 'consistent_hashing',
        'virtual_nodes_per_shard': 200,
        'hash_function': 'murmur3'
    }
    
    # Tiered storage strategy
    storage_tiers = {
        'hot_tier': {
            'storage_type': 'in_memory',
            'data_retention': '24h',
            'content': ['session_data', 'active_user_embeddings']
        },
        'warm_tier': {
            'storage_type': 'ssd',
            'data_retention': '30d',
            'content': ['recent_user_activity', 'ad_performance_data']
        },
        'cold_tier': {
            'storage_type': 'object_storage',
            'data_retention': '365d',
            'content': ['historical_data', 'training_datasets']
        }
    }
    
    return {
        'shard_count': shard_count,
        'regional_allocation': regional_shards,
        'hash_ring_config': hash_ring_config,
        'storage_tiers': storage_tiers
    }
```

### Real-Time Serving Infrastructure

```python
def design_serving_infrastructure(peak_qps, p99_latency_target):
    """
    Design infrastructure for real-time ad serving.
    
    Parameters:
    - peak_qps: Peak queries per second
    - p99_latency_target: Target p99 latency in milliseconds
    
    Returns:
    - Serving infrastructure configuration
    """
    # Calculate stage-wise latency budget
    stage_latency = {
        'sourcing': int(p99_latency_target * 0.1),  # 10% of total
        'recall': int(p99_latency_target * 0.2),    # 20% of total
        'integrity': int(p99_latency_target * 0.1), # 10% of total
        'precision': int(p99_latency_target * 0.5), # 50% of total
        'delivery': int(p99_latency_target * 0.1)   # 10% of total
    }
    
    # Cache configuration
    cache_config = {
        'l1_cache': {
            'type': 'on-server',
            'size_per_instance': '4GB',
            'ttl': '5m',
            'cached_items': ['user_embeddings', 'frequent_ads']
        },
        'l2_cache': {
            'type': 'distributed',
            'size_total': '500GB',
            'ttl': '1h',
            'cached_items': ['ad_metadata', 'targeting_data']
        },
        'regional_cache': {
            'enabled': True,
            'instances_per_region': 3,
            'size_per_instance': '250GB'
        }
    }
    
    # Calculate required serving capacity
    base_servers = math.ceil(peak_qps / QPS_PER_SERVER)
    total_servers = int(base_servers * 1.5)  # 50% overhead for spikes and redundancy
    
    # Model serving strategy
    model_serving = {
        'precision_layer': {
            'primary_model': 'deep_ensemble',
            'fallback_model': 'gradient_boosting',
            'model_update_strategy': 'shadow_deployment',
            'batch_size': 64,
            'max_concurrent_batches': 32
        },
        'speculative_execution': {
            'enabled': True,
            'fast_model': 'lightgbm_slim',
            'target_model': 'deep_ensemble',
            'confidence_threshold': 0.85



