Recommendation ML/AI System Design
==================================

Recommendation tasks can be generally categorized into three types:

- **Reactive**: Responding to explicit user actions or queries
- **Proactive**: Anticipating user needs without explicit prompting
- **Feed**: Combining elements of both in a curated content stream

All recommendation tasks, regardless of whether they are reactive, proactive, or feed-based, consist of four major components:

* **Task Understanding**: Interpreting user intent and needs through explicit queries, implicit signals, or predicted interests to determine what content would be most valuable in the current context.
* **Contextual Awareness**: Incorporating session data, location, time, device information, and other situational factors to enhance relevance and personalization of recommendations.
* **Delivery**: Creating a ranked list of items that best match the interpreted task and context, optimizing for user engagement, satisfaction, and business objectives.
* **Evaluation & Feedback**: Measuring recommendation performance through metrics, A/B testing, and user feedback to continuously improve the system and adapt to changing preferences and behaviors.


Reactive Recommendation
-----------------------

Reactive recommendation systems respond to explicit user actions or queries, delivering personalized results, and often based on immediate context and historical behavior. It consists of the following core components for success.

* **Query Understanding**: Interpreting user intent through natural language processing, query expansion, and entity recognition, etc.
* **Contextual Awareness**: Incorporating session data, location, time, and device information
* **Response Generation**: Creating a ranked list of items that best match the interpreted query
* **Real-time Delivery**: Serving recommendations with minimal latency (typically <500ms end-to-end latency)

Staged Recommendation System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-pass recommendation employs a :newconcept:`Staged Filtering` approach that progressively narrows down candidate items to achieve both computational efficiency and recommendation quality.

* The :newconcept:`Sourcing Layer` acquires potential candidates from diverse sources, makes proper storage and indexing before formal retrieval begins. 
  
  * :ub:`Integrates various data sources` (e.g., databases, vector stores, service APIs) for the recommendation task and establish the initial candidate pool.
    
    * The initial candidate pool size can range from thousands to tens of millions. 
    * The data sources could be 1P (including user-generated content), 2P (partner content), or 3P (non-partner external content). 
  
  * May involve both :ub:`pre-indexed data` (e.g., pre-indexed catalog, articles/posts/reviews, web data, etc.) and :ub:`real-time data` (e.g., on-sale products, real-time prices, trending news/topics).
  * Some feasible quick business filter rules could be applied (e.g., by market region/locale, contingent upon the data source's native support).
  * Typically requires very high recall (typically 95%+).
  * Usually takes 5% to 20% of total allowable e2e latency.

* The :newconcept:`Recall Layer` retrieves a manageable subset of potentially relevant items from a vast sourced candidate pool.

  * Typically aims at :ub:`reducing candidate pool to a manageable size` of thousands or less.
  * :ub:`For pre-indexed content, recall layer can leverage large-scale search tools`, including embedding-based Approximate Nearest Neighbor (ANN) search tools like using HNSW, FAISS or ScaNN (assuming the query and content both can be properly encoded), text search tools (e.g., Elasticsearch), or collaborative filtering tools (e.g., graph walk on user-item graph).
  * :ub:`For real-time content, simpler but quicker methods are applied` (because system does not yet have time to index real-time content), such as keyword matching, graph-walk based collaborative filtering.
  * Can :ub:`manage computational complexity` through dimensionality reduction techniques (e.g., PCA); can :ub:`reduce memory footprint` through quantization.
  * Typically requires high recall (typically 80%~95%).
  * Usually takes 10% to 30% of total allowable e2e latency.

* The :newconcept:`Integrity Layer` ensures candidates passing to the precision layer meet business rules, quality standards, and policy requirements.
  
  * Applied :ub:`after the recall layer` because it requires a manageable candidate size (thousands not millions).
  * Applied :ub:`before the precision layer` because it can reduce both noise and workload for the precision layer, as the precision layer could be the most complex and computational intense layer in the recommendation flow.
  * Typically implemented as lightweight filters/validation streams running in parallel .
  * Designed for high throughput with minimal latency impact.

* The :newconcept:`Precision Layer` applies sophisticated ranking to the filtered candidate set to identify the most relevant recommendations.

  * Typically processes hundreds of candidates to produce dozens of final recommendations; focus is on precision (accuracy of ranking) rather than recall at this stage.
  * Employs :ub:`complex` machine learning models including deep neural networks (e.g., Transformers) or gradient boosted decision trees (XGBoost, LightGBM, CatBoost).
  * Uses :ub:`rich feature sets` combining user, item, and contextual signals for fine-grained personalization.
  * Often implements :ub:`multi-objective optimization` balancing relevance, diversity, freshness, and business metrics.
  * May incorporate :ub:`ensemble` methods combining multiple ranking signals for more robust predictions.
  * Usually takes 30% to 60% of total allowable e2e latency, as it performs the :ub:`most computation-intensive` work.

.. figure:: ../../_static/images/system_design/recommendation_and_ads_system_design/staged_recommendation.png
    :alt: Multi-Stage Recommendation System
    :width: 100%
    :name: fig-multi-stage-recommendation

    An example of a staged recommendation system showing the progressive filtering from millions of candidates to dozens of recommendations through multiple specialized layers.

.. note:: 
  
  The :ub:`boundary between "sourcing layer" and "recall layer" is ambiguous` in practice, and sometimes they are referred to as just the "recall layer". We prefer a more clear cut in above description, that "source layer" focuses on integrating data sources for the recommendation task. 
  
  Some feasible quick business rules can be applied (e.g., by market region/locale) given that a data source natively supports it. For example, consider a vector store, if the data is already separated according to region/local, or it supports atttribute-based filter (e.g., Amazon Bedrock Knowledge Bases), then a quick region/local based filter can be performed at the sourcing stage; otherwise, such filter might not be feasible until after the recall layer.

Model Ensemble & Latency Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model ensemble techniques can be effectively applied to both recall and precision layers to enhance recommendation quality while managing computational constraints.

* Individual models can output binary decisions and scores for candidate items. 
  
  * Model scores often use different scales and distributions. Each model can in addition provide normalized confidence categorization (e.g., high, medium, low) to enable more effective ensemble decision-making.

* **Recall Layer Ensemble**: prioritizes capturing potential relevant items over precision to ensure valuable candidates aren't prematurely filtered out.

  * Implements more lenient inclusion criteria to maintain high recall
  * Uses simple aggregation rules such as: include items recommended by one model with high confidence, or by multiple models with medium confidence
  
* **Precision Layer Ensemble**: employs more strict criteria focused on ranking accuracy

  * Can utilize majority voting for final ranking decisions, or incorporate lightweight meta-models (e.g., decision trees) that combine outputs from multiple ranking models.

In production environments, effective latency management is critical for recommendation systems, with each processing layer constrained by strict time budgets. The system can :ub:`only incorporates model results that return within allocated time windows` into ensemble decisions. The time budget constraint requires ensemble mechanisms to gracefully :ub:`handle missing or delayed outputs from component models`. :refconcept:`Rule-based ensemble` (e.g., :refconcept:`hierarchical fallback`), :refconcept:`majority voting`, :refconcept:`weighted dynamic voting` and :refconcept:`Bayesian mean average` can satisfy this requirement.

To further reduce latency, :newconcept:`Speculative Inference` techniques can be employed, generating preliminary recommendations using faster, approximate models while awaiting results from more accurate, resource-intensive and computation-intensive models. This approach is particularly beneficial in scenarios where delivering timely responses is critical:
  
  * **Serve Preliminary Recommendation**: There are scenarios where we can serve users with prelminary recommendations in a natural way, for example
    
    * **Ads**: In digital advertising platforms, promptly presenting ads is essential to capture early user attention, especially when users open a website or app. Speculative inference allows the system to quickly display initial ad recommendations using lightweight models. These preliminary ads engage users immediately, while more sophisticated models refine and update the ad content shortly thereafter, ensuring both speed and relevance.
    * **Voice Assistant**: For voice assistants, responsiveness is key to user satisfaction. When a user requests "recommend me a new song", the assistant can promptly suggest a track using a lightweight model, such as, "Would you like to try 'That's So True' by Gracie Abraham, a current top song on Billboard?" Concurrently, more advanced models analyze the user's personal profile and interaction history to generate a more tailored recommendation. If the user does not respond, or declines the initial suggestion, the system can offer the refined option, e.g., "How about 'Eyes Open'? As a Taylor Swift fan, you might enjoy her latest release."

  * **Taking Advantage Of Downstream Latency**: In other scenarios, speculative inference remains valuable, :ub:`particularly when downstream processes involve substantial latency`, and faster models can adequately serve the majority of user requests. For instance, if the fastest precision layer model requires 80ms and the most accurate model needs 300ms, with downstream result delivery taking an additional 200ms, the optimal model's results would be ready before the entire pipeline completes. We will compare the outputs of both models before the final delivery, if they align, the system can confidently present the faster model's results without delay. If they differ, initiating result delivery for the optimal model adds 200ms. However, if the faster model effectively addresses 80% of user requests, this strategy reduces overall latency without compromising recommendation quality. The remaining 20% of requests, typically being more complex, may justify the additional latency, as users might anticipate and tolerate longer response time.

.. note::

   Techniques similar to "speculative inference" is a common pratcice in industry, not just for recommendation. A faster model gives preliminary response while a more advanced model to follow up or refine the previous answer given slightly more time. Again in the Voice Assistant case, when user asks "tell me about last week's flight accident", the faster model can generate a preliminary response with a high-level summary. The more detailed response by an advanced model can be prepared while voice assistant is delivering the initial answer. The voice assistant can even ask user "Do you want to know more?" to naturally earn more time for the advanced model.

For best utilization of latency budget and responsiveness, the :ub:`staged recommendation infra often needs to support streamlined and parallelled processing with buffering mechanism`. Items passed the previous layer will be pushed to the next layer without waiting for all items being processed at the current layer. The items are queued and stored in a buffer residing in the next layer, and the next layer will examine the buffer at some cadence, and process batches of items in parallel. 

* For example, if the combined latency budget for the recall and integrity layers is 100ms, and the precision layer has a 300ms budget, the precision layer can examine the buffer every 10ms. The precision layer can decide whether to process available items according to a pre-defined rule (such as processing at least 10 items at a time), or can also make dynamic decision based on its current workload. This strategy potentially reduces the latency for delivering the first recommendation by up to 90ms, achieving best responsiveness.


Multi-Modal & Cross-Funtional Recommendation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:newconcept:`Data Modality` refers to a specific type of data representation or information channel that captures a particular aspect of the task being modeled. Each modality typically has its own structure, format, encoding method, and semantic meaning. :newconcept:`Multi-Modal Learning` systems process and integrate diverse data types to create a richer understanding across different data modalities. Typical data modalities include:

* Multimedia
  
  * **Text**: Catalog, articles/posts/reviews, query, source code (e.g., HTML), metadata, etc.
  * **Images**: Product visuals, user-uploaded content, thumbnails, etc.
  * **Video**: Content previews, trailers, instructional material, etc.
  * **Audio**: Music, podcasts, voice interactions, etc.
  
* Personalization & Context Awareness

  * :newconcept:`User Profiles`: demographic information, user preferences (declared/explicit or implicit), account history, subscription status, etc.
  * :newconcept:`User Activities & Analytics`: Clicks, purchases, page views & time, reviews & ratings (not just shopping products, thumbs up/down on like music/video recommendations are also strong rating signals), engagement patterns, behavioral analytics, etc.
  * :newconcept:`Context Signals`: Temporal context (time, date, season), spatial context (location, proximity), device context (e.g., device type), situational context (e.g., weather, current screen contents, the current song/video being played, active timers/alarms/reminders, etc., and other ongoing activities), session context (conversation, click path, etc.)

.. note::

   :ub:`Data of the same type is not necessarily a single modality`. For example, conversation context, user profile, code, etc., are all text-based, but technically they can be treated as separated modalities. A modality can be a type of data that serve a distinguished purpose.

If categorizing by if the data is pre-indexed, we might have

* Pre-Indexed Data
  
  * :newconcept:`Text Catalog & Contents` (item catalog, articles, posts, reviews, etc., and their metadata).
  * :newconcept:`Media Catalog & Contents` (images, videos, audio, etc., and their metadata).
  * :refconcept:`User Profiles` as mentioned above.
  * :newconcept:`Historical User Activities & Analytics` (:refconcept:`User Activities & Analytics` from user past interactions with the system)
  
* Real-Time Data

  * :newconcept:`Real-Time User Activities & Analytics` (:refconcept:`User Activities & Analytics` from user interactions with the system in the current interaction session)
  * :refconcept:`Context Signals` as mentioned above.
  * :newconcept:`Real-Time Contents & Information` (e.g., :newconcept:`Real-Time UGC (User Generated Content)`, popular & trending items, on-sale items, real-time prices)

In modern times, we are typically looking at the following key qualities for multi-modal learning,

1. **Unified Representations**: The modality-specific features can be transformed into a unified representation space, not only that the transformed embeddings are comparable (e.g., an image can compare to a piece of text), but also enabling models to handle diverse inputs seamlessly.
2. **Multitasking Capabilities**: Models can efficiently handle diverse tasks such as chatting, question answering, image generation, audio captioning, etc.
3. **Robustness to Missing Modalities**: Effective multimodal systems can maintain performance even when some modalities are absent or incomplete. 
4. **Scalability and Flexibility**: As the diversity and volume of multimodal data grow, models must scale efficiently and adapt to new modalities.

In industry, multi-modal learning is also referred to as the :newconcept:`cross-functional learning`. This naming emphasizes the integration of diverse skills and knowledge across various departments to achieve common organizational goals. By fostering cross-functional collaboration, organizations can enhance innovation, improve problem-solving, and adapt more effectively to complex and changing market demands. 


Multi-Modal Fusion
^^^^^^^^^^^^^^^^^^

How to integrate information from multiple modalities is a critical step for multi-modal recommendation. We assume that features can be extracted by an encoder for each modality. 

* The embedding dimensions of different modalities can be different.
* A specialized :newconcept:`Missing Modality Embedding` can be reserved for each modality to represent "missing modality". For example, adding one more dimension to embedding, with it being 1 meaning the modality is missing.

Traditionally, there are three types of :newconcept:`Modality Feature Fusion` strategies depending on when the modality features are integrated.  

1. :newconcept:`Early Fusion`: Raw data from various modalities are first processed through their respective encoders to extract features. These :ub:`modality features are immediately combined` — typically through concatenation or another lightweight integration method (e.g., cross-modal attention) — before being input into the main model for further processing.
2. :newconcept:`Intermediate Fusion`: :ub:`The modality features are fused at a deeper stage` within the main model, after some independent processing (e.g., :newconcept:`Modality-Specific Subnets`). The fusion at a deeper stage can force the modality specific subnets learn to transform each modality into a unified representation space.
   
   * The :ub:`"modality-speicific subsets" is part of the main model`, and they are trained through the learning process. On the contrary, "modality encoders" are usually off-the-shelf models not part of the learning process. 

3. :newconcept:`Late fusion` :ub:`Each modality is handled separately`, allowing for decoupled and tailored processing pipelines optimized for the unique characteristics of each modality. The :ub:`outputs from each modality pipeline are combined using lightweight techniques`, such as averaging, majority voting, or a lightweight model (e.g., a decision tree, or a GBDT model), to arrive at a final prediction.
   
   * An outstanding disadvantage of late fusion is potentially missing complex cross-modal correlations. :ub:`A simple but effective mitation could be to also take convenient meta data and numerical signals from each modality`, consider them together with modality outputs, and develop a quick GBDT model. This appraoch has minimum impact on the key advantages of late fusion - decoupled development, less blocking, and relatively lightweight fusion at the end - but could significantly mitigate the issue of missing modality correlations. However, it does require additional engineer efforts to pass modality signals to the decision layer, in addition to the modality outputs.

For both early/intermediate fusion, :ub:`task heads can be added` for multi-task training, as studies found they help with the unified representation across modalities (e.g., `UniT: Multimodal Multitask Learning with a Unified Transformer <https://arxiv.org/abs/2102.10772>`_).

In the context of LLM, integrating non-text modalities involves converting them into a format compatible with text-based processing:

     * :newconcept:`Prompt-Based Fusion`: This approach involves converting all modalities into text and combining them into a single prompt, allowing the use of pre-trained LLMs without additional training for non-text signals. This approach more aligns with "Early Fusion".
     * :newconcept:`Token-Based Fusion` Non-text modalities are projected into "tokens" in the same embedding space as the text token embeddings, through modality encoders plus lightweight adapters. Special tokens are used to mark the boundary between different modalities (e.g., "<image> ... </image><audio> ... </audio><text> ... </text>"). This approach more alignes with "Intermediate Fusion". For example,
       
       * An **image** is processed by a pre-trained image encoder to extract visual features. These features are then transformed into embeddings that align with the token embeddings used by the LLM. A lightweight adapter, such as a small multilayer perceptron (MLP), is often employed to map the image features to the appropriate token dimension. This allows the LLM to integrate visual information seamlessly alongside textual data.
       * Processing **videos** is more complex due to the temporal dimension. Typically, videos are segmented into shorter clips. Each segment is processed by a video encoder to extract spatiotemporal features, which are then converted into token embeddings compatible with the LLM. Techniques like :newconcept:`Video Moment Retrieval` can be involves identifying specific segments within a video that correspond to the user inputs, enhancing the LLM's ability to understand and respond to user inputs with long videos.

.. _fig-fusion-strategies:

.. figure:: ../../_static/images/system_design/recommendation_and_ads_system_design/multimodal_fusion_strategies.png
   :alt: Multi-Modal Fusion Strategies
   :width: 100%
   :name: fig-multimodal-fusion-strategies

   Illustrations of of multi-modal fusion strategies.

The best strategy needs to consider both technology and business (operational) factors. "Early fusion" and "intermediate fusion" share many pros and cons in common, in contrast to "late fusion". The following is a comparison.

.. raw:: html

  <table class="docutils">
    <tr>
      <th></th>
      <th colspan="2">Pros</th>
      <th colspan="2">Cons</th>
    </tr>
    <tr>
      <td><b>Early Fusion</b></td>
      <td rowspan="2">
        <strong>Technical:</strong><br />
        • Enable deeper cross-modal interactions to complex relationships<br />
        • Unified representation learning<br />
        <br />
        <strong>Operational:</strong><br />
        • Single-team ownership of the main model; other teams provide modality-specific encoders and features<br />
        • Simpler to maintain the main model with faster model development cycles<br />
        • Raw data distribution change is typically more gradual and less frequent than "outputs" in late fusion, thus less maintainance coordination<br />
      </td>
      <td>
        <strong>Technical:</strong><br />
        • Simpler architecture than intermediate fusion with lower computational cost and latency<br />
      </td>
      <td rowspan="2">
        <strong>Technical:</strong><br />
        • Higher computational requirements than post fusion<br />
        <br />
        <strong>Operational:</strong><br />
        • "Big bang" deployments instead of incremental<br />
        • Centralized team must maintain the main model, and the team may require expertise across all modalities<br />
        • Risk of single-point (single-team) failure
      </td>
      <td>
        <strong>Technical:</strong><br />
        • High-dimensional inputs can lead to overfitting<br />
        • If fusion is unweighted, then it might ignores modality importance<br />
      </td>
    </tr>
    <tr>
      <td><b>Intermediate Fusion</b></td>
      <td>
        <strong>Technical:</strong><br />
        • More flexible architecture than early fusion<br />
        • Allows to reduce dimensionality before fusion<br />
        <br />
        <strong>Operational:</strong><br />
        • Specialized teams can help experiment and develop modality-specific subnet architecture
      </td>
      <td>
        <strong>Technical:</strong><br />
        • More complex architecture than early fusion with computational cost and latency<br />
        • Integration point needs careful consideration and experiments<br />
        <br />
        <strong>Operational:</strong><br />
        • If subnets are developed by specialized teams, then it requires more cross-team coordination
      </td>
    </tr>
    <tr>
      <td><b>Late Fusion</b></td>
      <td colspan="2">
        <strong>Technical:</strong><br />
        • Usually lightweight techniques to combine outputs from different modalities and thus easier for launch and less risk for failure<br />
        <br />
        <strong>Operational:</strong><br />
        • Minimum distruption to existing teams; only a small team is needed for the lightweight decision layer<br />
        • Best for parallel development with minimal blocking <br />
        • Suitable for bias for action to launch multi-modal recommendation user experience soon without potential reorganization and heavy model development, if the urgent business need is justified
      </td>
      <td colspan="2">
        <strong>Technical:</strong><br />
        • Only shallow integration of modalities and thus may miss complex cross-modal interactions (mitigation possible to integrate convenient modality signals at decision layer, but still suboptimal)<br />
        • The decision layer is often fitted to the current "modailty outputs" and not robust against upstream changes.<br />
        <br />
        <strong>Operational:</strong><br />
        • May form modality silos because each team develops their own modality models and pipelines.<br />
        • Higher post-lauch maintainance effort. The decision layer learns the "modality outputs", and any abrupt changes in upstream models might break the decision layer.<br />
      </td>
    </tr>
  </table>

.. note::

   In practice, the naming can vary (some people lump early+intermediate together), and many real‐world pipelines actually employ hybrids (e.g., partial early/late fusion). The above three categories are reference points on a spectrum, and teams in practice often pick and choose what works best operationally.

Scaling Recommendation System For Billions Of Users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A cross-functional recommendation system that scales to billions of users must optimize infrastructure, data processing, model inference, and feedback loops while ensuring cost-effective operations. The following provides an overview of things that matter for scalability of a recommendation system. A majority of recommendation system scalability are defined by the data infrastructure and efficiency.

.. _fig-multimodal-recommendation-architecture:

.. figure:: ../../_static/images/system_design/recommendation_and_ads_system_design/multimodal_recommendation_architecture.png
   :alt: Multi-Modal Recommendation System Architecture
   :width: 100%
   :name: fig-multimodal-recommendation-architecture

   A high-level multi-modal recommendation system architecture illustrating a framework for a scalable recommendation system, including data ingestion layers, real-time and pre-indexed data processing, real-time data analytics, the staged recommendation pipeline designed, and the serving & feedback layer to support logging, experiments and feedback loop.

:ub:`Globally Distributed Architecture`:

* **Regional Deployment**: Deploy services in data centers strategically located near major user populations to minimize latency. Implement edge computing solutions to handle latency-sensitive tasks, ensuring rapid response times for end-users.​
* **Data Residency Compliance**: Adhere to regional data protection regulations (e.g., GDPR, CCPA) by partitioning data storage and processing geographically. This ensures compliance and builds user trust.​
* **Traffic Management**: Utilize global load balancing techniques to distribute incoming requests based on factors like server capacity, current load, and proximity to the user, optimizing both performance and resource utilization.​
* **Cross-Region Consistency**: Implement eventual consistency models to synchronize user profiles and recommendation data across regions, balancing data accuracy with system performance.

:ub:`Data Sharding Strategies`:

* **User-Based Sharding**: User-based sharding is a database partitioning strategy that distributes user data across multiple servers to achieve balanced workloads and facilitate horizontal scaling. This approach assigns each user's data to a specific shard, ensuring that no single server becomes a bottleneck.​
  
  * :newconcept:`Hash-Based Sharding`: A common method involves applying a hash function to a user's unique identifier (e.g., user ID or username). The hash function's output determines the shard where the user's data will reside. For example, computing hash(user_id) % number_of_shards assigns the data to a specific shard. This technique promotes an even distribution of users across shards, enhancing load balancing.
  * :newconcept:`Consistent Hashing`: To accommodate dynamic changes in the number of shards (such as adding or removing servers), `consistent hashing <https://en.wikipedia.org/wiki/Consistent_hashing>`_ is employed. Unlike traditional hashing methods that may require extensive data redistribution when the number of shards changes, consistent hashing ensures that only a minimal portion of data needs to be moved. This efficiency is achieved by mapping both data and servers to a :newconcept:`hash ring`, allowing for scalable and flexible data distribution. ​

* **Item-Based Sharding**: :newconcept:`Item-based sharding` (a.k.a. :newconcept:`attrbiute-based sharding`) is a database partitioning strategy that distributes data (typically retrieval targets) across multiple shards based on key attributes (e.g., category, popularity, name) from the items. This approach aims to enhance data retrieval efficiency and manageability by logically grouping related items together.
* **Temporal Partitioning**: Differentiate storage solutions for historical and recent data, optimizing based on access patterns and storage costs.​
* **Activity-Based Allocation**: Allocate computational resources dynamically, focusing on active users to maintain responsiveness while efficiently managing inactive user data.

:ub:`Data Storage Optimization`:

* **Tiered Storage Architecture**: Implementing a hierarchical :newconcept:`Tiered Storage System` that balances between data retrieval and storage costs. For example:

  * **Hot Tier**: In-memory databases (e.g., Redis, Memcached) provide rapid access to active session data.​
  * **Warm Tier**: SSDs store recent user interactions and frequently accessed items, balancing speed and cost.​
  * **Cold Tier**: HDDs or object storage solutions archive infrequently accessed data, optimizing cost-efficiency.

* **Utilizing Specialized Databases**: 

  * **Vector Databases**: Solutions like FAISS or Milvus efficiently store and retrieve high-dimensional embeddings, crucial for storing pre-computed text or multimedia features, and facilitating similarity searches.​
  * **Document databases**: Store, retrieve, and manage semi-structured data as documents, using formats like JSON (JavaScript Object Notation) or XML to represent data, allowing for flexible and scalable data modeling.​
  * **Temporal (Time-Series) Databases**: Databases designed for temporal data effectively manage and query time-dependent user interactions.​
  * **Graph Databases**: Graph databases model and query complex relationships between users and items, enhancing recommendation accuracy.

:ub:`Data Ingestion Pipeline`:

  * **Event Streaming Architecture**: Implementing a robust event streaming architecture is essential for :newconcept:`real-time data ingestion`. Platforms like **Apache Kafka** and **Amazon Kinesis** facilitate the continuous collection and streaming of data, ensuring low-latency data availability for downstream applications, and enabling diverse data format support, scalability, fault tolerance, etc. 
  * **Buffering and Backpressure**: To handle sudden increases in log data, the ingestion pipeline incorporates a :newconcept:`ingestion buffer` — a temporary storage area that holds and queues input data before it's processed. This buffer absorbs spikes in data volume, allowing the system to process logs at a consistent and manageable rate. When the buffer reaches its capacity, :newconcept:`backpressure` mechanisms signal the upstream to slow down or pause data ingestion, preventing buffer overflow and potential data loss. This ensures that the system remains stable even under high load conditions.
  * **Priority-Based Processing**: Prioritizing critical user interactions (e.g., clicks, purchases) in the data processing pipeline ensures timely critical updates to recommendation models.​
  * **Offline Batch Processing**: Scheduling regular :newconcept:`ETL (Extract, Transform, Load)` jobs during off-peak hours processes large volumes of data (e.g., for analysis, feed-back learning and offline experiment purposes) without impacting system performance.

:ub:`Hierarchical Data Caching`: A hierarchical caching system employs multiple levels of caches, each designed to store data based on access frequency and proximity to the user. This multi-tiered approach ensures that frequently accessed data is readily available, reducing latency and server load.​

  * L1 Cache (:newconcept:`On-Server Cache`): This is the closest cache residing within the same server as the application. It stores recently accessed data, enabling rapid retrieval for repeated requests without the need for network calls. This proximity significantly reduces access time and alleviates pressure on downstream caches or databases.​
  * L2 Cache (:newconcept:`Distributed Cache`): 
    
    * As application traffic grows, relying on a single cache can lead to bottlenecks. Distributed caching allows data to be stored across multiple servers or nodes, enabling the system to handle increased load by adding more cache servers without disrupting existing operations. ​:newconcept:`Content Delivery Networks (CDNs)` is a common method for implementing distributed caching.
    * In a distributed cache system, if one cache server fails, requests can be rerouted to another server, ensuring continuous availability of cached data. This redundancy enhances the system's fault tolerance and reliability. 
    * The cached data can be spread across multiple nodes or servers, often located in various data centers worldwide.
    * A :newconcept:`regional cache` (also sometimes called "L3 Cache") is a subset of a distributed cache, strategically placed within a specific geographic area to serve users in that region for optimizing localized performance and complying with local data residency regulations and privacy laws.
  
`Staged Recommendation System`_ as described in the earlier section, with techniques including like **model ensemble**, **speculative inference**, **streamlined and parallel processing** to scale the system. 

:ub:`Effective & Efficient Feedback Processing`: Essential for refining recommendation systems, either dynamically adapting swiftly to runtime user behaviors and preferences signals, or preserving log data for offline model/pipeline improvements. Key components include:​
  
  * **Real-Time Experimentation & Testing**: Implementing a robust experimentation framework allows for continuous optimization of the recommendation system:
    
    * **A/B Testing**: Developing a scalable :refconcept:`A/B Testing` infra to enable controlled experiments on variations of recommendation algorithms and relevance factors. This infrastructure must support high-throughput traffic allocation while maintaining statistical validity and minimizing latency impact.
      
      * The infra needs to support traffic allocation control during the experiment. The experiment can start with allocating a small amount of live traffic (e.g., 5%) for a test run on the updated recommendation pipeline to catpure any unexpected consequences. If everything turns out good, the A/B testing can gradually allocate more users to experiment the update, e.g., from 10% to 30%, to max 50%. Once the experiment concludes the udpated recommendation pipeline statistically outperform the old version in a significant way, the allocation will change to 100% (full launch).

    * **Interleaved Testing**: A real-time evaluation technique helping developers directly compare different models or the updated/old pipeline under the same user. :newconcept:`Interleaved Testing` will present items from different recommendation algorithms within a single results list and measures which algorithm/pipeline's recommendations receive more engagement, or alternatively it present results from a different model/pipeline each time. 
      
      * This approach requires fewer users than traditional A/B testing while providing direct comparative insights (becasue there are comparable results associated with the same user).
      * Interleaved Testing is usually conducted before the large-scale A/B Testing to make sense of the performance gap, observe the results and make any further improvement if possible.
      * This testing can also be leveraged to collect user preference feedback, presenting results from different models/pipelines, and asking user help to annotate.

    * **Multi-Armed Bandit Testing**: Unlike traditional A/B testing, :refconcept:`Multi-Armed Bandit` approaches dynamically adjust traffic allocation based on real-time performance, optimizing for exploration (testing new variants) and exploitation (leveraging successful variants). This methodology is particularly valuable for recommendation systems where user preferences change rapidly.


  * **Distributed Logging & Signal Aggregation**: Implementing scalable logging systems is crucial for capturing user interactions. Combining feedback signals from various sources—such as clicks, purchases, and dwell time—provides a holistic understanding of user engagement. 
  * :newconcept:`Near Real-Time (NRT) Analytics`: Processing feedback data as it arrives allows the system to quickly adapt to changing user behaviors. Stream processing frameworks such as **Apache Kafka** and **Amazon Kinesis** enable NRT analytics on multiple data streams, supporting timely updates to recommendation pipeline and models.
    
    * For example, one of the model ensemble method **weighted dynamic voting** needs runtime signals of each model's success rate or failure rate (e.g., when there is strong signal showing user accepted or denied that recommendation). NRT analytics ensure the model ensemble weights can be promptly updated.

.. note::
   
   "**near real-time**" and "**real-time**" are terms frequently used exechangeably for recommendation pipelines. That "real‐time" data can actually be slightly behind (e.g., seconds to a few minutes) due to stream‐processing lags or micro‐batch intervals. 
   
   If absolute sub‐second freshness is critical (as in trending‐news scenarios), you often have to engineer faster specialized paths that bypass the usual stream‐processing buffers. If an application truly requires hard real-time guarantees (e.g., single-digit milliseconds or microseconds, strict deadlines for control systems, etc.), you would typically use a specialized real-time system or direct in-memory message passing. But for most recommendation and analytics use cases—such as updating model signals or feeding a near real-time dashboard—Kafka and Kinesis suffice and are widely used.