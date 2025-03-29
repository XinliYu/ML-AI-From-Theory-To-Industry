Transformer Models
==================

Transformer Architecture
------------------------

The :newconcept:`Transformer Architecture` has become a foundational building block in modern deep learning, particularly for sequence modeling tasks in natural language processing, computer vision, and search/recommendation systems. Unlike the previously popular architecture :newconcept:`Recurrent Neural Networks (RNNs)`, transformers process all elements of a sequence in parallel through :newconcept:`self-attention` mechanisms. This approach enables more efficient training and better captures long-range dependencies within data. The following is an executable complete example code.

.. literalinclude:: ../../_static/code/modeling/classic_modeling/transformer.py
   :class: folding
   :name: complete_example_transformer_architecture
   :language: python
   :linenos:

Multi-Head Attention
~~~~~~~~~~~~~~~~~~~~

At the core of Transformer architecture is the :newconcept:`Multi-Head Attention` mechanism, which allows the model to jointly attend to information from different representation subspaces.

* **Motivation**: Multi-head attention is motivated by the idea that a value's embedding might have multiple "components" - part of the embedding might be about topic A, another part about topic B, and for different components the weighting might be different.
* **Implementation**: The multi-head attention divides the model dimension (``d_model``) into multiple heads (``num_heads``), where each head focuses on a different aspect of the representation:

  .. code-block:: python
     :class: folding
     :name: multi_head_attention_init

     class MultiHeadAttention(nn.Module):

        def __init__(self, d_model, num_heads):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_head = d_model // num_heads

            # Linear projections for query, key, and value transformations
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            
            # Final output projection after concatenating heads
            self.W_o = nn.Linear(d_model, d_model)

* **Scaled Dot-Product Attention**: Within each attention head, a :newconcept:`Scaled Dot-Product Attention` mechanism computes attention weights:

  .. code-block:: python
     :class: folding
     :name: scaled_dot_product_attention
     
        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            """
            Compute scaled dot-product attention between query Q, key K, and value V.
            Typically used inside each attention head.

            Scaled-doc product can be viewed as soft retrieval, where we obtain probabilistic scores for values in V through Q and K.
            The probabilistic scores are the attention scores.

            Args:
                Q (Tensor):
                    Shape (batch_size, num_heads, query_size, d_head).
                K (Tensor):
                    Shape (batch_size, num_heads, index_size, d_head).
                V (Tensor):
                    Shape (batch_size, num_heads, index_size, d_head).
                mask (Tensor, optional):
                    Shape broadcastable to (batch_size, 1, query_size, index_size),
                    typically a padding of size (batch_size, 1, 1, index_size),
                    with zero (False) entries indicating positions to mask out.

            Returns:
                output (Tensor):
                    Shape (batch_size, num_heads, query_size, d_head).
                attn_weights (Tensor):
                    Shape (batch_size, num_heads, query_size, index_size).
                    These are the softmax-normalized attention scores.
            """


            # -----------------------
            # 1) Compute raw attention scores
            # -----------------------
            # Calculate raw attention scores (before applying softmax).
            # We use negative indexes to indicate transposing the last two dimension of K. We do not use positive indexes so the code can work in a more general way, as K might have 3 or 4 dimensions with "multi-head" and "batch".
            # Dividing by the square root of `d_head` is recommended by the paper is to prevent the raw attention scores from becoming too large for large dimensions.
            # The `attn_scores` dimensions are (`batch_size`, `num_heads`, `query_size`, `index_size`)
            # (or (`batch_size`, `num_heads`, `seq_length`, `seq_length`) in the sequence processing case).
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

            # -----------------------
            # 2) Mask out invalid positions (if any)
            # -----------------------
            if mask is not None:
                # We can mask out keys/values by replacing their attention scores to near-zero values
                # Typically a padding mask of size (batch_size, 1, 1, index_size),
                #  but any mask broadcastable to (batch_size, 1, query_size, index_size) should work here.
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

            # -----------------------
            # 3) Apply softmax to get normalized weights
            # -----------------------
            # Apply softmax along the last dimension / feature dimension, i.e. normalizing each row)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # -----------------------
            # 4) Multiply weights by values (weighted sum of V)
            # -----------------------
            # Apply attention weights to values.
            # The `output` dimensions are (`batch_size`, `num_heads`, `query_size`, `d_head`)
            # (or (`batch_size`, `num_heads`, `seq_length`, `d_head`) in the sequence processing case),
            # which can be viewed as a weighted sum of features (rows) of V, weighted by the `attn_weights`,
            # or as if it had "retrieved" features from V in a weighted way.
            output = torch.matmul(attn_weights, V)
            return output, attn_weights

  * This can be viewed as a :ub:`soft retrieval operation`, obtaining probabilistic scores for values in V through query-key interactions
  * Dividing by ``sqrt(d_head)`` stabilizes gradients by preventing attention scores from becoming too large for high dimensions

* **Forward Pass Implementation**: The forward method demonstrates how multi-head attention is actually computed:

  .. code-block:: python
     :class: folding
     :name: multi_head_attention_forward

        def forward(self, Q, K, V, mask=None):
            """
            Forward pass of multi-head attention. Splits Q, K, and V into num_heads
            chunks, applies scaled dot-product attention for each head, then
            concatenates the heads back into d_model.

            Args:
                Q (Tensor): (batch_size, query_size, d_model)
                K (Tensor): (batch_size, index_size, d_model)
                V (Tensor): (batch_size, index_size, d_model)
                mask (Tensor, optional):
                    Shape broadcastable to (batch_size, 1, query_size, index_size),
                    typically a padding of size (batch_size, 1, 1, index_size),
                    with zero (False) entries indicating positions to mask out.
            Returns:
                output (Tensor):
                    (batch_size, query_size, d_model)
                attn_weights (Tensor):
                    (batch_size, num_heads, query_size, index_size)
            """
            batch_size = Q.size(0)

            # -----------------------
            # 1) Linear projections + reshape into heads
            # -----------------------
            # 1. Keep the batch size.
            # 2. Breaking down the last dimension (the feature dimension) to two dimensions of size `num_heads` and `d_head`.
            # 3. The `-1` means the `view` function will infer the corresponding dimension (the index size dimension).
            # Transposing the index size dimension (dimension 1) and the `num_heads` dimension (dimension 2) is for computational convenience
            # because the scaled-doc product will be performed per batch per head.
            # After transposing, the dimensions are (`batch_size`, `num_heads`, `query_size`, `d_head`) for Q, and (`batch_size`, `num_heads`, `index_size`, `d_head`) for K, V
            # (or all (`batch_size`, `num_heads`, `seq_length`, `d_head`) in the sequence self-attention case).
            Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
            K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
            V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

            # -----------------------
            # 2) Scaled dot-product attention
            # -----------------------
            # The `output` dimensions are (`batch_size`, `num_heads`, `query_size`, `d_head`)
            # (or (`batch_size`, `num_heads`, `seq_length`, `d_head`) in the sequence processing case),
            # The `attn_scores` dimensions are (`batch_size`, `num_heads`, `query_size`, `index_size`)
            # (or (`batch_size`, `num_heads`, `seq_length`, `seq_length`) in the sequence processing case).
            output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

            # -----------------------
            # 3) Recombine heads
            # -----------------------
            # Combine the heads and reshape back to original input embedding dimension sizes.
            # 1. Transposing the dimensions back to (`batch_size`, `query_size`, `num_heads`, `d_head`) (or (`batch_size`, `seq_length`, `num_heads`, `d_head`) in the sequence processing case).
            # 2. The `contiguous` method enforces the underlying tensor representation be continuous.
            # 3. The `view` method merges the last two dimensions, so it becomes (`batch_size`, `query_size`, `d_model`) (or (`batch_size`, `seq_length`, `d_model`) in the sequence processing case).
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

            # -----------------------
            # 4) Final linear projection
            # -----------------------
            # Apply output transformation on the `d_model` dimension.
            # `output` dimensions unchanged.
            output = self.W_o(output)

            # NOTE:
            # The above approach is mathematically equivalent to perform multiple Q/K/V scaled-dot-products and then concatenate them.
            # However, above view/transpose operations are more efficient implementation.

            return output, attn_weights

  * The key operations include:

    1. Projecting inputs through linear layers :math:`Q, K, V`.
    2. Splitting the model dimension across multiple heads.
    3. Applying attention independently to each head.
    4. Concatenating outputs from all heads.
    5. Applying the final output projection :math:`W_o` to produce the combined attention result.

  * The code implementation is mathematically equivalent to perform multiple Q/K/V scaled-dot-products and then concatenate them; however, :ub:`the view/transpose operations are more efficient implementation`.

Positional Encoding
~~~~~~~~~~~~~~~~~~~

Since multi-head attention process input tokens simultaneously (in parallel), they lack inherent sequential information. :newconcept:`Positional Encoding` addresses this limitation:

* **Purpose**: Positional encoding can be viewed as a type of :ub:`multi-modal learning`.

  * It combined position information with the token embeddings, allowing the model to distinguish between different positions in the sequence.
  * In the canonical transformer architecture, it is by simply adding the positional encodings to the token embeddings before input to the encoder layers, as well as adding to the encoder output before input to the decoder layers.

* **Implementation**: Uses sine and cosine functions of different frequencies to create unique position identifiers:

  .. code-block:: python
     :class: folding
     :name: transformer_positional_encoding

        class PositionalEncoding(nn.Module):
            """
            Positional encoding injects information about the absolute or relative position
            of tokens in the sequence, enabling the model to capture ordering context.

            The sine/cosine pattern allows the model to generalize to positions beyond
            the training range and provides unique encodings for each position.

            Implementation details:
              - Sine values assigned to even indices.
              - Cosine values assigned to odd indices.
              - Dimensions with higher indices (features) oscillate faster.
              - Final tensor is registered as a buffer, so it's not trainable.
            """

            def __init__(self, d_model, max_seq_length=5000):
                super().__init__()

                pe = torch.zeros(max_seq_length, d_model)
                position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

                # the following uses start:stop:step notation
                pe[:, 0::2] = torch.sin(position * div_term)  # Assigns sine values to even indices (0,2,4,...)
                pe[:, 1::2] = torch.cos(position * div_term)  # Assigns cosine values to odd indices (1,3,5,...)

                # The original pe tensor has shape [`max_seq_length`, `d_model`],
                # and `unsqueeze(0)` adds a new dimension at index 0.
                # After unsqueeze, shape becomes [1, `max_seq_length`, `d_model`],
                # and this extra dimension allows for batch processing.

                # `self.register_buffer` in PyTorch is a method used to register a tensor as a buffer
                # that should be saved along with model parameters during model.state_dict() calls,
                # but is not considered a model parameter (meaning it doesn't receive gradients during backpropagation).
                self.register_buffer('pe', pe.unsqueeze(0))

            def forward(self, x):
                """
                Add positional encoding to input embeddings.

                Args:
                    x (Tensor):
                        Shape (batch_size, seq_length, d_model).

                Returns:
                    (Tensor):
                        Same shape as input, with positional encodings added elementwise.
                """

                # If x has shape [`batch_size`, `seq_length`, `d_model`]
                # pe[:, :x.size(1)] will broadcast from [1, `seq_length`, `d_model`]
                # to match x's shape.
                return x + self.pe[:, :x.size(1)]

* **Key Properties**:

  * Projects integer positions onto a circle in 2D dimension, using sine/cosine values to identify positions
  * Includes dimension index in the formula to:
    1. Match the model feature dimension size (``d_model``)
    2. Help each positional encoding uniquely identify a position

.. note::
    The positional encoding has deep connection with :newconcept:`Fourier Transform` in mathematics & signal processing (Fourier transform decomposes a signal into different frequency components). Given the formula of positional encoding $\text{position} \cdot \text{div_term}$ where

    .. math::

        \text{div_term} = \text{exp}(\text{arange}(0, \text{d_model}, 2) \cdot (-\text{log}(10000.0) / \text{d_model}))

    We can see it is using sine and cosine functions with different frequencies for each dimension, as illustrated in the figure below.

    .. figure:: /_static/images/modeling/classic_modeling/transformer_models/transformer_positional_encoding.png
        :alt: Transformer Positional Encoding Interpretation
        :width: 80%
        :align: center

        Visualization of positional encoding frequencies

    Lower dimensions use lower frequencies (changing slowly across positions, so it is more distinguishing long-range positional difference), while higher dimensions use higher frequencies (changing rapidly across positions, more distinguishing short-range local positional difference). This multi-scale approach gives the model the :ub:`ability to reason about positional relationships at different levels of granularity`.

Positionwise Feed-Forward Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Between attention layers, the Transformer employs :newconcept:`Positionwise Feed-Forward Networks`, a fully connected feedforward neural network applied independently and identically to each data point in the sequence. This means that the same feedforward network is applied to each data point's representation without any interaction between different positions at this stage.

* **Implementation**: Two linear transformations with a ReLU activation in between. The hidden layer size ``d_ff`` is usually 2x to 4x larger than ``d_model``. Both non-linear activation and larger hidden layer size enhance the model's representational capacity:

  .. code-block:: python
     :class: folding
     :name: transformer_positionwise_feedforward

        class PositionWiseFeedForward(nn.Module):
            """
            A feedforward network applied independently at each time step (position).
            In other words, for each position in the sequence, the same two-layer MLP is used:
              1) Linear transform: d_model -> d_ff
              2) Non-linear activation (ReLU)
              3) Linear transform: d_ff -> d_model

            d_ff is larger than d_model. For example, the classic setting is 4x larger.
            This larger hidden dimension in the feed-forward sub-layer helps increase the representational capacity of the network,
            allowing more complex transformations to be applied at each position.
            After this larger intermediate projection, it then projects back down to d_model.

            Because it is applied "position-wise," there's no interaction across different sequence positions here.
            """

            def __init__(self, d_model, d_ff):
                super().__init__()
                self.fc1 = nn.Linear(d_model, d_ff)
                self.fc2 = nn.Linear(d_ff, d_model)
                self.relu = nn.ReLU()

            def forward(self, x):
                """
                Forward pass through the position-wise feed-forward network.

                Args:
                    x (Tensor):
                        Input tensor of shape (batch_size, seq_length, d_model).

                Returns:
                    (Tensor):
                        Output tensor of shape (batch_size, seq_length, d_model).
                """
                return self.fc2(self.relu(self.fc1(x)))


Layer Normalization
~~~~~~~~~~~~~~~~~~~

:newconcept:`Layer Normalization` is applied after each sub-layer within both encoder and decoder stacks to stabilize the learning process (see `Numerical Stability <https://en.wikipedia.org/wiki/Numerical_stability>`_):

* Layer Normalization :ub:`applies z-score normalization` across the feature dimension for each sample in the batch (i.e., applied across the feature dimension). For an input vector :math:`\mathbf{x}` with features :math:`x_1, x_2, ..., x_d`, the formula is:

  .. math::

     \text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

  Where:
  
  * :math:`\mu = \frac{1}{d} \sum_{i=1}^{d} x_i` is the mean across features
  * :math:`\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2` is the variance across features
  * :math:`\epsilon` is a small constant added for numerical stability (preventing division by zero)
  * :math:`\gamma` and :math:`\beta` are learnable parameters for scaling and shifting

* **Implementation**: In PyTorch, this is implemented as ``nn.LayerNorm`` under ``torch.nn``.

* **Purpose**:
  
  * :ub:`Preventing covariate shift` and thus stabilizing the distribution of activations.
  * :ub:`Enables higher learning rates` and faster convergence.
  * :ub:`Reduces sensitivity to initialization` and helps gradient flow through deep networks
  * Unlike batch normalization, works effectively with :ub:`variable-length sequences and small batch sizes`

* :ub:`Applied with random dropout and residual connection`: In Transformer implementations, Layer Normalization is applied together with :newconcept:`residual connections` using the formula ``LayerNorm(x + Dropout(Sublayer(x)))``.

  .. code-block:: python
     
     # In the encoder layer
     def forward(self, x, mask=None):
         # Self-attention with residual connection and layer normalization
         attn_output, _ = self.self_attn(x, x, x, mask)
         x = self.norm1(x + self.dropout(attn_output))  # Add & Norm
         
         # Feed-forward with residual connection and layer normalization
         ff_output = self.feed_forward(x)
         x = self.norm2(x + self.dropout(ff_output))  # Add & Norm
         
         return x

  This "Add & Norm" pattern (residual connection followed by layer normalization) is empirically found effective for training deep Transformer networks.

  * Improved gradient flow through direct pathways
  * Easier optimization of residual mappings compared to direct mappings
  * Ability to train much deeper networks without performance degradation
  * Ensemble-like behavior that improves generalization

.. admonition:: Batch Normalization vs. Layer Normalization
   :class: note
   :newconcept:`Batch Normalization` is a common normalizing technique applied before the Transformer era. While both normalization techniques serve similar purposes, they differ in several key aspects:
   
   .. list-table::
      :header-rows: 1
      
      * - Aspect
        - Batch Normalization
        - Layer Normalization
      * - **Normalization Dimension**
        - Across batch dimension for each feature
        - Across feature dimension for each sample
      * - **Formula Difference**
        - μ and σ calculated per feature across batch
        - μ and σ calculated per sample across features
      * - **Batch Size Dependency**
        - Performance degrades with small batches
        - Works well regardless of batch size
      * - **Sequence Handling**
        - Struggles with variable-length sequences
        - Handles variable-length sequences naturally
      * - **Inference Behavior**
        - Uses pre-computed mean and variance from training during inference (problematic for small batches)
        - Same computation in training and inference
      * - **Parallel Processing**
        - Less effective for distributed training
        - More suitable for distributed training
      * - **Common Applications**
        - CNNs, fixed-size inputs
        - RNNs, Transformers, NLP models
   
   Transformers use Layer Normalization because it better accommodates variable sequence lengths, works with any batch size, and maintains consistent behavior between training and inference.


Encoder-Decoder Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Transformer follows an :newconcept:`Encoder-Decoder Architecture`, with the encoder processing the input sequence and the decoder generating the output sequence.

.. _code-transformer-encoder:

* **Encoder Layer**: Each encoder layer consists of:

  1. Multi-head :ub:`self-attention` mechanism to mix the encoder input embeddings.
  2. :ub:`Position-wise feed-forward` network.
  3. :ub:`Layer norm with dropout and residuals` twice after each of above self-attention and feed-forward layers.

  Also note that :ub:`positional encoding is not applied inside encoder/decoder layers`. They have been added to the encoder/decoder input token embeddings (early fusion).

     .. code-block:: python
          :class: folding
          :name: transformer_encoder_layer

            class EncoderLayer(nn.Module):
                """
                The Encoder layer is composed of:
                1) Multi-head self-attention over the input sequence.
                2) Position-wise feed-forward network.
                3) Each sub-layer is followed by a residual connection and LayerNorm.

                The output size is the same as the input embeddings shape: (batch_size, src_seq_len, d_model).
                """
                def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
                    super().__init__()
                    self.self_attn = MultiHeadAttention(d_model, num_heads)
                    self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

                    # LayerNorm is normalizing the feature dimension using z-score (with perturbed variance)
                    self.norm1 = nn.LayerNorm(d_model)
                    self.norm2 = nn.LayerNorm(d_model)

                    self.dropout = nn.Dropout(dropout)

                def forward(self, x, mask=None):
                    """
                    Args:
                        x (Tensor):
                            The input embeddings/hidden states of shape (batch_size, src_seq_len, d_model).
                        mask (Tensor, optional):
                            A padding mask for the source sequence.
                            Typically shaped (batch_size, 1, 1, src_seq_len).
                            Defaults to None if no mask is used.

                    Returns:
                        x (Tensor):
                            The output of this encoder layer, shape (batch_size, src_seq_len, d_model).
                    """

                    # -----------------------
                    # 1) Self-Attention
                    # -----------------------
                    # Each position in 'x' attends to other positions in the same sequence.
                    # The shapes inside attention temporarily become (batch_size, num_heads, src_seq_len, d_head).
                    attn_output, _ = self.self_attn(x, x, x, mask)

                    # Residual connection + dropout + LayerNorm
                    x = self.norm1(x + self.dropout(attn_output))
                    # x remains (batch_size, src_seq_len, d_model)

                    # -----------------------
                    # 2) Position-Wise Feed-Forward
                    # -----------------------
                    # Transform each position from d_model -> d_ff -> d_model independently.
                    ff_output = self.feed_forward(x)

                    # Residual connection + dropout + LayerNorm
                    x = self.norm2(x + self.dropout(ff_output))
                    # x remains (batch_size, src_seq_len, d_model)

                    return x

* **Decoder Layer**: Each decoder layer has:

  1. Causal-masked multi-head :ub:`self-attention` on the decoder inputs

     * Decoder inputs are usually the training targets.
     * The mask is a combined padding mask and causal mask to prevent attending to future positions.
     * This causal style inference needs a "seed" at the beginning the decoder targets, therefore a :newconcept:`start-of-sequence special token` is needed at the beginning of decoder input sequence.
     * During inference, the decoder needs to identify when to stop, therefore a :newconcept:`end-of-sequence special token` is also needed at the end of decoder input sequence.

  2. Multi-head :ub:`cross-attention` over encoder outputs
  3. :ub:`Position-wise feed-forward` network.
  4. :ub:`Layer norm with dropout and residuals` three times after each of above layers.

  Decoder input embeddings usually represent the learning targets. The :newconcept:`decoder vocabulary` consists of all possible decoder learning targets. Under transformer, :ub:`each decoder input embedding functions like a query to retrieve a mix from encoder output embeddings` to represent learning target.

  .. _code-transformer-decoder:

     .. code-block:: python
          :class: folding
          :name: transformer_decoder_layer

            class DecoderLayer(nn.Module):
                """
                The Decoder layer contains:
                1. Masked self-attention over the (partial) target sequence.
                2. Cross-attention over the encoder's output.
                3. Position-wise feed-forward network.
                4. Each above sub-layer is followed by a residual connection and LayerNorm.

                The output size is the same as the decoder input embeddings, both (batch_size, target_seq_len, d_model).
                """
                def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
                    super().__init__()
                    self.self_attn = MultiHeadAttention(d_model, num_heads)
                    self.cross_attn = MultiHeadAttention(d_model, num_heads)
                    self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

                    self.norm1 = nn.LayerNorm(d_model)
                    self.norm2 = nn.LayerNorm(d_model)
                    self.norm3 = nn.LayerNorm(d_model)

                    self.dropout = nn.Dropout(dropout)

                def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
                    """
                    Args:
                        x (Tensor): Decoder input embeddings of shape (batch_size, target_seq_len, d_model).
                        enc_output (Tensor): Final encoder outputs of shape (batch_size, source_seq_len, d_model).
                        src_mask (Tensor or None): Padding mask for encoder outputs, shape (batch_size, 1, 1, source_seq_len).
                        tgt_mask (Tensor or None): Combined padding + causal (future) mask for decoder inputs,
                                                   shape (batch_size, 1, target_seq_len, target_seq_len).

                    Returns:
                        x (Tensor): Decoder layer output of shape (batch_size, target_seq_len, d_model).
                    """

                    # -----------------------
                    # 1) Masked Self-Attention
                    # -----------------------
                    # Decoder attends to itself (partial target sequence).
                    # Causal mask ensures a token can’t attend to future positions.
                    self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
                    x = self.norm1(x + self.dropout(self_attn_output))

                    # -----------------------
                    # 2) Cross-Attention
                    # -----------------------
                    # Each decoder input embedding retrieves a mixture from encoder output
                    cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
                    x = self.norm2(x + self.dropout(cross_attn_output))

                    # -----------------------
                    # 3) Position-Wise Feed-Forward
                    # -----------------------
                    # Applies two linear layers (d_model -> d_ff -> d_model)
                    # at each position independently.
                    ff_output = self.feed_forward(x)
                    x = self.norm3(x + self.dropout(ff_output))

                    return x

* **Attention Masking**: Two types of masks are employed:

  1. **Source Padding Mask**: Handles padding in the input sequence
  2. **Target Mask**: Combines padding mask with a future-masking mechanism to prevent the decoder from accessing future tokens

  .. _code-transformer-masking:

  .. code-block:: python
       :class: folding
       :name: transformer_masking

        def generate_mask(self, src, tgt):
            """
            Generate:
             - src_mask to hide padding tokens in the encoder input.
             - tgt_mask to hide padding and future positions in the decoder input.

            Args:
                src (Tensor): (batch_size, src_seq_len) of token indices.
                tgt (Tensor): (batch_size, tgt_seq_len) of token indices.

            Returns:
                src_mask (Tensor): (batch_size, 1, 1, src_seq_len) with 1/True = keep, 0/False = mask out.
                tgt_mask (Tensor): (batch_size, 1, tgt_seq_len, tgt_seq_len) combined padding + causal mask.
            """

            # The `unsqueeze(1).unsqueeze(2)` operations add extra dimensions to the `src` tensor, transforming its shape from [`batch_size`, `seq_length`] to [`batch_size`, 1, 1, `seq_length`]
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

            # The `unsqueeze(1).unsqueeze(3)` operations add extra dimensions to the `tgt` tensor, transforming its shape from [`batch_size`, `seq_length`] to [`batch_size`, 1, `seq_length`, 1]
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
            seq_length = tgt.size(1)
            # Prevent the decoder from accessing future tokens in the target sequence.
            # The resulting `causal_mask` is a boolean mask of shape [1, `seq_length`, `seq_length`], with lower triangle (diagonal included) being True and all others being False.
            # It looks like this for a sequence length of 5:
            # [[[ True, False, False, False, False],
            #   [ True,  True, False, False, False],
            #   [ True,  True,  True, False, False],
            #   [ True,  True,  True,  True, False],
            #   [ True,  True,  True,  True,  True]]]
            # The added extra dimension enables later `tgt_mask & causal_mask` operation.
            # The `causal_mask`'s new shape [1, `seq_length`, `seq_length`] is broadcastable to `tgt_mask`'s shape [`batch_size`, 1, 1, `seq_length`]
            causal_mask = ~(torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1).bool())

            # Consider the target is represented by single tensor tgt = [[5, 3, 7, 0, 0]] (i.e., batch size is 1), where 0 denotes padding tokens.
            # The bitwise AND operation results in:
            # [[[True, False, False, False, False],
            #    [True,  True, False, False, False],
            #    [True,  True,  True, False, False],
            #    [False, False, False, False, False],
            #    [False, False, False, False, False]]]
            tgt_mask = tgt_mask & causal_mask

            return src_mask, tgt_mask

* **Complete Transformer Model**: The full Transformer combines multiple encoder and decoder layers.

  * An example task is summary, where you have input tokens for both the encoder layers and decoder layers.

    .. code-block:: python
        :class: folding
        :name: transformer_input_example_summary

        def create_dummy_summarization_data(num_examples=10):
            articles = [
                           "the cat sat on the mat and watched the birds fly by in the clear blue sky",
                           "scientists discover new species of butterfly in the amazon rainforest last summer",
                           "local restaurant wins award for best pizza in the city fifth year in row",
                           "students develop innovative app to help reduce food waste in school cafeterias",
                           "new study shows benefits of regular exercise on mental health and productivity",
                           "artist creates stunning mural celebrating community diversity and unity",
                           "volunteer group organizes successful beach cleanup removing plastic waste",
                           "tech company launches eco friendly laptop made from recycled materials",
                           "urban garden project transforms abandoned lot into community vegetable garden",
                           "music festival brings together local talents raising funds for education"
                       ] * (num_examples // 10 + 1)

            summaries = [
                            "cat watches birds from mat",
                            "new butterfly species found in amazon",
                            "restaurant wins best pizza award again",
                            "students create food waste reduction app",
                            "exercise improves mental health study finds",
                            "artist paints community unity mural",
                            "volunteers clean beach of plastic",
                            "eco friendly laptop launches",
                            "community garden replaces empty lot",
                            "local music festival supports education"
                        ] * (num_examples // 10 + 1)

            return articles[:num_examples], summaries[:num_examples]

  * This is a complete transformer model assembling the encoder/decoder layers.

    .. code-block:: python
        :class: folding
        :name: transformer_assembled_encoder_decoder_layers

        class Transformer(nn.Module):
            """
            The default Transformer model adopts encoder-decoder architecture, where the encoder input and decode input could be different (e.g., translation).
            """

            def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length=5000,
                         dropout=0.1):
                super().__init__()

                self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
                self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
                self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

                self.encoder_layers = nn.ModuleList([
                    EncoderLayer(d_model, num_heads, d_ff, dropout)
                    for _ in range(num_layers)
                ])

                self.decoder_layers = nn.ModuleList([
                    DecoderLayer(d_model, num_heads, d_ff, dropout)
                    for _ in range(num_layers)
                ])

                self.fc = nn.Linear(d_model, tgt_vocab_size)
                self.dropout = nn.Dropout(dropout)

            def generate_mask(self, src, tgt):
                # ... as implemented in above ...

            def forward(self, src, tgt):
                """
                Forward pass of the Transformer.

                Args:
                    src (Tensor): (batch_size, src_seq_len) of token indices for the encoder.
                    tgt (Tensor): (batch_size, tgt_seq_len) of token indices for the decoder.

                Returns:
                    logits (Tensor): (batch_size, tgt_seq_len, tgt_vocab_size)
                                     unnormalized scores for each token in the target sequence.
                """
                # -----------------------
                # 1) Generate Attention Masks
                # -----------------------
                src_mask, tgt_mask = self.generate_mask(src, tgt)

                # -----------------------
                # 2) Encoder
                # -----------------------

                # Embed and add positional encoding (early fusion of position and sequence embeddings)
                enc_input = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

                # Pass through several encoder layers
                enc_output = enc_input
                for enc_layer in self.encoder_layers:
                    enc_output = enc_layer(enc_output, src_mask)

                # -----------------------
                # 3) Decoder
                # -----------------------

                # Embed and add positional encoding (early fusion of position and sequence embeddings)
                dec_input = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
                dec_output = dec_input

                # Pass through several decoder layers
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

                # -----------------------
                # 4) Final Output Projection
                # -----------------------
                # Project decoder output to vocab dimension
                logits = self.fc(dec_output)
                return logits


GPT Architecture
----------------

The :newconcept:`GPT (Generative Pre-Trained Transformer)` architecture represents a significant evolution of the original Transformer model, designed specifically for generative tasks. Unlike the standard encoder-decoder Transformer, GPT employs only the decoder component with modifications to :ub:`enable autoregressive text generation` capabilities. This section is diving into the publicly disclosed GPT2 architecture.

GPT2 model architecture differs from the original Transformer in several ways:

* **Decoder-Only Design**: GPT uses only a stack of modified decoder blocks without the encoder component.

  * **No Cross-Attention**: As a result, GPT's decoder blocks lack the encoder-decoder cross-attention mechanism found in the original Transformer's decoder.

* **Pre-LayerNorm Architecture**: GPT found layer normalization before attention and feed-forward operations rather than after can enhance training stability.

  * **Direct Residual Connections**: As a result, the residual connections are outside the layer norm to improve gradient flow in deep networks.

* :newconcept:`Weight Tying`: GPT implements parameter sharing between the input embedding matrix and the output projection matrix.

  * With "Weight Tying", the GPT decoder can be interpreted as inferencing weights across token feature dimensions, so that the weighted sums of the features can serve as logits across the vocabulary.

* **Autoregressive Generation**: GPT generates tokens sequentially, with each new token conditioned on all previously generated tokens.

GPT Decoder
~~~~~~~~~~~

At the core of GPT is the :newconcept:`GPT Decoder Block`, a modified version of the original `Transformer decoder layer <02_transformer_models.html#code-transformer-decoder>`_.

  * For the original Transformer decoder, it is "self-attention $\\rightarrow$ dropout + residual connection + layer norm $\\rightarrow$ cross-attention $\\rightarrow$ dropout + residual connection + layer norm $\\rightarrow$ position-wise feedforward $\\rightarrow$ dropout + residual connection + layer norm".
  * For GPT decoder, it is "layer norm $\\rightarrow$ self attention $\\rightarrow$ dropout + residual connection $\\rightarrow$ layer norm $\\rightarrow$ position-wise feedforward $\\rightarrow$ dropout + residual connection".

.. code-block:: python
   :class: folding
   :name: gpt2_decoder_block

    class GPTDecoderBlock(nn.Module):
        """
        GPT style decoder block using transformer components.

        Key differences:
        1. Doesn't have cross-attention to attend to encoder outputs like in Transformer's `DecoderLayer`, because GPT employs encoder-only design.
        2. Uses pre-LayerNorm (before attention); Transformer's `DecoderLayer` uses post-LayerNorm (after attention).
        3. Applies direct residual connection, where the residual connections outside layer norm to improve gradient flow in deep networks.
        """

        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            # Using the MultiHeadAttention from transformer
            self.self_attn = MultiHeadAttention(d_model, num_heads)
            # Using the PositionWiseFeedForward from transformer
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            # Pre-LayerNorm architecture
            # Recent researches found Pre-LayerNorm (GPT-style) is generally more stable during training,
            # and helps avoid vanishing/exploding gradients in deep networks,
            # and therefore it allows for training deeper models and using larger learning rates.
            x_norm = self.norm1(x)

            # MultiHeadAttention expects Q, K, V, they all equal to `x_norm` here for post-norm self-attention.
            attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, mask)
            # The previous `x` is added to the transformed output, creating a residual/skip path.
            # 1. Provides direct paths for gradients to flow backward.
            # 2. Helps combat vanishing gradients in deep networks, and therefore allows for training much deeper networks effectively.
            x = x + self.dropout(attn_output)

            # Feed-forward with pre-LayerNorm
            x_norm = self.norm2(x)
            ff_output = self.feed_forward(x_norm)
            # The previous `x` is added to the transformed output, creating a residual/skip path.
            x = x + self.dropout(ff_output)

            return x

The full GPT model, represented by the ``GPTDecoder`` class, assembles multiple decoder blocks into a cohesive architecture that output logits in its ``forward`` function.

* The ``generate_mask`` function is consistent with the `Transformer causal mask <02_transformer_models.html#code-transformer-masking>`_.
* The output logits will be further processed for next-token generation, with various hyper-parameters (e.g., temperature, top_p, top_k) and different strategies (e.g., beam search).

.. code-block:: python
   :class: folding
   :name: gpt2_decoder_class

    class GPTDecoder(nn.Module):
        def __init__(self,
                     vocab_size,           # Size of vocabulary (number of possible tokens)
                     d_model=768,          # Dimension of embeddings and hidden states
                     num_heads=12,         # Number of attention heads
                     num_layers=12,        # Number of decoder layers
                     d_ff=3072,            # Dimension of feed-forward layer
                     max_seq_length=1024,  # Maximum sequence length supported
                     dropout=0.1,          # Dropout rate
                     pad_token_id=0):      # ID of padding token
            super().__init__()

            self.vocab_size = vocab_size
            self.max_seq_length = max_seq_length
            self.pad_token_id = pad_token_id

            # Token embedding
            # Input shape: (batch_size, seq_length) of token IDs
            # Output shape: (batch_size, seq_length, d_model)
            self.token_embedding = nn.Embedding(vocab_size, d_model)

            # Positional encoding
            # Input shape: (batch_size, seq_length, d_model)
            # Output shape: (batch_size, seq_length, d_model)
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

            # Dropout layer - same shape in and out
            # Input/Output shape: (batch_size, seq_length, d_model)
            self.dropout = nn.Dropout(dropout)

            # Decoder layers - stack of identical transformer blocks
            # Each layer:
            # Input shape: (batch_size, seq_length, d_model)
            # Output shape: (batch_size, seq_length, d_model)
            self.decoder_layers = nn.ModuleList([
                GPTDecoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])

            # Final layer normalization
            # Input shape: (batch_size, seq_length, d_model) or (batch_size, d_model) during inference
            # Output shape: Same as input
            self.norm_final = nn.LayerNorm(d_model)

            # Linear projection to vocabulary (language model head)
            # Input shape: (batch_size, seq_length, d_model) or (batch_size, d_model) during inference
            # Output shape: (batch_size, seq_length, vocab_size) or (batch_size, vocab_size) during inference
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

            # Weight tying: Share parameters between token embedding and output projection
            # Saves vocab_size × d_model parameters (e.g., 50,257 × 768 ≈ 38.6M parameters for GPT-2)
            self.lm_head.weight = self.token_embedding.weight

        def generate_mask(self, input_ids):
            """
            Generate a mask for self-attention that handles both padding tokens and causal masking.
            Adapted from the Transformer's generate_mask method.

            Args:
                input_ids (Tensor): Input token ids of shape (batch_size, seq_length)

            Returns:
                mask (Tensor): Combined padding and causal mask
            """
            # Create padding mask (batch_size, 1, 1, seq_length)
            # True = keep, False = mask out (padding)
            padding_mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)

            # Create causal mask (1, seq_length, seq_length)
            seq_length = input_ids.size(1)
            causal_mask = ~(torch.triu(torch.ones(1, seq_length, seq_length, device=input_ids.device), diagonal=1).bool())

            # Combine masks with bitwise AND (broadcasting handles the different shapes)
            # Both masks have True for positions to keep/attend to and False for positions to mask out
            combined_mask = padding_mask & causal_mask

            return combined_mask

        def forward(self, input_ids, inference=False):
            """
            Forward pass through the GPT model.

            Args:
                input_ids: Tensor of shape (batch_size, seq_length) containing token IDs
                inference: Boolean flag for inference mode (only return last token's logits)

            Returns:
                - Training mode: logits of shape (batch_size, seq_length, vocab_size)
                - Inference mode: logits of shape (batch_size, vocab_size) for just the final token
            """
            # Token embedding: (batch_size, seq_length) → (batch_size, seq_length, d_model)
            x = self.token_embedding(input_ids)

            # Add positional encoding: (batch_size, seq_length, d_model) → (batch_size, seq_length, d_model)
            x = self.positional_encoding(x)

            # Apply dropout: shape unchanged
            x = self.dropout(x)

            # Generate mask for attention
            # Shape: (batch_size, 1, seq_length, seq_length)
            mask = self.generate_mask(input_ids)

            # Pass through decoder layers
            # Each layer maintains shape: (batch_size, seq_length, d_model)
            for layer in self.decoder_layers:
                x = layer(x, mask)

            # Apply final normalization
            if inference:
                # During inference, only process the last token position
                # x[:, -1, :] will extract the last
                # (batch_size, seq_length, d_model) → (batch_size, d_model)
                x = self.norm_final(x[:, -1, :])

                # Project to vocabulary
                # (batch_size, d_model) → (batch_size, vocab_size)
                logits = self.lm_head(x)
            else:
                # During training, process all positions
                # Shape maintained: (batch_size, seq_length, d_model)
                x = self.norm_final(x)

                # Project to vocabulary
                # (batch_size, seq_length, d_model) → (batch_size, seq_length, vocab_size)
                logits = self.lm_head(x)

            return logits



.. admonition:: Weight Tying in Large Language Models
   :class: note

   **Weight Tying** is a parameter sharing technique where models constrain certain layers to use identical weights, most commonly between the input token embedding matrix and the output projection layer (LM head).

   .. math::

      \mathbf{W}_\text{embed} = \mathbf{W}_\text{lm_head}^T

   Where :math:`\mathbf{W}_\text{embed} \in \mathbb{R}^{V \times d}` is the embedding matrix mapping tokens to vector space, and :math:`\mathbf{W}_\text{lm_head} \in \mathbb{R}^{d \times V}` projects hidden states back to vocabulary logits.

   **Benefits:**

   * **Parameter Efficiency**: For a vocabulary size of 50K and embedding dimension of 4096, weight tying saves ~205M parameters.
   * **Regularization Effect**: Enforces consistency between input representations and output predictions.
   * **Faster Convergence**: May accelerate training by creating a more direct learning signal.

   **Current Implementation in Modern LLMs:**

   .. list-table::
      :header-rows: 1
      :widths: 30 20 50

      * - Model
        - Weight Tying
        - Implementation Details
      * - **GPT-4**
        - Likely used
        - Not officially confirmed, but consistent with earlier GPT models
      * - **Claude 3.5/3.7**
        - Unknown
        - Not disclosed in public documentation
      * - **LLaMA 3.2**
        - Selectively implemented
        - Used in the 3B parameter model, status varies in larger variants
      * - **DeepSeek-V3**
        - Not used by default
        - Configurable via ``tie_word_embeddings=False`` default setting

   **Implementation Considerations:**

   The decision to implement weight tying appears to be influenced by model size and architecture. Smaller models (like LLaMA 3.2 3B) tend to benefit more from the parameter efficiency, while larger models forgo weight tying to maximize performance as the benefit diminishes in comparison to the much larger overall model size due to increased number of decoder layers.

   **Trade-offs:**

   While weight tying enhances efficiency, it introduces a constraint that may limit representational flexibility. The optimal decision depends on whether priority is given to parameter efficiency or maximum model expressiveness.

Generation Strategies
~~~~~~~~~~~~~~~~~~~~~

After a GPT model produces logits from its forward pass, these logits must be converted into actual tokens through a :newconcept:`decoding strategy`. The choice of decoding strategy significantly impacts the quality, diversity, and coherence of the generated text. GPT models support several approaches, ranging from simple deterministic methods to more sophisticated algorithms that :ub:`balance quality and diversity`.

In contemporary large language models (LLMs) such as GPT-4, Claude, DeepSeek, and LLaMA, :ub:`sampling-based decoding strategies have become the preferred method for text generation` (see `Sampling-Based Generation`_). These strategies introduce controlled randomness, enabling the production of diverse and creative outputs, which is particularly beneficial for open-ended tasks like conversational AI and creative writing.

* Traditionally, `beam search`_ has been employed to maintain multiple candidate sequences, selecting the most probable one. While effective in tasks requiring high accuracy and coherence, such as machine translation, beam search can lead to repetitive and less diverse outputs, making it less suitable for open-ended text generation.
* However, today :ub:`beam search still remains a valuable decoding strategy in specific scenarios where generating the most probable and coherent sequence is crucial`. Key applications include: Machine Translation, Speech Recognition, Text Summarization, Reasoning and Planning Tasks (e.g. for AI Agents).

.. code-block:: python
   :class: folding
   :name: unified_generate_with_strategy

    def generate_with_strategy(
            self,
            input_ids: torch.LongTensor,
            max_new_tokens: int,
            strategy: str = "sample",
            **kwargs
    ) -> torch.LongTensor:
        """
        Unified generation method supporting multiple decoding strategies.

        Args:
            input_ids: Input token ids
            max_new_tokens: Maximum number of new tokens to generate
            strategy: One of "greedy", "sample", or "beam"
            **kwargs: Additional arguments for specific strategies
        """
        if strategy == "greedy":
            return self.greedy_search(
                input_ids,
                max_new_tokens,
                pad_token_id=kwargs.get("pad_token_id"),
                eos_token_id=kwargs.get("eos_token_id")
            )
        elif strategy == "sample":
            return self.sample(
                input_ids,
                max_new_tokens,
                temperature=kwargs.get("temperature", 1.0),
                top_k=kwargs.get("top_k"),
                top_p=kwargs.get("top_p"),
                pad_token_id=kwargs.get("pad_token_id"),
                eos_token_id=kwargs.get("eos_token_id")
            )
        elif strategy == "beam":
            return self.beam_search(
                input_ids,
                max_new_tokens,
                num_beams=kwargs.get("num_beams", 5),
                pad_token_id=kwargs.get("pad_token_id"),
                eos_token_id=kwargs.get("eos_token_id"),
                length_penalty=kwargs.get("length_penalty", 1.0)
            )
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")

Greedy Search
^^^^^^^^^^^^^

**Greedy Search** represents the simplest decoding approach for autoregressive language models. At each generation step, it selects the single token with the highest predicted probability. This approach can be characterized as:

* **Locally Optimal Yet Myopic**: [Pro & Con] Makes the best choice at each individual step, but it doesn't consider how current choices affect future options.
* **Deterministic Yet Lacking Creativity**: [Mostly Con] Given the same input and model, always produces identical output. No exploration of alternatives, even when multiple options have similar probabilities.
* **Memory-Efficient**: [Pro] Only needs to track a single candidate sequence.

Before the transformer era, greedy decoding was often sufficient for simpler sequence-to-sequence tasks. As language models grew in capability and application domains expanded, its limitations became more apparent, driving the development of more sophisticated alternatives like sampling and beam search. Today its application is limited to

* Tasks requiring exact reproducibility and a single fixed answer is sufficient and diversity isn't important.
* Resource-constrained environments.

.. code-block:: python
   :class: folding
   :name: gpt_greedy_search

    class GPTDecoder(nn.Module):
        # ... existing methods ...

        def process_logits_for_finished_sequences(
                self,
                logits: torch.Tensor,
                finished_sequences_flags: torch.Tensor,
                pad_token_id: Optional[int] = None
        ) -> torch.Tensor:
            """
            Process logits to ensure finished sequences only generate padding tokens.

            Args:
                logits: Tensor of logits with shape (batch_size, vocab_size)
                finished_sequences_flags: Boolean tensor of shape (batch_size) tracking which
                                   sequences have already generated an EOS token
                pad_token_id: Token ID to use for padding (if None, no special handling occurs)

            Returns:
                Modified logits where finished sequences can only select padding token
            """
            # If no sequences are finished or no pad token provided, return logits unchanged
            if pad_token_id is None or not finished_sequences_flags.any():
                return logits

            # torch.full_like` creates a new tensor with the same size and data type as the input tensor, but filled with a specified value
            # Below sets all token probabilities to -infinity (impossible to select)
            # Shape: [batch_size, vocab_size] - same as logits
            pad_mask = torch.full_like(logits, float('-inf'))
            # Set only the padding token's logit to 0 (making it the only possible choice)
            pad_mask[:, pad_token_id] = 0

            # Apply this padding mask only to sequences that are marked as finished
            # For unfinished sequences, keep original logits
            return torch.where(
                        finished_sequences_flags.unsqueeze(1),  # Need to add a dimension for proper broadcasting with `logits`: [batch_size, 1]
                        pad_mask,                         # Use pad_mask for finished sequences
                        logits                            # Use original logits for unfinished sequences
                    )

        def update_finished_sequence_flags(
                self,
                finished_sequences_flags: torch.Tensor,
                next_token: torch.Tensor,
                eos_token_id: Optional[int] = None
        ) -> torch.Tensor:
            """
            Update tracking for which sequences have finished generating.

            Args:
                finished_sequences_flags: Boolean tensor of shape (batch_size) tracking which
                                   sequences have already generated an EOS token
                next_token: Tensor of shape (batch_size, 1) or (batch_size) containing
                           the next token generated for each sequence
                eos_token_id: Token ID that indicates end of sequence (if None, no sequences are marked as finished)

            Returns:
                Updated finished_sequences_flags tensor
            """
            # If no EOS token specified, no sequences finish
            if eos_token_id is None:
                return finished_sequences_flags

            # Ensure next_token is a 1D tensor by removing the last dimension if present
            if next_token.dim() > 1:
                next_token = next_token.squeeze(-1)

            # Identify which sequences generated an EOS token in this step
            just_finished = (next_token == eos_token_id)

            # Update existing finished flags with newly finished sequences
            return finished_sequences_flags | just_finished

        def crop_input_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
            """
            Crop input IDs to respect the model's maximum sequence length.

            If input sequences are longer than the model's maximum sequence length,
            only the most recent tokens up to max_seq_length are kept.

            Args:
                input_ids: Tensor of shape (batch_size, seq_length) or
                          (batch_size * num_beams, seq_length) for beam search

            Returns:
                Tensor with the same batch dimension but sequence length
                limited to at most self.max_seq_length
            """
            return input_ids if input_ids.size(1) <= self.max_seq_length else input_ids[:, -self.max_seq_length:]

        @torch.no_grad()
        def greedy_search(
                self,
                input_ids: torch.LongTensor,
                max_new_tokens: int,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None
        ) -> torch.LongTensor:
            """
            Simplest decoding method: always choose the most likely next token.

            Handles sequences that finish at different times by tracking which sequences
            have generated an EOS token and padding future tokens for those sequences.
            """

            # Get the batch size, i.e., the number of sequences
            batch_size = input_ids.shape[0]

            # ----------------
            # 1. Creates an array to track completed sequences
            # ----------------
            # One-dimensional array (length is `batch_size`) to track which sequences are finished (have generated EOS)
            #   `GPTDecoder`'s forward function is not aware of the sequence completion and will continue to generate the next token
            #   The sequence completion is tracked here in the generation function
            finished_sequence_flags = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

            # For example, suppose the vocabulary has 5 tokens with the first token as the padding token, and the second token as the EOS token
            # logits = [
            #     [2.5, 1.0, 3.2, 0.8, 1.5],  # Sequence 1 (finished)
            #     [0.7, 4.1, 2.3, 3.0, 1.2]   # Sequence 2 (still generating)
            # ]
            #
            # finished_sequence_flags = [True, False]

            # The generation loop
            for _ in range(max_new_tokens):
                # Crop sequence if needed
                input_ids_cond = self.crop_input_ids(input_ids)

                # ----------------
                # 2. Get raw logits from the model
                # ----------------
                # Get model predictions - with inference=True, this directly returns logits for the last token
                # Shape: (batch_size, vocab_size)
                logits = self(input_ids_cond, inference=True)

                # For finished sequences, replace logits with padding-token-only logits
                # This ensures only padding tokens will be generated for these sequences
                # For the above example, we would have
                # pad_mask = [
                #    [0.0, -inf, -inf, -inf, -inf],  # Only token 0 (padding) can be selected, and same for all sequences
                #    [0.0, -inf, -inf, -inf, -inf]
                # ]

                # ----------------
                # 3. Process logits for finished sequences
                # ----------------
                # Handle finished sequences by forcing them to generate padding
                # For the above example, we would have
                # logits = [
                #     [0.0, -inf, -inf, -inf, -inf],  # Sequence 1 (can only generate padding)
                #     [0.7, 4.1, 2.3, 3.0, 1.2]       # Sequence 2 (unchanged)
                # ]
                logits = self.process_logits_for_finished_sequences(
                    logits,
                    finished_sequence_flags,
                    pad_token_id
                )

                # ----------------
                # 4. Get next token and append them to input_ids
                # ----------------
                # Select the highest probability token for each sequence in batch
                # Shape: [batch_size, 1]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # ----------------
                # 5. Update finished_sequence_flags and break the loop if all sequences have completed
                # ----------------
                # Append the selected tokens to our running sequences for auto-regressive generation
                # Shape after concatenation: [batch_size, sequence_length + 1]
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Check for and handle sequences that just generated an EOS token
                finished_sequence_flags = self.update_finished_sequence_flags(
                    finished_sequence_flags,
                    next_token,
                    eos_token_id
                )

                # Break the loop if all sequences have completed
                if finished_sequence_flags.all():
                    break

            return input_ids


Sampling-Based Generation
^^^^^^^^^^^^^^^^^^^^^^^^^

**Sampling-Based Generation** introduces controlled randomness into text generation by treating the model's output logits as a probability distribution. Instead of always selecting the token with the highest probability (as in greedy search), sampling draws tokens from this distribution based on their relative likelihoods. This produces more diverse and creative outputs while maintaining coherence.

.. code-block:: python
   :class: folding
   :name: gpt_sampling_generation

    class GPTDecoder(nn.Module):
        # ... existing methods ...

        def apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
            """
            Apply temperature scaling to logits.

            Args:
                logits: Raw logits from model output, shape (batch_size, vocab_size)
                temperature: Scaling factor to control randomness
                             Lower values make distribution more peaked (deterministic)
                             Higher values make distribution more uniform (random)

            Returns:
                Temperature-scaled logits with shape (batch_size, vocab_size)
            """
            # Avoid division by zero or negative temperature
            if temperature <= 0:
                raise ValueError("Temperature must be positive")

            # Apply temperature scaling - shape remains (batch_size, vocab_size)
            return logits / temperature

        def apply_top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
            """
            Apply top-k filtering to restrict sampling to top k tokens.

            Args:
                logits: Logits after temperature scaling, shape (batch_size, vocab_size)
                top_k: Number of highest probability tokens to consider

            Returns:
                Filtered logits with only top-k tokens having finite values
                Shape (batch_size, vocab_size)
            """
            if top_k <= 0:
                return logits  # No filtering, shape (batch_size, vocab_size)

            # Find values and indices of the k largest elements (per batch)
            # top_k_values shape: (batch_size, min(top_k, vocab_size))
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))

            # Get the threshold value for each item in the batch
            # This selects the smallest value among the top-k values in each batch
            # min_values shape: (batch_size, 1)
            min_values = top_k_values[:, -1].unsqueeze(1)

            # Create a mask for values to filter out (below the threshold)
            # filter_mask shape: (batch_size, vocab_size)
            # True where logits < min_values (these will be masked out)
            filter_mask = logits < min_values

            # Apply the mask by setting filtered values to -infinity
            # Return shape: (batch_size, vocab_size)
            return logits.masked_fill(filter_mask, float('-inf'))

        def apply_top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
            """
            Apply nucleus (top-p) sampling to dynamically filter tokens based on
            cumulative probability mass.

            Args:
                logits: Logits after temperature scaling (and possibly top-k filtering)
                       Shape (batch_size, vocab_size)
                top_p: Probability threshold (0.0 to 1.0)
                       Only tokens comprising cumulative probability mass <= top_p
                       will be kept for sampling

            Returns:
                Filtered logits with only tokens in the nucleus having finite values
                Shape (batch_size, vocab_size)
            """
            if top_p <= 0.0 or top_p >= 1.0:
                return logits  # No filtering for invalid values, shape (batch_size, vocab_size)

            # Sort logits in descending order
            # sorted_logits shape: (batch_size, vocab_size)
            # sorted_indices shape: (batch_size, vocab_size)
            # For example,
            # Let's say these were the original token IDs before sorting:
            # Token IDs:        [ A,    B,    C,    D,    E,    F ]
            # Original logits:  [1.4,  2.7,  3.0,  1.9,  1.0,  0.0]
            #                     D     B     A     C     E     F
            #
            # After sorting by descending logit values:
            # Sorted logits:     [3.0,  2.7,  1.9,  1.4,  1.0,  0.0]
            # Sorted Token IDs:  [ C,    B,    D,    A,    E,    F ]
            # sorted_indices:    [ 2,    1,    3,    0,    4,    5 ]  (positions in original array)
            #
            # The sorted_indices tell us where each position in our sorted arrays
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)

            # Calculate probabilities from sorted logits
            # sorted_probs shape: (batch_size, vocab_size)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)

            # Calculate cumulative probabilities
            # cumulative_probs shape: (batch_size, vocab_size)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create nucleus mask where cumulative probability exceeds threshold
            # Each row of the mask identifies tokens outside the nucleus
            # nucleus_mask shape: (batch_size, vocab_size)
            # True where tokens are outside the nucleus
            nucleus_mask = cumulative_probs > top_p

            # Always keep at least one token (the highest probability one)
            # First column of nucleus_mask is set to False
            nucleus_mask[:, 0] = False

            # Shift mask by one position to exclude the token that pushed us over top_p
            # This ensures we keep all tokens up to and including the one that crosses threshold
            # Shape remains (batch_size, vocab_size)
            #
            # Continuing above example,
            # we have these sorted token probabilities for top_p = 0.7:
            # Token Index:     0     1     2     3     4     5
            # Probability:    0.4   0.3   0.15  0.08  0.05  0.02
            # Cumulative:     0.4   0.7   0.85  0.93  0.98  1.00
            #                       ^
            #                       |
            #             Crosses 0.7 threshold here
            #
            # Initial `nucleus_mask` (where cumulative > 0.7):
            # [False, False, True,  True,  True,  True]
            #   |      |
            #   Keep   Keep
            # After shifting:
            # [False, False, False, True,  True,  True]
            #   |      |      |
            #   Keep   Keep   Keep
            #
            # This keeps tokens 0, 1, and 2 (the threshold-crossing token),
            # with total probability mass of 0.85 (> 0.7)
            nucleus_mask[:, 1:] = nucleus_mask[:, :-1].clone()

            # Convert the sorted mask back to the original token ordering
            # `to_remove` shape: (batch_size, vocab_size)
            to_remove = torch.zeros_like(logits, dtype=torch.bool)

            # Continuing our example:
            # `nucleus_mask`: [False, False, False, True, True, True]
            # `to_remove` will be initialized as: [F, F, F, F, F, F]
            #
            # `scatter_` puts values from nucleus_mask into to_remove at positions specified by sorted_indices
            # After `scatter_`:
            # to_remove:         [True, False, False, False, True, True]
            # Token IDs:         [  A,    B,     C,     D,     E,    F ]
            # Original logits:   [ 1.4,  2.7,   3.0,   1.9,   1.0,  0.0]
            #
            # This correctly maps our decision to keep the top 3 tokens by probability (i.e. C, B, D)
            #   back to their original positions
            to_remove = to_remove.scatter_(1, sorted_indices, nucleus_mask)

            # Apply the mask by setting filtered tokens to -infinity
            # Return shape: (batch_size, vocab_size)
            #
            # Continuing our example:
            # Original logits:   [1.4,   2.7,   3.0,   1.9,   1.0,  0.0]
            # to_remove:         [True, False, False, False, True, True]
            #
            # After applying mask:
            # Filtered logits:   [-inf, 2.7,  3.0,  1.9, -inf, -inf]
            #
            # When these logits are converted to probabilities via softmax:
            # Final probs:       [0.0,  0.35, 0.47, 0.18, 0.0,  0.0]
            #
            # Now only tokens B, C, and D have non-zero probabilities,
            # which are properly normalized to sum to 1.0.
            # These three tokens correspond exactly to the top 0.85 probability mass
            # from our sorted distribution, as we intended with top_p = 0.7.
            return logits.masked_fill(to_remove, float('-inf'))

        @torch.no_grad()
        def sample(
                self,
                input_ids: torch.LongTensor,
                max_new_tokens: int,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None
        ) -> torch.LongTensor:
            """
            Generate new tokens autoregressively using sampling strategies.
            Combines temperature scaling with optional top-k and top-p filtering.

            Args:
                input_ids: Starting token IDs of shape (batch_size, seq_length)
                max_new_tokens: Maximum number of tokens to generate
                temperature: Controls randomness (lower = more deterministic)
                top_k: If set, only sample from the top k most likely tokens
                top_p: If set, only sample from the smallest set of tokens whose
                       cumulative probability exceeds p
                pad_token_id: Token ID used for padding
                eos_token_id: Token ID that signals sequence completion

            Returns:
                Tensor of shape (batch_size, seq_length + generated_length)
            """
            # ----------------
            # 1. Use greedy search for temperature=0
            # ----------------
            if temperature == 0:
                return self.greedy_search(
                    input_ids,
                    max_new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id
                )

            # Get the batch size from the input
            # input_ids shape: (batch_size, seq_length)
            batch_size = input_ids.shape[0]

            # ----------------
            # 2. Create tracking for completed sequences
            # ----------------
            # finished_sequence_flags shape: (batch_size)
            # Initialized with zeros (False) - no sequences are finished yet
            finished_sequence_flags = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

            # The generation loop - will run for max_new_tokens iterations unless all sequences finish early
            for _ in range(max_new_tokens):
                # ----------------
                # 3. Prepare inputs for current iteration
                # ----------------
                # Crop sequence if needed to respect model's maximum sequence length
                # input_ids_cond shape: (batch_size, seq_length')
                # where seq_length' = min(seq_length, max_seq_length)
                input_ids_cond = self.crop_input_ids(input_ids)

                # ----------------
                # 4. Get raw logits from the model
                # ----------------
                # logits shape: (batch_size, vocab_size)
                logits = self(input_ids_cond, inference=True)

                # ----------------
                # 5. Process logits for finished sequences
                # ----------------
                # Ensure sequences that have already generated EOS only produce padding tokens
                # logits shape remains (batch_size, vocab_size)
                logits = self.process_logits_for_finished_sequences(
                    logits,
                    finished_sequence_flags,
                    pad_token_id
                )

                # ----------------
                # 6. Apply sampling strategies to filter logits
                # ----------------
                # First adjust with temperature
                if temperature != 1.0:
                    # logits shape remains (batch_size, vocab_size)
                    logits = self.apply_temperature(logits, temperature)

                # Then apply top-k filtering (if specified)
                if top_k is not None and top_k > 0:
                    # logits shape remains (batch_size, vocab_size)
                    logits = self.apply_top_k_filtering(logits, top_k)

                # Finally apply top-p filtering (if specified)
                if top_p is not None and 0.0 < top_p < 1.0:
                    # logits shape remains (batch_size, vocab_size)
                    logits = self.apply_top_p_filtering(logits, top_p)

                # ----------------
                # 7. Sample from the filtered distribution
                # ----------------
                # Convert logits to probabilities
                # probs shape: (batch_size, vocab_size)
                probs = torch.softmax(logits, dim=-1)

                # Sample from the probability distribution (one token per sequence)
                # next_token shape: (batch_size, 1)
                next_token = torch.multinomial(probs, num_samples=1)

                # ----------------
                # 8. Append sampled tokens and update tracking
                # ----------------
                # Add new tokens to sequences
                # Updated input_ids shape: (batch_size, seq_length + 1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Update tracking of which sequences have finished
                # finished_sequence_flags shape remains (batch_size)
                finished_sequence_flags = self.update_finished_sequence_flags(
                    finished_sequence_flags,
                    next_token,
                    eos_token_id
                )

                # Break if all sequences have finished
                if finished_sequence_flags.all():
                    break

            # Return the complete sequences with generated tokens
            # Final shape: (batch_size, original_seq_length + generated_length)
            return input_ids

The above sampling approach introduces several key hyperparameters that control the generation behavior:

* :newconcept:`Temperature` controls how concentrated or spread out the probability distribution becomes before sampling:

  .. math::

        p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}

  where :math:`z_i` are the logits and :math:`T` is the temperature parameter.

  This seemingly simple operation has profound effects on the distribution's entropy due to the :ub:`mathematical interplay between the linear scaling on logits and the exponential scaling by the softmax operation`:

  * **Temperature = 1.0**: Uses the model's raw probabilities without adjustment
  * **Temperature < 1.0**: Sharpens the distribution, making high-probability tokens more likely

    * **Low values (0.3-0.7)**: Produce more focused, deterministic text.
    * :ub:`As T approaches 0, sampling approaches greedy search behavior`.

  * **Temperature > 1.0**: Flattens the distribution, giving more weight to low-probability tokens.

    * **High values (1.2-2.0)**: Generate more random, diverse, and unexpected text.
    * :ub:`Very high values Can produce incoherent or ungrammatical output`.

  On the high level, :ub:`Temperature is essentially a creativity-coherence trade-off parameter` - lower values prioritize accuracy and coherence, while higher values enable more creative exploration of the distribution.

  .. admonition:: Mathematical Insights on Temperature
     :class: note

     The power of temperature sampling comes from several key mathematical properties:

     **1. Exponential Amplification/Suppression**

     The exponential function magnifies small changes in input into large changes in output:

     * When **T < 1**: Differences between logits get amplified exponentially
     * When **T > 1**: Differences between logits get compressed exponentially

     For example, with logits [5.0, 3.0, 1.0]:

     .. list-table::
        :header-rows: 1
        :widths: 15 25 25 25

        * - Temperature
          - T = 0.5 (Low)
          - T = 1.0 (Normal)
          - T = 2.0 (High)
        * - Scaled Logits
          - [10.0, 6.0, 2.0]
          - [5.0, 3.0, 1.0]
          - [2.5, 1.5, 0.5]
        * - Probabilities
          - [0.982, 0.017, 0.001]
          - [0.867, 0.119, 0.014]
          - [0.649, 0.249, 0.102]

     **2. Entropy Control**

     Temperature $T$ directly controls the entropy $H(P)$ of the resulting distribution :math:`P=\{p_0, p_1, ..., p_{|V|}\}` (:math:`|V|` is the vocabulary size):

     .. math::

        H(P) = -\sum_i p_i \log p_i

     As temperature approaches extreme values, the entropy behavior can be formalized:

     .. math::

        \lim_{T \to 0} H(P) = 0 \quad \text{(deterministic)}

        \lim_{T \to \infty} H(P) = \log(|V|) \quad \text{(uniform distribution)}

     **3. Scale Invariance**

     Temperature scaling preserves the relative ranking of tokens while changing their probability gaps. The most likely token remains the most likely regardless of temperature, but the probability gap between tokens changes dramatically.

     **4. Information Theory Connection**

     From an information theory perspective, temperature controls the "surprise" factor in sampling. Lower temperatures favor high-probability, low-surprise tokens (minimizing information content per token), while higher temperatures allow more surprising, information-rich tokens to appear.

     **5. Statistical Physics Origin**

     The temperature parameter gets its name from statistical physics, where the Boltzmann distribution takes a similar form:

     .. math::

        p(E) \propto e^{-E/kT}

     where E is energy, k is Boltzmann's constant, and T is temperature. In physics, higher temperatures lead to more random states with higher entropy - exactly the behavior we see in text generation.

     The mathematical elegance of temperature scaling lies in how a single parameter can smoothly interpolate between deterministic behavior and completely random selection, offering precise control over the exploration-exploitation trade-off in language generation.

* :newconcept:`Top-K Sampling` and :newconcept:`Top-P Sampling` (a.k.a. :newconcept:`Nucleus Sampling`) :ub:`prevent sampling from the long tail` of low-probability tokens by truncating the distribution:

  * **Top-K Sampling** identifies the $K$ tokens with the highest next-token logits, and mask out the remaining.

    * **K = 1**: Equivalent to greedy search.
    * **Small K (5-20)**: More focused, predictable output.
    * **Large K (40-100)**: More diverse, creative output.

  * **Top-P Sampling** sorts tokens by probability in descending order, and selects the smallest set of tokens whose cumulative probability exceeds threshold $P$.

    * It addresses a key limitation of Top-K: when the model's confidence is dispersed across many tokens, Top-K might truncate too aggressively, while when the model is very confident, Top-K might include unnecessary options.
    * **Common P values range from 0.9 to 0.95**: Include only the tokens comprising 90-95% of probability mass.
    * **Lower P values (0.5-0.7)**: More conservative generation.

.. admonition:: Practical Setup of Temperature, Top-K, and Top-P Sampling
   :class: note

    In practice, Temperature, Top-K, and Top-P Sampling are often combined (:ub:`Top-K is more optional, but at least Temperature and Top-P Sampling are combined`) to control sampling-based generation behavior. Different LLM frameworks have distinct default configurations:

    * **OpenAI GPT-4.5**:

      * Temperature: 1.0 (balanced creativity)
      * Top-K: Not directly exposed in the API
      * Top-P: 1.0 (considers the entire distribution)

    * **Claude 3.7**:

      * Temperature: 1.0 via Anthropic API, 0.5-1.0 on platforms like Amazon Bedrock
      * Top-K: Not explicitly defined in core API; 250 or disabled on some platforms
      * Top-P: Not explicitly defined in core API; 0.999-1.0 on some platforms

    * **DeepSeek models**:

      * Temperature: 1.0 (API default), but with task-specific recommendations:

        * Coding/Math: 0.0-0.3 (deterministic)
        * Data Analysis: 0.5-0.7 (balanced)
        * Conversation/Translation: 0.7-1.3 (more creative)
        * Creative Writing: 1.3-1.5 (highly creative)

      * Top-K: 0 (disabled by default in API)
      * Top-P: 1.0 (no truncation by default)
      * Note: DeepSeek V3 employs an internal mapping where model temperature = API temperature × 0.3

    * **Llama 3.2**:

      * Temperature: Varies by implementation (0.6-1.0), with 0.8 common for llama.cpp
      * Top-K: Varies (0-50), with 40 common for llama.cpp
      * Top-P: Varies (0.95-1.0), with 0.9 common for llama.cpp

    **Practical tuning recommendations by task**:

    1. **Factual/Technical** (code, data analysis, math):

       * Temperature: 0.0-0.5
       * Top-P: 0.9-0.95
       * Top-K: 20-50 if available

    2. **Balanced/Conversational**:

       * Temperature: 0.6-0.8
       * Top-P: 0.9-1.0
       * Top-K: 40-50 if available

    3. **Creative** (stories, poetry, brainstorming):

       * Temperature: 0.9-1.5 (where supported)
       * Top-P: 0.95-1.0
       * Top-K: 0 (disabled) or 50-100

    When tuning parameters, adjust one at a time to understand its impact, and always consult the specific platform's documentation for accurate defaults and supported ranges.


Beam Search
^^^^^^^^^^^

:newconcept:`Beam Search` represents the most sophisticated decoding approach among the strategies discussed. Unlike greedy search and sampling, which maintain a single sequence, beam search :ub:`tracks multiple sequences (beams) simultaneously` to explore different possibilities in parallel rather than following a single generation trajectory. This fundamental difference makes beam search a unique approach with its own specific workflow:

1. **Initialization**: Start with the input sequence, duplicated for each beam
2. **Expansion**: For each beam:

   * Get next-token probabilities from the model
   * Calculate the cumulative score for each possible next token by adding the beam's current score (in log space)
   * This effectively maximizes the product of probabilities in the sequence

3. **Selection**: From all potential beam × vocabulary continuations:

   * Select the top N (where N = number of beams) highest-scoring candidates
   * These become the new beams for the next iteration

4. **EOS Handling**: Special processing for sequences that produce end-of-sequence tokens:

   * Completed sequences (those ending with EOS) are stored separately. Final output selection prioritizes completed sequences.
   * When sufficient completed sequences are collected, that batch item is marked as finished

5. **Termination**: Continue until:

   * Maximum length is reached, or
   * All sequences in the batch have generated an end-of-sequence token

6. **Output Selection**: Return the single highest-scoring completed sequence for each batch item

The following interactive visualization below demonstrates beam search in action with `Length Penalty`_. A complete implementation is also provided with common techniques like :refconcept:`Search Expansion Factor` to mitigate the issue of early EOS diminishing active beams, and configurable `Length Penalty`_ to balance between shorter and longer sequence generation.

.. react-component:: ../../_static/images/modeling/classic_modeling/transformer_models/BeamSearchVisualization.tsx
    :width: auto
    :max-width: 1300px
    :center:
    :katex:

.. code-block:: python
   :class: folding
   :name: gpt_beam_search

    class GPTDecoder(nn.Module):
        # ... existing methods ...

        def normalize_scores_by_length(
            self,
            scores: torch.Tensor,
            sequence_length: int,
            alpha: float
        ) -> torch.Tensor:
            """
            Normalize scores by the standard length penalty used in sequence generation.

            Uses the formula: scores / (((5 + length)^alpha) / ((5 + 1)^alpha))

            Args:
                scores: Tensor containing the scores to normalize
                sequence_length: Length of the sequence
                alpha: Length penalty strength parameter (commonly between 0.6 and 1.0)
                       - alpha = 0: No penalty (favors shorter sequences)
                       - alpha > 0: Encourages longer sequences

            Returns:
                Tensor of same shape as scores containing the length-normalized scores
            """
            if alpha == 0.0:
                # No length penalty
                return scores.clone()
            else:
                # Standard length penalty formula from GNMT paper
                length_penalty = ((5 + sequence_length) ** alpha) / ((5 + 1) ** alpha)
                return scores / length_penalty

        @torch.no_grad()
        def beam_search(
                self,
                input_ids: torch.LongTensor,
                max_new_tokens: int,
                num_beams: int = 5,
                beam_expansion: float = 2.0,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None,
                length_penalty: float = 1.0
        ) -> torch.LongTensor:
            """
            Beam search decoding: maintains multiple candidate sequences (beams) and expands
            the most promising ones at each step, eventually returning the highest-scoring complete sequence.

            Args:
                input_ids: Tensor of shape (batch_size, seq_length), initial input token IDs.
                max_new_tokens: Maximum number of new tokens to generate.
                num_beams: Number of candidate beams to maintain at each generation step.
                beam_expansion: Factor (>1.0) that temporarily expands the candidate beam pool to
                                `beam_expansion * num_beams`, increasing exploration and
                                likelihood of higher-quality sequences.
                pad_token_id: Token ID used for padding finished sequences.
                eos_token_id: Token ID signaling the end of generation.
                length_penalty: Controls the bias towards shorter or longer sequences
                                (>1.0 penalizes long sequences, <1.0 rewards long sequences).
                                In beam search, each candidate sequence’s score is usually the cumulative (log-) probability
                                of all tokens generated so far. Without adjustments, shorter sequences might naturally have
                                higher scores (closer to zero, since log probabilities are negative),
                                making the model biased toward shorter sequences.
                                To balance this bias, a length penalty is introduced, making beam search more flexible
                                in promoting shorter or longer sequences based on the value you set:

            Returns:
                Tensor of shape (batch_size, seq_length + generated_length) containing
                the highest-scoring sequence for each item in the batch
            """
            # ----------------
            # 1. Setup initial state
            # ----------------
            batch_size = input_ids.shape[0]
            device = input_ids.device
            vocab_size = self.vocab_size

            # Track which sequences have finished generating - using same tensor format as other strategies.
            # finished_sequence_flags shape: (batch_size)
            finished_sequence_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # Create storage for completed sequences and their scores
            # We'll store the best completed sequence for each batch item
            completed_sequences = [[] for _ in range(batch_size)]
            completed_scores = [[] for _ in range(batch_size)]

            # Initialize beam scores to keep track of active beams: first beam active (score=0), others inactive (score=-inf)
            beam_scores = torch.zeros((batch_size, num_beams), device=device)
            beam_scores[:, 1:] = float('-inf')

            # Now flatten to shape (batch_size * num_beams).
            # The `.view(-1)` operation is used to reshape the 2D tensor into a 1D tensor (a flat array).
            #   Say `batch_size=2` and `num_beams=3` and we first create a tensor of shape (2, 3):
            #   [[0, -inf, -inf],   # Batch item 0, beams 0,1,2
            #    [0, -inf, -inf]]   # Batch item 1, beams 0,1,2
            #   After `.view(-1)`, it horizontally concatenates the rows becomes [0, -inf, -inf, 0, -inf, -inf] of shape (6,),
            #   where indices 0,1,2 correspond to beams 0,1,2 for batch item 0,
            #   and indices 3,4,5 correspond to beams 0,1,2 for batch item 1.
            # The reason for this operation is that:
            #   1. Beam search treats as if there were `batch_size * num_beams` batches, so the forward function can work without change.
            #   2. Later operations like `next_scores = log_probs + beam_scores.unsqueeze(1)` require such flattened array for operational convenience.
            # beam_scores shape: (batch_size * num_beams)
            beam_scores = beam_scores.view(-1)

            # Expand input_ids to create initial beams (repeat each input num_beams times).
            # Original: [batch_0, batch_1, ...] → Expanded: [batch_0, batch_0, ..., batch_1, batch_1, ...]
            # From shape (batch_size, seq_length) to (batch_size * num_beams, seq_length)
            input_ids = input_ids.repeat_interleave(num_beams, dim=0)

            # ----------------
            # 2. Generation loop
            # ----------------
            # Continue until all sequences finish or max length is reached.
            max_length = input_ids.shape[1] + max_new_tokens
            while input_ids.shape[1] < max_length and not finished_sequence_flags.all():
                # Crop input to respect maximum context length if needed.
                input_ids_cond = self.crop_input_ids(input_ids)

                # ----------------
                # 3. Get predictions from model
                # ----------------
                # Forward pass through model to get next-token logits
                # logits shape: (batch_size * num_beams, vocab_size)
                logits = self(input_ids_cond, inference=True)

                # Ensure any batch items that have finished generate only padding tokens.
                logits = self.process_logits_for_finished_sequences(
                    logits,
                    finished_sequence_flags.repeat_interleave(num_beams),
                    pad_token_id
                )

                # Convert to log probabilities for numerical stability in beam calculations.
                # log_probs shape: (batch_size * num_beams, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1)

                # ----------------
                # 4. Calculate scores for all possible next tokens
                # ----------------
                # Add current beam scores to next-token scores over the vocabulary,
                #   i.e. adding the beam score to every token score in `log_probs`
                #   so that `next_scores` represents scores over all possible next beams involving the "current beams",
                #   plus all possible next tokens from the vocabulary.
                # Also note that initially `beam_scores` only have the first beam activated.
                #   See above `beam_scores[:, 1:] = float('-inf')`.
                #   After the initial `beam_scores` adding `log_probs`, the `next_scores` is of shape (batch_size * num_beams, vocab_size),
                #     but only `batch_size` of the resulting `next_scores` have valid scores, because only one beam is valid per batch item at this time.
                #   However, this issue will resolve for the next loop, as we expect all beams become valid after the initial loop.
                # next_scores shape: (batch_size * num_beams, vocab_size)
                #
                # For example,
                # log_probs (after forwarding through the model):
                # Shape: (batch_size * num_beams, vocab_size) = (4, 3)
                #
                # [
                #  [-0.2, -1.5, -3.0],  # Batch item 0, Beam 0
                #  [-0.7, -2.1, -0.4],  # Batch item 0, Beam 1
                #  [-1.0, -0.3, -2.5],  # Batch item 1, Beam 0
                #  [-0.5, -1.8, -1.2]   # Batch item 1, Beam 1
                # ]
                #
                # beam_scores: cumulative scores for each beam
                # Shape: (batch_size * num_beams) = (4,)
                # [-0.5, -1.2, -0.8, -1.5]
                #
                # Then after adding log_probs by `next_scores = log_probs + beam_scores.unsqueeze(1)`, we have
                # Shape: (batch_size * num_beams, vocab_size) = (4, 3)
                #
                # [
                #   [-0.5 + (-0.2), -0.5 + (-1.5), -0.5 + (-3.0)],  # Batch item 0, Beam 0 + Next Token
                #   [-1.2 + (-0.7), -1.2 + (-2.1), -1.2 + (-0.4)],  # Batch item 0, Beam 1 + Next Token
                #   [-0.8 + (-1.0), -0.8 + (-0.3), -0.8 + (-2.5)],  # Batch item 1, Beam 0 + Next Token
                #   [-1.5 + (-0.5), -1.5 + (-1.8), -1.5 + (-1.2)]   # Batch item 1, Beam 1 + Next Token
                # ] = [
                #   [-0.7, -2.0, -3.5],  # Batch item 0, Beam 0 + Next Token
                #   [-1.9, -3.3, -1.6],  # Batch item 0, Beam 1 + Next Token
                #   [-1.8, -1.1, -3.3],  # Batch item 1, Beam 0 + Next Token
                #   [-2.0, -3.3, -2.7]   # Batch item 1, Beam 1 + Next Token
                # ]
                next_scores = log_probs + beam_scores.unsqueeze(1)

                # Reshape for beam search: horizontally combine beam score.
                #
                # Continuing above example, after `next_scores = next_scores.view(batch_size, num_beams * vocab_size)`, we have
                # Shape: (batch_size, num_beams * vocab_size) = (2, 6)
                # [
                #   [-0.7, -2.0, -3.5, -1.9, -3.3, -1.6],  # Batch item 0, all (beam + next token) scores concat horizontally
                #   [-1.8, -1.1, -3.3, -2.0, -3.3, -2.7]   # Batch item 1, all (beam + next token) scores concat horizontally
                # ]
                # next_scores shape: (batch_size, num_beams * vocab_size)
                next_scores = next_scores.view(batch_size, num_beams * vocab_size)

                next_length = input_ids.shape[1] + 1
                next_scores_length_normalized = self.normalize_scores_by_length(next_scores, next_length, length_penalty)

                # ----------------
                # 5. Select top `beam_expansion * num_beams` candidates
                # ----------------
                # We temporarily select the top beam_expansion * num_beams candidate beams to provide additional options for exploration. This buffer ensures
                # 1. maintaining enough active beams, even if some beams terminate early with an EOS token;
                # 2. and overall increasing the likelihood of discovering higher-quality sequences.
                # Also note the top next beams may come from the same previous beam.
                # This one key characteristic that makes beam search different from simply keeping the top token from each beam.
                # next_scores shape: (batch_size, beam_expansion * num_beams)
                # next_tokens shape: (batch_size, beam_expansion * num_beams)
                #
                # Continuing our example with beam_expansion = 2, num_beams = 2, and length_penalty = 1.5:
                # Assuming input_ids.shape[1] = 5 (current sequence length), next_length = 6 (after adding next token)
                #
                # Let's start with our raw next_scores:
                # [
                #   [-0.7, -2.0, -3.5, -1.9, -3.3, -1.6],  # Batch item 0, all (beam + next token) scores
                #   [-1.8, -1.1, -3.3, -2.0, -3.3, -2.7]   # Batch item 1, all (beam + next token) scores
                # ]
                #
                # Applying length penalty: next_scores_length_normalized = next_scores / (6 ** 1.5)
                # With 6^1.5 ≈ 14.7, our normalized scores become:
                # [
                #   [-0.048, -0.136, -0.238, -0.129, -0.224, -0.109],  # Batch item 0, normalized scores
                #   [-0.122, -0.075, -0.224, -0.136, -0.224, -0.184]   # Batch item 1, normalized scores
                # ]
                #
                # Now we select the top 4 candidates based on normalized scores:
                # For batch item 0, sorting the normalized scores:
                # Top 4 normalized scores: [-0.048, -0.109, -0.129, -0.136]
                # These correspond to positions [0, 5, 3, 1] in the flattened array
                #
                # For batch item 1, sorting the normalized scores:
                # Top 4 normalized scores: [-0.075, -0.122, -0.136, -0.184]
                # These correspond to positions [1, 0, 3, 5] in the flattened array
                #
                # After topk on normalized scores, we have:
                # next_tokens = [
                #   [0, 5, 3, 1],  # Batch item 0 - positions in flattened array
                #   [1, 0, 3, 5]   # Batch item 1 - positions in flattened array
                # ]
                k = int(beam_expansion * num_beams)
                next_scores_length_normalized, next_tokens = torch.topk(
                    next_scores_length_normalized, k, dim=1, largest=True, sorted=True
                )

                # Get original scores at the positions specified by next_tokens.
                # We need to keep the original scores for score accumulation. The `torch.gather` operation
                #   helps us retrieve these original scores at the positions selected by next_tokens:
                #
                # `torch.gather(next_scores, dim=1, index=next_tokens)` works by:
                # 1. For each row in next_scores
                # 2. Select elements at indices specified by the corresponding row in next_tokens
                #
                # For batch item 0:
                # - Original scores: [-0.7, -2.0, -3.5, -1.9, -3.3, -1.6]
                # - Positions to select: [0, 5, 3, 1]
                # - Selected scores: [-0.7, -1.6, -1.9, -2.0]
                #
                # For batch item 1:
                # - Original scores: [-1.8, -1.1, -3.3, -2.0, -3.3, -2.7]
                # - Positions to select: [1, 0, 3, 5]
                # - Selected scores: [-1.1, -1.8, -2.0, -2.7]
                #
                # Result of gather operation:
                # next_scores = [
                #   [-0.7, -1.6, -1.9, -2.0],  # Batch item 0, original scores at selected positions
                #   [-1.1, -1.8, -2.0, -2.7]   # Batch item 1, original scores at selected positions
                # ]
                next_scores = torch.gather(next_scores, dim=1, index=next_tokens)

                # ----------------
                # 6. Decode beam indices and token indices
                # ----------------
                # Calculate which beam each candidate came from based on `next_tokens`.
                # source_beam_id/next_token_id shape: (batch_size, beam_expansion * num_beams)

                source_beam_id = next_tokens // vocab_size # Calculate which beam each candidate came from
                next_token_id = next_tokens % vocab_size # Calculate which token to add to each beam

                # ----------------
                # 7. Build new beams
                # ----------------
                # This step requires a loop-based approach rather than pure tensor operations for several reasons:
                #
                # 1. CONDITIONAL SELECTION: We need to handle different logic for finished vs. unfinished batch items
                #    - For finished batch items, we preserve the best beam. There is no need to further process this batch.
                #    - For unfinished batch items, we perform filtering and selection of active beams.
                #
                # 2. EOS TOKEN SPECIAL HANDLING: Completed candidates (ending with EOS) need special treatment
                #    - Candidates ending with EOS are kept as potential final answers, but not used for further expansion
                #    - This requires conditional logic that's difficult to vectorize
                #
                # 3. DYNAMIC FILTERING: The number of candidates varies based on EOS tokens and beam candidates
                #    - We need to prioritize EOS-ending beams even if they might not be in the top-N scores
                #    - We need to continue collecting beams until we have enough active beams
                #
                # 4. CAPACITY MANAGEMENT: We need to ensure we get exactly num_beams active continuations
                #    - This requires tracking how many we've collected so far (beam_idx counter)
                #    - Early stopping when we've collected enough non-EOS beams

                # Initialize containers for new beams (active beams for next loop)
                new_beam_scores = torch.zeros((batch_size, num_beams), device=device)
                new_beam_tokens = torch.zeros((batch_size, num_beams), device=device, dtype=torch.long)
                new_beam_ids = torch.zeros((batch_size, num_beams), device=device, dtype=torch.long)

                # Process each batch item separately
                for batch_idx in range(batch_size):
                    # ----------------
                    # 7.1 Handle finished batch items
                    # ----------------
                    if finished_sequence_flags[batch_idx]:
                        # For finished sequences, copy the best beam across all positions.
                        # This sets values in `new_beam_scores`, `new_beam_tokens`, and `new_beam_ids` for completed batch items
                        #   to maintain consistency with ongoing batch items, which is necessary for correctly updating `input_ids` later.
                        # Since `next_scores` and `next_token_id` are already sorted by the `topk` operation above,
                        #   the highest-scoring candidate is at index 0.
                        # Also note that the previous call to `process_logits_for_finished_sequences` ensures that
                        #   the next token for finished sequences will always be the padding token.
                        new_beam_scores[batch_idx, :] = next_scores[batch_idx, 0]
                        new_beam_tokens[batch_idx, :] = next_token_id[batch_idx, 0]
                        new_beam_ids[batch_idx, :] = source_beam_id[batch_idx, 0]
                        continue

                    # ----------------
                    # 7.2 Process unfinished batch items
                    # ----------------
                    beam_idx = 0 # Counter for active beams (those that don't end with EOS)

                    # Consider all candidates in order of decreasing score (highest scores first)
                    for score_idx in range(beam_expansion * num_beams):
                        # Get details of this candidate
                        beam_score = next_scores[batch_idx, score_idx]
                        beam_next_token = next_token_id[batch_idx, score_idx]
                        beam_id = source_beam_id[batch_idx, score_idx]

                        # ----------------
                        # 7.3 Check for EOS token
                        # ----------------
                        is_eos = (eos_token_id is not None and beam_next_token.item() == eos_token_id)

                        # ----------------
                        # 7.4 Handle EOS and non-EOS tokens differently
                        # ----------------
                        if is_eos:
                            # Get the source beam's tokens by computing its offset in the flattened input_ids
                            src_beam_offset = batch_idx * num_beams + beam_id
                            beam_tokens = input_ids[src_beam_offset].clone()

                            # For EOS tokens, store as completed sequence
                            # First, create the full sequence by appending the EOS token
                            # `beam_next_token` is a scalar so it needs `.unsqueeze(0)` to turn it into an array to concat with `beam_tokens`.
                            complete_sequence = torch.cat([beam_tokens, beam_next_token.unsqueeze(0)], dim=0)

                            # Store this completed sequence and its length-normalized score
                            completed_sequences[batch_idx].append(complete_sequence)
                            completed_scores[batch_idx].append(next_scores_length_normalized[batch_idx, score_idx])

                            # Note: We don't increment beam_idx here because EOS sequences aren't used for further expansion
                        else:
                            # Add candidate to active beams only if there is room left and candidate does not end with EOS token.
                            if beam_idx < num_beams:
                                new_beam_ids[batch_idx, beam_idx] = beam_id
                                new_beam_tokens[batch_idx, beam_idx] = beam_next_token
                                new_beam_scores[batch_idx, beam_idx] = beam_score
                                beam_idx += 1  # Increment active beam counter

                    # ----------------
                    # 7.5 Check batch completion
                    # ----------------
                    # Mark as finished if we have collected at least num_beams completed sequences
                    if eos_token_id is not None and len(completed_sequences[batch_idx]) >= num_beams:
                        finished_sequence_flags[batch_idx] = True
                    elif beam_idx < num_beams:
                        # Handling left-over beams by filling in dummy values for unused beam slots (if any)
                        for j in range(beam_idx, num_beams):
                            # We'll use -inf score to ensure this beam won't be selected later
                            new_beam_scores[batch_idx, j] = float('-inf')
                            new_beam_tokens[batch_idx, j] = pad_token_id if pad_token_id is not None else 0
                            new_beam_ids[batch_idx, j] = 0  # safe fallback, will be ignored

                # ----------------
                # 8. Update beam state
                # ----------------
                # Obtains global beam indexes considering the overall `batch_size * num_beams` beams.
                # For example,
                # new_beam_ids = tensor([
                #     [0, 1],  # For batch 0, selected beam indices 0 and 1 (local indices)
                #     [1, 0],  # For batch 1, selected beam indices 1 and 0 (local indices)
                # ])
                # We flatten this first by `new_beam_indices.view(-1)` and obtain [0, 1, 1, 0].
                # Meanwhile, `(torch.arange(batch_size) * num_beams)` gives the starting beam index for each batch item,
                #   and in this example it is [0, 2].
                # Repeat interleaved by num_beams times by `.repeat_interleave(num_beams)`, we have [0, 0, 2, 2].
                # As a result, beam_indices = [0, 1, 1, 0] + [0, 0, 2, 2] = [0, 1, 3, 2],
                #   which are the global beam indexes considering the global `2 * 2 = 4` (`batch_size * num_beams`) beams.
                beam_indices = new_beam_ids.view(-1) + (torch.arange(batch_size, device=device) * num_beams).repeat_interleave(num_beams, dim=0)

                # Append next tokens to selected beams
                next_token_ids = new_beam_tokens.view(-1).unsqueeze(1)
                input_ids = torch.cat([input_ids[beam_indices], next_token_ids], dim=-1)

                # Update beam scores
                beam_scores = new_beam_scores.view(-1)

            # ----------------
            # 9. Prepare final output
            # ----------------
            # Determine maximum sequence length for output tensor
            max_seq_len = input_ids.shape[1]

            # Initialize output tensor with padding tokens
            output_ids = torch.full((batch_size, max_seq_len),
                                    pad_token_id if pad_token_id is not None else 0,
                                    device=device, dtype=torch.long)

            # For each batch item, select the best sequence (either completed or in-progress)
            for batch_idx in range(batch_size):
                # Check if we have any completed sequences
                has_completed = len(completed_sequences[batch_idx]) > 0

                if has_completed:
                    # Find the best completed sequence
                    best_seq_idx = torch.tensor(completed_scores[batch_idx], device=device).argmax().item()
                    best_sequence = completed_sequences[batch_idx][best_seq_idx]

                    # Copy to output tensor (will be padded if shorter than max_seq_len)
                    seq_len = best_sequence.size(0)
                    output_ids[batch_idx, :seq_len] = best_sequence
                else:
                    # If no completed sequences, find best in-progress beam
                    beam_offset_start = batch_idx * num_beams
                    beam_offset_end = beam_offset_start + num_beams

                    # Extract scores for this batch's beams with length normalization
                    batch_beam_scores = self.normalize_scores_by_length(
                        beam_scores[beam_offset_start:beam_offset_end], input_ids.shape[1], length_penalty
                    )

                    # Find best beam
                    best_beam_idx = batch_beam_scores.argmax().item()
                    best_beam_offset = beam_offset_start + best_beam_idx

                    # Copy best in-progress sequence to output
                    best_sequence = input_ids[best_beam_offset]
                    seq_len = best_sequence.size(0)
                    output_ids[batch_idx, :seq_len] = best_sequence

            return output_ids


Beam Width & Search Expansion
"""""""""""""""""""""""""""""

The :newconcept:`Beam Width` (``num_beams``) represents how many candidate sequences to maintain at each decoding step. Unlike greedy search which considers only the single most probable continuation, beam search explores multiple possibilities in parallel:

* **Beam Width Trade-offs**:

  * **Higher values** (more beams, e.g., 5-10) provide more thorough exploration of the search space, :ub:`benefiting diversity and difficult generation scenarios`.
  * **Lower values** (less beams, e.g., 1-4) reduce computational requirements but may miss better solutions and reduce outcome diversity.
  * When ``num_beams = 1``, beam search becomes equivalent to greedy search

* **Task-Specific Recommendations**:

  .. list-table::
     :header-rows: 1
     :widths: 25 20 55

     * - Task Type
       - Recommended Range
       - Rationale
     * - **Translation**
       - 4-8
       - Balance between fluency and accuracy; larger values benefit difficult language pairs
     * - **Summarization**
       - 4-6
       - Helps maintain coherence while capturing key information
     * - **Open-ended Generation**
       - 5-10
       - Wider exploration for creative tasks with multiple valid continuations
     * - **Constrained Generation**
       - 3-5
       - Narrower search space when outputs must follow specific patterns

* **Scaling Considerations**:

  * **Sequence Length**: :ub:`Longer target sequences benefit from larger beam widths` (8-12).
  * **Model Size**: :ub:`The benefits of wider beams increase with more expressive and powerful models` (i.e., typically benefiting larger models).
  * **Diminishing Returns**: :ub:`Performance typically plateaus after 10 beams`  for most tasks.

    * There is known phenomenon called :newconcept:`Beam Search Curse` where very large beam widths (15+) can paradoxically produce worse outputs by favoring common, generic sequences over more appropriate specific ones

  * **Hardware Constraints**: For resource-constrained environments, consider 2-4 beams as a practical maximum.

The :newconcept:`Search Expansion Factor` (``beam_expansion``) is a multiplier that temporarily increases the beam pool size to ``beam_expansion * num_beams`` during the search process. This mechanism enhances beam search by creating a richer candidate pool before filtering back down to ``num_beams``:

* :ub:`Addressing Early Termination`: When sequences complete (by generating EOS tokens), the expansion ensures enough candidates remain to maintain ``num_beams`` active sequences.
* :ub:`Mitigating Search Collapse`: Helps avoid situations where all beams converge prematurely on very similar sequences.
* :ub:`Improving Exploration Efficiency & Diversity`: By considering more candidates at each step but carrying forward only the most promising, the approach balances exploration breadth with computational efficiency. Allowing consideration of more sequences that might initially score lower but could lead to better final results also improves diversity.

* **Parameter Selection**:

  * **Default Value**: 1.5-2.0 provides a good exploration-efficiency balance.
  * **Conservative Setting**: 1.2-1.5 when resources are limited.
  * **Aggressive Exploration**: 2.0-3.0 for tasks requiring high diversity.

Length Penalty
""""""""""""""

:newconcept:`Length Penalty` is a crucial mechanism in beam search that balances between shorter and longer sequence generations. Without this correction, beam search has an inherent bias toward shorter sequences because:

1. **Probability Multiplication Effect**: Each token added to a sequence multiplies the overall probability by a value less than 1 (or adds a negative log probability), naturally causing longer sequences to have lower cumulative scores.
2. **Search Space Pruning**: During beam search, shorter sequences might monopolize the available beams, preventing the exploration of potentially better but longer sequences.

The standard length penalty formula (used in Transformer-based models like BERT, GPT, and T5) is:

.. math::

   \text{LengthPenalty}(L) = (\frac{5 + L}{5 + 1})^{\alpha}

where :math:`L` is the length of the generated sequence, and

* :math:`\alpha` is the length penalty hyperparameter (commonly between 0.6 and 1.0).

  * :math:`\alpha = 0`: **No penalty** - Original scores are used without modification, which typically favors shorter sequences.
  * :math:`0 < \alpha < 1`: **Mild penalty** - Moderately encourages longer sequences.
  * :math:`\alpha = 1`: **Linear penalty** - The penalty grows linearly with sequence length.
  * :math:`\alpha > 1`: **Strong penalty** - Aggressively favors longer sequences.

* The "5+" constant in the formula serves to dampen the penalty effect for very short sequences, ensuring that the penalty scales more smoothly and reasonably when the sequence length is very short (e.g., 1~3 tokens).

Higher $α$ values not only encourage longer sequences but can also increase diversity among the top beams. Simply put, :ub:`encouraging longer sequences provides more opportunities to explore different possible paths` - by more aggressively rewarding length, higher α values enable the consideration of alternative trajectories that might initially have lower probabilities but lead to more diverse final outputs. This effectively prevents "beam collapse" where multiple beams converge on nearly identical sequences with only minor variations.

  * **Domain-Specific Tuning**: Different domains may require different $α$ values, with higher $α$ for better diversity.

    * Translation tasks often use values around $α \\in [0.6, 0.8]$.
    * Summarization might use $α \\in [0.8, 1.0]$.
    * Creative text generation can benefit from $α \\in [1.0, 1.2]$.

The final score calculation during beam search becomes:

.. math::

   \text{Score} = \frac{\log P(\text{sequence})}{\text{LengthPenalty}(L)}

This is often called :newconcept:`Length-Normalized Log-Probability`.

In implementation, there is also a design choice between :ub:`early vs. late length normalization`. The length penalty can be applied at each step (early, more encouraging longer sequences and diversity) or only when comparing final sequences (late, more efficient). The code example above applies early path normalization.

.. admonition:: Origin and Purpose of the Constants
    :class: note

    1. **The "5+" Constant**:

       The addition of 5 to the sequence length serves several critical purposes:

       * **Smoothing Effect**: It prevents extreme penalties for very short sequences. Without this constant, sequences of length 1 or 2 would receive dramatically different penalties compared to slightly longer ones.

       * **Stability for Short Sequences**: For short sequences ($L < 5$), the penalty differences would be too extreme without this offset, potentially causing unstable behavior during early generation steps.

       * **Gradual Scaling**: The "5+" ensures that the penalty scales more gradually across the entire range of possible sequence lengths, creating a more balanced comparison.

    2. **The Denominator (5+1)**:

       * **Normalization Factor**: This ensures that a sequence of length 1 has a penalty value of $(\\frac{6}{6})^α = 1$ when $α > 0$.
       * **Reference Point**: It establishes a reference point against which all other sequence lengths are compared.
       * **Baseline Calibration**: This denominator calibrates the entire penalty function, ensuring that very short sequences aren't unduly penalized or favored.

    **Practical Implications**:

    1. **Effect on Different Sequence Lengths**:

       For $α = 1.0$, the penalties scale as follows:

       * $L = 1$: $\\frac{5+1}{6} = 1.0$ (no effect)
       * $L = 5$: $\\frac{5+5}{6} = 1.67$ (moderate penalty)
       * $L = 10$: $\\frac{5+10}{6} = 2.5$ (stronger penalty)
       * $L = 20$: $\\frac{5+20}{6} = 4.17$ (significant penalty)

    2. **Modified Constants**:

       Some implementations use different constants (like $3+L$ or $7+L$) based on empirical results for specific tasks:

       * **Smaller Constants** (e.g., $3+L$): Create more aggressive length penalties that differentiate more strongly between short sequences.
       * **Larger Constants** (e.g., $7+L$): Produce more conservative penalties with gentler scaling.

    3. **Historical Context**:

       The $5+L$ formulation was first introduced in Google's Neural Machine Translation (GNMT) system and has since become a standard in many sequence generation models. The choice of 5 was determined through empirical testing across various language pairs and generation tasks.


