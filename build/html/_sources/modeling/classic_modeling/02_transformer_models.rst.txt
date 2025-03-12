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

         # Calculate raw attention scores (before applying softmax).
         # We use negative indexes to indicate transposing the last two dimension of K. We do not use positive indexes so the code can work in a more general way, as K might have 3 or 4 dimensions with "multi-head" and "batch".
         # Dividing by the square root of `d_head` is recommended by the paper is to prevent the raw attention scores from becoming too large for large dimensions.
         # Typical `attn_scores` dimensions are (`batch_size`, `num_heads`, `query_size`, `index_size`)
         # (or (`batch_size`, `num_heads`, `seq_length`, `seq_length`) in the sequence processing case).
         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
         
         if mask is not None:
             attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
             
         attn_weights = torch.softmax(attn_scores, dim=-1)
         output = torch.matmul(attn_weights, V)
         return output, attn_weights

  * This can be viewed as a :ub:`soft retrieval operation`, obtaining probabilistic scores for values in V through query-key interactions
  * Dividing by ``sqrt(d_head)`` stabilizes gradients by preventing attention scores from becoming too large for high dimensions

* **Forward Pass Implementation**: The forward method demonstrates how multi-head attention is actually computed:

  .. code-block:: python
     :class: folding
     :name: multi_head_attention_forward

     def forward(self, Q, K, V, mask=None):
         batch_size = Q.size(0)

         # Linear transformations and reshape to separate heads
         # Transform from (batch_size, seq_length, d_model) to (batch_size, num_heads, seq_length, d_head)
         Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
         K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
         V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

         # Apply scaled dot-product attention independently to each head
         output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

         # Reshape and apply final linear transformation
         # Transform back from (batch_size, num_heads, seq_length, d_head) to (batch_size, seq_length, d_model)
         output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

         # Apply output transformation on the `d_model` dimension.
         output = self.W_o(output)

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

* **Purpose**: Positional encoding can be viewed :ub:`one type of multi-modal learning`.

  * It combined position information with the token embeddings, allowing the model to distinguish between different positions in the sequence.
  * In the canonical transformer architecture, it is by simply adding the positional encodings to the token embeddings before input to the encoder layers, as well as adding to the encoder output before input to the decoder layers.

* **Implementation**: Uses sine and cosine functions of different frequencies to create unique position identifiers:

  .. code-block:: python
     :class: folding
     :name: transformer_positional_encoding

     class PositionalEncoding(nn.Module):
         def __init__(self, d_model, max_seq_length=5000):
             super().__init__()
             pe = torch.zeros(max_seq_length, d_model)
             position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
             div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

             # the following uses start:stop:step notation
             pe[:, 0::2] = torch.sin(position * div_term)  # Sine for even indices
             pe[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd indices

             # The original pe tensor has shape [`max_seq_length`, `d_model`],
             # and `unsqueeze(0)` adds a new dimension at index 0.
             # After unsqueeze, shape becomes [1, `max_seq_length`, `d_model`],
             # and this extra dimension allows for batch processing.

             # `self.register_buffer` in PyTorch is a method used to register a tensor as a buffer
             # that should be saved along with model parameters during model.state_dict() calls,
             # but is not considered a model parameter (meaning it doesn't receive gradients during backpropagation).
             self.register_buffer('pe', pe.unsqueeze(0))

         def forward(self, x):
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

Position-Wise Feed-Forward Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Between attention layers, the Transformer employs :newconcept:`Position-wise Feed-Forward Networks`, a fully connected feedforward neural network applied independently and identically to each data point in the sequence. This means that the same feedforward network is applied to each data point's representation without any interaction between different positions at this stage.

* **Implementation**: Two linear transformations with a ReLU activation in between,  enhancing the model's representational capacity:

  .. code-block:: python
     :class: folding
     :name: transformer_positionwise_feedforward

     class PositionWiseFeedForward(nn.Module):
         def __init__(self, d_model, d_ff):
             super().__init__()
             self.fc1 = nn.Linear(d_model, d_ff)
             self.fc2 = nn.Linear(d_ff, d_model)
             self.relu = nn.ReLU()

         def forward(self, x):
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

.. note:: Batch Normalization vs. Layer Normalization

   :newconcept:`Batch Normalization` is another common normalizing technique applied before the Transformer era. While both normalization techniques serve similar purposes, they differ in several key aspects:
   
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
----------------------------

The Transformer follows an :newconcept:`Encoder-Decoder Architecture`, with the encoder processing the input sequence and the decoder generating the output sequence.

.. note::

     In the canonical transformer, :ub:`positional encoding is not applied inside encoder/decoder layers`. They have been added to the encoder/decoder input token embeddings (early fusion).

* **Encoder Layer**: Each encoder layer consists of:

  1. Multi-head :ub:`self-attention` mechanism.
  2. :ub:`Position-wise feed-forward` network.
  3. :ub:`Layer norm with dropout and residuals` twice after each of above self-attention and feed-forward layers.

     .. code-block:: python
          :class: folding
          :name: transformer_encoder_layer

          class EncoderLayer(nn.Module):
              def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
                  super().__init__()
                  self.self_attn = MultiHeadAttention(d_model, num_heads)
                  self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
                  self.norm1 = nn.LayerNorm(d_model)
                  self.norm2 = nn.LayerNorm(d_model)
                  self.dropout = nn.Dropout(dropout)

              def forward(self, x, mask=None):
                  # Self-attention block with residual connection and layer normalization
                  attn_output, _ = self.self_attn(x, x, x, mask)
                  x = self.norm1(x + self.dropout(attn_output))

                  # Feed-forward block with residual connection and layer normalization
                  ff_output = self.feed_forward(x)
                  x = self.norm2(x + self.dropout(ff_output))

                  return x

* **Decoder Layer**: Each decoder layer has:

  1. Masked multi-head :ub:`self-attention` (must prevent attending to future positions)
  2. Multi-head :ub:`cross-attention` over encoder outputs
  3. :ub:`Position-wise feed-forward` network.
  4. :ub:`Layer norm with dropout and residuals` three times after each of above layers.

     .. code-block:: python
          :class: folding
          :name: transformer_decoder_layer

          class DecoderLayer(nn.Module):
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
                  # Self-attention block with residual connection and layer normalization
                  # This attention is masked to prevent attending to future positions
                  self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
                  x = self.norm1(x + self.dropout(self_attn_output))

                  # Cross-attention block with residual connection and layer normalization
                  # Attends to encoder output based on decoder queries
                  cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
                  x = self.norm2(x + self.dropout(cross_attn_output))

                  # Feed-forward block with residual connection and layer normalization
                  ff_output = self.feed_forward(x)
                  x = self.norm3(x + self.dropout(ff_output))

                  return x

* **Attention Masking**: Two types of masks are employed:

  1. **Source Mask**: Handles padding in the input sequence
  2. **Target Mask**: Combines padding mask with a future-masking mechanism to prevent the decoder from accessing future tokens

  .. code-block:: python
       :class: folding
       :name: transformer_masking

       def generate_mask(self, src, tgt):
           # The `unsqueeze(1).unsqueeze(2)` operations add extra dimensions to the `src` tensor, transforming its shape from [`batch_size`, `seq_length`] to [`batch_size`, 1, 1, `seq_length`]
           src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

           # The `unsqueeze(1).unsqueeze(3)` operations add extra dimensions to the `tgt` tensor, transforming its shape from [`batch_size`, `seq_length`] to [`batch_size`, 1, `seq_length`, 1]
           tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
           seq_length = tgt.size(1)

           # Prevent the decoder from accessing future tokens in the target sequence.
           # The resulting nopeek_mask is a boolean mask of shape [1, seq_length, seq_length], with lower triangle (diagonal included) being True and all others being False.
           # It looks like this for a sequence length of 5:
           # [[[ True, False, False, False, False],
           #   [ True,  True, False, False, False],
           #   [ True,  True,  True, False, False],
           #   [ True,  True,  True,  True, False],
           #   [ True,  True,  True,  True,  True]]]
           nopeek_mask = ~(torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1).bool())

           # Consider the target is represented by single tensor tgt = [[5, 3, 7, 0, 0]] (i.e., batch size is 1), where 0 denotes padding tokens.
           # The bitwise AND operation results in:
           # [[[True, False, False, False, False],
           #    [True,  True, False, False, False],
           #    [True,  True,  True, False, False],
           #    [False, False, False, False, False],
           #    [False, False, False, False, False]]]
           tgt_mask = tgt_mask & nopeek_mask

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
                src_mask, tgt_mask = self.generate_mask(src, tgt)

                # Encoder
                # Early fusion of positional encoding with encoder input tokens
                enc_input = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

                enc_output = enc_input
                for enc_layer in self.encoder_layers:
                    enc_output = enc_layer(enc_output, src_mask)

                # Decoder
                # Early fusion of positional encoding with decoder input tokens
                dec_input = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
                dec_output = dec_input

                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

                output = self.fc(dec_output)
                return output
