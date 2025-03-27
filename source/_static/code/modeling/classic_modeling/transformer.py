import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism that splits the model dimension (d_model)
    into multiple heads, each head computing scaled dot-product attention in parallel.
    The outputs of all heads are then concatenated and projected back to d_model.

    Motivation:
        Having multiple heads allows the model to attend to different features
        or subspaces of the input representation independently.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

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


# Example usage:
def create_transformer(src_vocab_size=1000, tgt_vocab_size=1000):
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    return model


# Text preprocessing utilities
# Text preprocessing utilities
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.next_idx = len(self.word2idx)

    def fit(self, texts):
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.next_idx
                    self.idx2word[self.next_idx] = word
                    self.next_idx += 1

    def encode(self, text, max_length=None):
        tokens = [self.word2idx.get(word, self.word2idx['<UNK>'])
                  for word in text.lower().split()]
        tokens = [self.word2idx['<START>']] + tokens + [self.word2idx['<END>']]

        if max_length is not None:
            tokens = tokens[:max_length]
            tokens = tokens + [self.word2idx['<PAD>']] * (max_length - len(tokens))

        return torch.tensor(tokens)

    def decode(self, tokens):
        return ' '.join([self.idx2word[token.item()]
                         for token in tokens
                         if token.item() not in [self.word2idx['<PAD>'],
                                                 self.word2idx['<START>'],
                                                 self.word2idx['<END>']]])


# Example data and training setup
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


def train_epoch(model, optimizer, criterion, src_data, tgt_data, src_tokenizer, tgt_tokenizer, max_length=50):
    model.train()
    total_loss = 0

    for src_text, tgt_text in zip(src_data, tgt_data):
        optimizer.zero_grad()

        # Prepare input data
        src = src_tokenizer.encode(src_text, max_length=max_length).unsqueeze(0)
        tgt = tgt_tokenizer.encode(tgt_text, max_length=max_length).unsqueeze(0)

        # Create target for loss calculation (shifted right)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        output = model(src, tgt_input)

        # Calculate loss
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def generate_summary(model, src_text, src_tokenizer, tgt_tokenizer, max_length=50):
    model.eval()

    with torch.no_grad():
        # Encode input text
        src = src_tokenizer.encode(src_text, max_length=max_length).unsqueeze(0)

        # Initialize target with START token
        tgt = torch.tensor([[tgt_tokenizer.word2idx['<START>']]])

        # Generate summary token by token
        for _ in range(max_length - 1):
            output = model(src, tgt)
            next_token = output[:, -1].argmax(dim=-1).unsqueeze(1)
            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == tgt_tokenizer.word2idx['<END>']:
                break

    return tgt_tokenizer.decode(tgt[0])


# Training example
def train_summarization_model():
    # Create dummy data
    articles, summaries = create_dummy_summarization_data(10)

    # Initialize tokenizers
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()

    # Fit tokenizers
    src_tokenizer.fit(articles)
    tgt_tokenizer.fit(summaries)

    # Create model
    model = create_transformer(
        src_vocab_size=len(src_tokenizer.word2idx),
        tgt_vocab_size=len(tgt_tokenizer.word2idx)
    )

    # Training setup
    # `nn.CrossEntropyLoss` expects logits — the raw, unnormalized scores output by the network—rather than probabilities.
    #  Internally, `nn.CrossEntropyLoss` applies a logarithm and a softmax
    #  (i.e., it uses the log-softmax formulation) before computing the loss.
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        loss = train_epoch(
            model, optimizer, criterion,
            articles, summaries,
            src_tokenizer, tgt_tokenizer
        )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    return model, src_tokenizer, tgt_tokenizer


# Example usage
if __name__ == "__main__":
    # Train model
    model, src_tokenizer, tgt_tokenizer = train_summarization_model()

    # Test on training examples
    test_articles = [
        "the cat sat on the mat and watched the birds fly by in the clear blue sky",
        "scientists discover new species of butterfly in the amazon rainforest last summer"
    ]

    print("\nGenerated Summaries:")
    for article in test_articles:
        summary = generate_summary(model, article, src_tokenizer, tgt_tokenizer)
        print(f"\nArticle: {article}")
        print(f"Summary: {summary}")
