import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    # Multi-head is motivated by the idea that a value's embedding might have multiple "components",
    # part of the embedding might be about topic A, the other part might be about topic B, etc.,
    # and for different components the weighting might be different.

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
        # Scaled-doc product can be viewed as soft retrieval,
        # where we obtain probabilistic scores for values in V through Q and K.
        # The probabilistic scores are the attention scores.

        # Calculate raw attention scores (before applying softmax).
        # We use negative indexes to indicate transposing the last two dimension of K. We do not use positive indexes so the code can work in a more general way, as K might have 3 or 4 dimensions with "multi-head" and "batch".
        # Dividing by the square root of `d_head` is recommended by the paper is to prevent the raw attention scores from becoming too large for large dimensions.
        # Typical `attn_scores` dimensions are (`batch_size`, `num_heads`, `query_size`, `index_size`)
        # (or (`batch_size`, `num_heads`, `seq_length`, `seq_length`) in the sequence processing case).
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            # We can mask out keys/values by replacing their attention scores by
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights (along the last dimension / feature dimension, i.e. normalizing each row)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # The input Q, K, V are of size (`batch_size`, `index_size`, `d_model`)
        # Linear transformations and reshape.
        # 1. Keep the batch size.
        # 2. Breaking down the last dimension (the feature dimension) to two dimensions of size `num_heads` and `d_head`.
        # 3. The `-1` means the `view` function will infer the corresponding dimension (the index size dimension).
        # Transposing the index size dimension (dimension 1) and the `num_heads` dimension (dimension 2) is for computational convenience
        # because the scaled-doc product will be performed per batch per head.
        # After transposing, the dimensions are (`batch_size`, `num_heads`, `index_size`, `d_head`)
        # (or (`batch_size`, `num_heads`, `seq_length`, `d_head`) in the sequence processing case).
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Apply scaled dot-product attention
        output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Reshape and apply final linear transformation
        # 1. Transposing the dimensions back to (`batch_size`, `index_size`, `num_heads`, `d_head`) (or (`batch_size`, `seq_length`, `num_heads`, `d_head`) in the sequence processing case).
        # 2. The `contiguous` method enforces the underlying tensor representation be continuous.
        # 3. The `view` method merges the last two dimensions, so it becomes (`batch_size`, `index_size`, `d_model`) (or (`batch_size`, `seq_length`, `d_model`) in the sequence processing case).
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Apply output transformation on the `d_model` dimension.
        output = self.W_o(output)

        # NOTE:
        # The above approach is mathematically equivalent to perform multiple Q/K/V scaled-dot-products and then concatenate them.
        # However, above view/transpose operations are more efficient implementation.

        return output, attn_weights


class PositionWiseFeedForward(nn.Module):
    """
    A pointwise feedforward network refers to a fully connected feedforward neural network applied independently and identically to each data point in the sequence.
    This means that the same feedforward network is applied to each data point's representation without any interaction between different positions at this stage.

    Here, this feedforward network consists of two linear transformations with a non-linear activation function in between.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """
    Positional encoding provides information about the position of each data point in a sequence, enabling the model to capture the order of the sequence, which is essential for understanding context in tasks like language modeling.
    Since Transformers process input tokens simultaneously (i.e., in parallel), they lack inherent sequential information. Positional encoding would address this by adding unique position-based vectors to the token embeddings, allowing the model to distinguish between different positions in the sequence.

    The positional encoding (PE) used by Transformer is projecting integer positions onto a circle in 2D dimension, and use sine/cosine values to identify the positions on the circle.
    Dimension index is included in the PE formula for three reasons,
    1. PE need to be of the same dimension size as the model feature dimension size (i.e. `d_model`), in order to be compatible.
    2. Mathematically it allows representing the position at different scales.
    3. It helps each PE uniquely identify a position.
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
        # If x has shape [`batch_size`, `seq_length`, `d_model`]
        # pe[:, :x.size(1)] will broadcast from [1, `seq_length`, `d_model`]
        # to match x's shape.
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        # LayerNorm is normalizing the feature dimension using z-score (with perturbed variance)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        # Encode each data point by soft-retrieval of other data points in the inputs
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


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
        # Self-attention
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Cross-attention
        cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward
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
