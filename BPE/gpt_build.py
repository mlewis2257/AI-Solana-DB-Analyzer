import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Load the data
# Download: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print(len(text))
# print(text[:300])

# 2. Build the vocabulary — get the sorted list of unique characters in `text`
chars = sorted(list(set(text)))
vocab_size = len(chars)

# print(vocab_size)
# print(''.join(chars))

# 3. Build encode/decode
#    stoi: dict mapping char -> integer
#    itos: dict mapping integer -> char
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# print(stoi['h'])
# print(itos[46])


def encode(s):
    # string -> list of integers
    return [stoi[c] for c in s]


def decode(l):
    # list of integers -> string
    return ''.join([itos[i] for i in l])


test = encode("Hello World")
# print(test)
# print(decode(test))
# 4. Encode the entire dataset into a tensor of integers (dtype=torch.long)
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

# 5. Split into train (first 90%) and val (last 10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# print(len(train_data), len(val_data))

torch.manual_seed(1337)
batch_size = 32
block_size = 8


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch('train')
# print('inputs:', xb.shape)
# print(xb)
# print('inputs:', yb.shape)
# print(yb)

# BigramLanguage models


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Self Attention

# torch.manual_seed(42)
# B, T, C = 4, 8, 32
# x = torch.randn(B, T, C)

# head_size = 16
# key = nn.Linear(C, head_size, bias=False)
# query = nn.Linear(C, head_size, bias=False)
# value = nn.Linear(C, head_size, bias=False)

# k = key(x)
# q = query(x)

# wei = q @ k.transpose(-2, -1)

# trill = torch.tril(torch.ones(T, T))
# wei = wei.masked_fill(trill == 0, float('-inf'))
# wei = F.softmax(wei, dim=-1)

# v = value(x)

# out = wei @ v * head_size**-0.5


class Head(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadedAttention(n_head, head_size, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# m = BigramLanguageModel(vocab_size)
# logits, loss = m(xb, yb)
# idx = torch.zeros((1, 1), dtype=torch.long)

# optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# n_embd = 32
# n_head = 4
# block = Block(n_embd, n_head)
# x = torch.randn(4, 8, n_embd)
# out = block(x)
# print(out.shape)


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop to last block_size tokens -- position table only has that many rows
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


n_embd = 32
n_head = 4
n_layer = 3

model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for steps in range(5000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 500 == 0:
        print(f"step {steps}: loss {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")

idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=300)[0].tolist()))
# for steps in range(10000):
#     xb, yb = get_batch('train')

#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

#     if steps % 1000 == 0:
#         print(f"step {steps}: loss {loss.item():.4f}")
# print(f"final loss: {loss.item():.4f}")
# print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
# print(logits.shape)
# print(loss)

# print(wei[0])
# print(out.shape)
# sanity check — don't touch, just verify these run once you fill in the above
# print(f"vocab_size: {vocab_size}")
# print(f"chars: {''.join(chars)}")
# print(f"data shape: {data.shape}, dtype: {data.dtype}")
# print(decode(encode("hello world")))  # should print "hello world"
