
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256 # waay bigger than before
max_iters = 5000
eval_interval = 500
lr = 3e-4
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# open training file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# character level encoding
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# encode dataset
data = torch.tensor(encode(text), dtype=torch.long).to(device)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters).to(device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
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
        # idx is (B, T)
        for _ in range(max_new_tokens):
            # only take context of the last tokens that fit in the block size
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class Block(nn.Module):
    """A whole attention block, layering attention and computation."""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        # The tokens look at each other, but now they need time to process that attention and do something w it
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # literally just adding is creating a residual connection
        # this allows gradients to flow between x's, not JUST through the self attention and feed forward,
        # important because as the network gets deeper the gradients need help flowing backwards
        # layer norm before attention and feed forward, instead of after as in Vaswani et al
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Head(nn.Module):
    '''Self attention head.
        Kinda like a group convolution? I guess just denser, since we still maintain the same block size,
        focusing more attention on the same embedding size
    '''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # not a model parameter, assign it with this
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # dropout randomly shuts down some neurons to 0 every single pass (at training time)
        # This creates different independent ensembles of neurons that are all turned on at inference time
        # prevents overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)

        # attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, -1)
        wei = self.dropout(wei)
        # do the weighted averaging
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    '''multi head attention'''

    def __init__(self, num_heads, head_size):
        super().__init__()
        # literally just a bunch of attention heads???
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # four communication channels in parallel, these have to be smaller
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Why is this projection layer necessary for residual cnxs?
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # fattening up the feed forward network by a factor of 4
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)



m = BigramLanguageModel()
m = m.to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=lr)

# training time!
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"{iter}: train {losses['train']}, val: {losses['val']}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))