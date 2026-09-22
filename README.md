[README_RNN.md](https://github.com/user-attachments/files/32510981/README_RNN.md)
# Character-Level RNN — Next-Letter Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

A minimal recurrent neural network built from scratch in PyTorch that learns to predict the next character in a word. **225 trainable parameters, trained on a single word.**

The goal is not accuracy on a benchmark — it is to see exactly what happens inside a recurrent layer: where the memory lives, how it is updated, and why a network with memory can do something a feed-forward network cannot.

---

## Demo

```
$ Harflar kiriting: mak

  Prediction : mak + tab  ->  maktab
  Confidence : t: 99.8% | b: 0.1% | a: 0.1%
```

The model reconstructs the full word from any prefix:

| Input | Prediction | Output | Confidence (top-1) |
|---|---|---|---|
| `m` | `+aktab` | `maktab` | 99.8% |
| `ma` | `+ktab` | `maktab` | 99.8% |
| `mak` | `+tab` | `maktab` | 99.8% |
| `makt` | `+ab` | `maktab` | 99.9% |
| `z` | — | `Error: letter not in vocabulary` | — |

## Training results

| Metric | Value |
|---|---|
| Trainable parameters | 225 |
| Vocabulary size | 5 (`a`, `b`, `k`, `m`, `t`) |
| Hidden size | 10 |
| Optimizer / LR | Adam, 0.02 |
| Epochs | 200 |
| Loss | 8.1763 → **0.0083** |
| First fully correct prediction | epoch 20 |

Random-guess loss for 5 classes over 5 timesteps is `5 × ln(5) ≈ 8.05`, which is where training starts — the model begins with no knowledge at all and converges smoothly, without oscillation.

---

## How it works

```
"a" ──► one-hot ──► nn.RNN ──► nn.Linear ──► softmax ──► "k"
          [5]      ▲   │  [10]     [5]
                   │   │
       h(t-1) ─────┘   └────► h(t)   (memory, passed to the next step)
```

A single recurrent step is one formula:

```
h_t = tanh(W_ih · x_t + b_ih  +  W_hh · h_{t-1} + b_hh)
y_t = W_fc · h_t + b_fc
```

The network adds two signals — one from the **current letter**, one from the **accumulated history** — and squashes the sum into `[-1, 1]`. That addition is the entire memory mechanism.

Parameter breakdown:

| Tensor | Shape | Count | Role |
|---|---|---|---|
| `rnn.weight_ih_l0` | 10 × 5 | 50 | maps the incoming letter into hidden space |
| `rnn.bias_ih_l0` | 10 | 10 | input bias |
| `rnn.weight_hh_l0` | 10 × 10 | 100 | transforms old memory into new memory |
| `rnn.bias_hh_l0` | 10 | 10 | recurrent bias |
| `fc.weight` | 5 × 10 | 50 | scores every letter from the hidden state |
| `fc.bias` | 5 | 5 | per-letter bias |
| | | **225** | |

Training data is the word shifted by one position, which yields five supervised pairs:

```
x:  m  a  k  t  a
y:  a  k  t  a  b
```

---

## The interesting part: why `a` is the real test

The letter `a` appears twice in `maktab`, and each occurrence requires a **different** answer:

- position 2: `m·a` → `k`
- position 5: `kt·a` → `b`

The input tensor is identical in both cases — `[1, 0, 0, 0, 0]`. A feed-forward network cannot separate them; it would have to output the same letter both times.

The trained model gets both right, with 99.8% and 99.9% confidence. Inspecting the hidden state at those two steps shows why:

| | step 2 | step 5 |
|---|---|---|
| Input vector | `[1,0,0,0,0]` | `[1,0,0,0,0]` (identical) |
| Euclidean distance between the two hidden states | | **4.00** |
| Prediction | `k` | `b` |

The context is carried entirely by `h_{t-1}`. This is the one observation that justifies the whole architecture.

---

## Quickstart

**Google Colab** — open the notebook and run all cells (no setup, no GPU required):

```
notebooks/rnn_maktab.ipynb
```

**Local:**

```bash
git clone https://github.com/abdurrohmanq/<repo-name>.git
cd <repo-name>
pip install torch matplotlib
jupyter notebook notebooks/rnn_maktab.ipynb
```

Training takes a few seconds on CPU.

## Inference

```python
result = bashorat("mak", nechta=3)

# {'kirish': 'mak',
#  'bashorat': 'tab',
#  'toliq': 'maktab',
#  'top': [('t', 0.998), ('b', 0.001), ('a', 0.001)]}
```

The function normalizes input, rejects characters outside the vocabulary instead of raising `KeyError`, returns top-k probabilities alongside the prediction, and feeds each prediction back as the next input (free-running generation rather than teacher forcing).

---

## Limitations

These are intentional — the project is a study of the mechanism, not a production model.

- **Training set is one word.** There is no validation or test split, so the results demonstrate that the mechanism works, not that the model generalizes.
- **Out-of-distribution inputs produce confident nonsense.** Given `a` alone, the model predicts `a` with 96.7% confidence, and given `b`, it predicts `a` with 99.1%. Neither context exists in training. Softmax always sums to 1, so the model cannot express "I don't know" — a property worth remembering before shipping any classifier.
- **Vanilla RNN.** Long-range dependencies would require LSTM or GRU gating; on a six-letter word the difference does not appear.
- **Teacher forcing during training.** The gap between teacher-forced and free-running generation (exposure bias) does not show up at this scale, but it does on longer sequences.

## Possible next steps

- Train on a corpus of Uzbek words and evaluate on a held-out split
- Swap `nn.RNN` for `nn.LSTM` and compare on longer words
- Replace one-hot inputs with `nn.Embedding`
- Add temperature sampling instead of `argmax` for varied generation
- Sequence-to-label variant: use only the final hidden state to classify whole sequences

## Built with

PyTorch · NumPy · Matplotlib

## License

MIT — see [LICENSE](LICENSE).
