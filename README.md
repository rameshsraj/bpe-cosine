# bpe-cosine

Two small, dependency-light scripts that show what actually happens to a prompt
before a language model ever sees it:

1. **`bpe.py`** — trains a Byte Pair Encoding tokenizer from scratch and encodes a prompt into token IDs.
2. **`cosine.py`** — ranks documents against a query using TF-IDF vectors and cosine similarity, then shows where lexical matching breaks.

Both use the same running example — an internal-policy question,
`"What is our retention period for Slack messages?"` — so the two halves of a
typical retrieval pipeline can be read back to back.

## Requirements

- Python 3.9+
- `bpe.py`: standard library only
- `cosine.py`: `numpy`, `scikit-learn`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn
```

## Running

```bash
python bpe.py
python cosine.py
```

Each script is self-contained: the corpus and the prompt are defined at the top
of the file, and everything prints to stdout. Edit the constants and re-run to
see how the results change.

## `bpe.py` — tokenization from scratch

The full BPE algorithm in about 100 lines of standard library Python:

- `get_word_counts` — splits text into words and appends an explicit `</w>` end-of-word marker so merges never cross word boundaries.
- `count_pairs` — counts every adjacent symbol pair, weighted by word frequency.
- `merge_pair` — replaces all occurrences of a pair with a single merged symbol.
- `train_bpe` — repeats the count/merge loop, stopping early once no pair occurs more than once (further merges would only memorise).
- `build_vocab` — assigns deterministic integer IDs (sorted by symbol length, then alphabetically).
- `tokenize` / `encode` — replays the learned merges **in the order they were learned** and maps symbols to IDs.

Training on the 150-word corpus in the file, 120 merges produce a 148-symbol
vocabulary. The first merges are pure character pairs (`'r' + 'e' -> 're'`),
and later ones stack on top of earlier results until whole words appear:

```
  12. 're' + 'te'        ->  'rete'
  13. 'rete' + 'n'       ->  'reten'
  14. 'reten' + 'tion</w>' -> 'retention</w>'
```

Nothing in the code knows that "retention" is a word. It only knows which pairs
of symbols co-occur, and the word falls out of the frequency statistics.

The prompt then encodes to 17 tokens for 8 words (≈2.12 tokens per word).
Frequent words in the corpus (`retention`, `period`, `for`) compress to a single
token; unseen ones decompose (`messages?` → `messag` + `es` + `?` + `</w>`, four
pieces, because that exact word never appears in the corpus). This is the
central property of subword tokenization: the tokenizer never fails on unseen
input, it just spends more tokens on it. The final section of the output prints
each prompt word next to its corpus frequency to make that trade-off explicit.

Because the corpus is tiny, the merges learned are specific to this text. That
is deliberate — real tokenizers run the same loop over hundreds of billions of
tokens, which is the only difference.

## `cosine.py` — retrieval and its failure mode

Six short policy documents, one query, and `TfidfVectorizer` +
`cosine_similarity` from scikit-learn. The output runs in three parts.

**Part 1 — the vector space.** The query is shown as an actual vector. With
English stop words removed, only two dimensions are non-zero (`retention`,
`slack`); the other 43 are exactly `0.0`. TF-IDF vectors are sparse, learned
embeddings are dense, but the geometry used to compare them is identical.

**Part 2 — ranked retrieval.** The intended document ranks first at `0.5333`.
Several documents score exactly `0.0000` — they share no vocabulary with the
query, so they sit at 90° to it in this space.

**Part 3 — where lexical matching breaks.** The same fitted vectorizer is
re-used (`vectorizer.transform`, not `fit_transform`, so the vocabulary is
frozen as it would be in a real index) and given a paraphrase:

> `"How long do we keep chat history before it is wiped?"`

Every score collapses to `0.0000`. A total retrieval failure, even though a
human reads that question and immediately knows which document answers it.
`"how long"` and `"three years"` share no tokens, and neither do
`"chat history"` and `"message retention"`.

That gap is exactly what learned embedding models close: they place paraphrases
near each other because they were trained on text where those phrasings appear
in similar contexts. The retrieval mechanism — cosine similarity over vectors —
does not change at all. Only the quality of the vectors does.

## Modifying the examples

- `bpe.py`: change `NUM_MERGES` to watch vocabulary size and tokens-per-word trade off against each other, or replace `CORPUS` with your own text to see which words become single tokens.
- `cosine.py`: add documents to `CORPUS` (with matching entries in `LABELS`), or change `PARAPHRASE` to probe how much lexical overlap is needed before retrieval recovers.
