from collections import Counter

PROMPT = "What is our retention period for Slack messages?"

# A deliberately small corpus, written to contain the kind of vocabulary an
# internal-policy assistant would see. Real tokenizers train on hundreds of
# billions of tokens; this trains on a few hundred words. The merges it learns
# are therefore local to this text, which is itself a useful thing to show.
CORPUS = """
Slack message retention is set to three years for all workspaces.
Legal hold overrides retention and suspends automatic deletion until released.
Email retention is seven years for regulated communications.
Teams message retention follows the same retention schedule as Slack.
Message retention settings are configured per workspace by administrators.
The retention period for archived channels is three years from archival.
Retention policies apply to messages, files, and direct messages.
Deletion is automatic once the retention period expires.
A legal hold suspends deletion for the messages in scope.
Administrators can review retention settings in the compliance console.
What is the retention period for direct messages in Slack?
The retention period is three years unless a legal hold applies.
Regulated messages are retained for the full statutory period.
Workspace owners are notified before automatic deletion occurs.
Compliance requires that retention settings are reviewed annually.
Messages in archived workspaces retain their original retention period.
"""


def get_word_counts(text):
    """Split into words, mark end-of-word with </w> so BPE keeps word boundaries."""
    counts = Counter()
    for word in text.lower().split():
        # Represent each word as a tuple of symbols, ending with </w>.
        symbols = tuple(word) + ("</w>",)
        counts[symbols] += 1
    return counts


def count_pairs(word_counts):
    """Count every adjacent symbol pair, weighted by word frequency."""
    pairs = Counter()
    for symbols, freq in word_counts.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(word_counts, pair):
    """Replace every occurrence of `pair` with the merged symbol."""
    merged_symbol = pair[0] + pair[1]
    new_counts = {}
    for symbols, freq in word_counts.items():
        out = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(merged_symbol)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_counts[tuple(out)] = new_counts.get(tuple(out), 0) + freq
    return new_counts


def train_bpe(text, num_merges):
    """The whole BPE algorithm: repeatedly merge the most frequent adjacent pair."""
    word_counts = get_word_counts(text)
    merges = []
    for step in range(num_merges):
        pairs = count_pairs(word_counts)
        if not pairs:
            break
        best, freq = pairs.most_common(1)[0]
        if freq < 2:
            # Nothing repeats any more; further merges would just memorise.
            break
        merges.append((best, freq))
        word_counts = merge_pair(word_counts, best)
    return merges


def build_vocab(text, merges):
    """Assign a stable integer ID to every symbol the tokenizer can emit."""
    symbols = set()
    for word in text.lower().split():
        symbols.update(word)
        symbols.add("</w>")
    for (a, b), _ in merges:
        symbols.add(a + b)
    # Sort for determinism: shorter symbols first, then alphabetical.
    ordered = sorted(symbols, key=lambda s: (len(s), s))
    return {sym: i for i, sym in enumerate(ordered)}


def tokenize(word, merges):
    """Apply the learned merges to one word, in the order they were learned."""
    symbols = list(word) + ["</w>"]
    for (a, b), _ in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i:i + 2] = [a + b]
            else:
                i += 1
    return symbols


def encode(text, merges, vocab):
    """Tokenize a full string and map to IDs. Unknown symbols are reported."""
    tokens, ids = [], []
    for word in text.lower().split():
        for sym in tokenize(word, merges):
            tokens.append(sym)
            ids.append(vocab.get(sym, -1))
    return tokens, ids


if __name__ == "__main__":
    NUM_MERGES = 120

    merges = train_bpe(CORPUS, NUM_MERGES)
    vocab = build_vocab(CORPUS, merges)

    print("=" * 72)
    print("BPE TRAINING")
    print("=" * 72)
    print(f"Corpus words:      {len(CORPUS.split())}")
    print(f"Merges requested:  {NUM_MERGES}")
    print(f"Merges learned:    {len(merges)}")
    print(f"Final vocab size:  {len(vocab)}")

    print("\nFirst 15 merges learned, in order:")
    for rank, ((a, b), freq) in enumerate(merges[:15], start=1):
        print(f"  {rank:>2}. {a!r} + {b!r}  ->  {a + b!r}   (seen {freq}x)")

    print("\nNotice the progression: single characters merge into fragments,")
    print("fragments merge into common whole words. Nobody specified this.")

    print()
    print("=" * 72)
    print("TOKENIZING THE PROMPT")
    print("=" * 72)
    print(f"Prompt: {PROMPT!r}\n")

    tokens, ids = encode(PROMPT, merges, vocab)

    print("Token -> ID")
    for tok, tid in zip(tokens, ids):
        flag = "   <- not in vocab" if tid == -1 else ""
        print(f"  {tok!r:<20} {tid}{flag}")

    print(f"\nIDs: {ids}")
    print(f"\nTokens: {len(tokens)}")
    print(f"Words:  {len(PROMPT.split())}")
    print(f"Ratio:  {len(tokens) / len(PROMPT.split()):.2f} tokens per word")

    print()
    print("=" * 72)
    print("WHY SOME WORDS SPLIT AND OTHERS DO NOT")
    print("=" * 72)
    for word in PROMPT.lower().split():
        pieces = tokenize(word, merges)
        n = len(pieces)
        corpus_freq = CORPUS.lower().split().count(word)
        status = "single token" if n == 1 else f"{n} pieces"
        print(f"  {word!r:<14} {status:<14} appears {corpus_freq}x in corpus")

    print("\nThis is the whole point of subword tokenization: frequent words")
    print("compress to one token, rare words decompose into fragments. The")
    print("tokenizer never fails on unseen input, it just uses more tokens.")
