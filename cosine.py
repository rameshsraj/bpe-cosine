import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROMPT = "What is our retention period for Slack messages?"

CORPUS = [
    "Slack message retention is set to three years for all workspaces. "
    "Legal hold overrides retention and suspends automatic deletion until released.",

    "Email archives are retained for seven years in line with regulatory obligations.",

    "Employees may request deletion of personal data via the privacy portal.",

    "Slack workspace admins can configure channel-level notification defaults.",

    "Travel expense claims must be submitted within thirty days of the trip.",

    "The office cafeteria is open from eight in the morning until six in the evening.",
]

LABELS = [
    "slack-retention-policy",
    "email-retention-policy",
    "privacy-deletion-requests",
    "slack-notification-settings",
    "travel-expense-policy",
    "cafeteria-hours",
]


def rank(query, corpus, labels, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
    else:
        matrix = vectorizer.transform(corpus)
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix)[0]
    order = np.argsort(scores)[::-1]
    return scores, order, vectorizer, q


if __name__ == "__main__":
    print("=" * 72)
    print("PART 1 -- THE VECTOR SPACE")
    print("=" * 72)

    scores, order, vec, qvec = rank(PROMPT, CORPUS, LABELS)

    print(f"Query: {PROMPT!r}")
    print(f"Vocabulary (vector dimensionality): {len(vec.vocabulary_)}")
    print(f"Query vector non-zero components: {qvec.nnz}")

    names = vec.get_feature_names_out()
    nz = qvec.toarray()[0].nonzero()[0]
    print("\nThe query as a vector (non-zero dimensions only):")
    for i in nz:
        print(f"  {names[i]:<16} {qvec.toarray()[0][i]:.4f}")
    print("\nEvery other dimension is 0.0. Real embedding vectors are dense --")
    print("every dimension carries a value -- but the geometry works the same way.")

    print()
    print("=" * 72)
    print("PART 2 -- RANKED RETRIEVAL (real cosine similarity)")
    print("=" * 72)
    for r, i in enumerate(order, start=1):
        marker = "  <-- top hit" if r == 1 else ""
        print(f"  {r}. {scores[i]:.4f}  {LABELS[i]}{marker}")

    print("\nThe intended document ranks first. Note that several documents score")
    print("exactly 0.0000 -- they share no vocabulary with the query at all, so")
    print("they sit at 90 degrees to it in this space.")

    print()
    print("=" * 72)
    print("PART 3 -- WHERE LEXICAL MATCHING BREAKS")
    print("=" * 72)
    print("Same corpus. A paraphrased query that shares almost no words with the")
    print("document that actually answers it:\n")

    PARAPHRASE = "How long do we keep chat history before it is wiped?"
    p_scores, p_order, _, _ = rank(PARAPHRASE, CORPUS, LABELS, vectorizer=vec)

    print(f"Query: {PARAPHRASE!r}\n")
    for r, i in enumerate(p_order, start=1):
        print(f"  {r}. {p_scores[i]:.4f}  {LABELS[i]}")

    if p_scores.max() == 0.0:
        print("\nEvery score is 0.0000. Total retrieval failure.")
    else:
        print(f"\nTop score collapsed to {p_scores.max():.4f}.")

    print("\nA human reads that question and instantly knows document 1 answers it.")
    print("TF-IDF cannot, because 'how long' and 'three years' share no tokens,")
    print("and neither do 'chat history' and 'message retention'.")
    print("\nThis is precisely the gap learned embedding models fill. They place")
    print("paraphrases near each other in vector space because they were trained")
    print("on text where those phrasings occur in similar contexts. The retrieval")
    print("mechanism is unchanged -- only the quality of the vectors improves.")
