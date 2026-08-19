# Retrieval concepts

Hybrid retrieval combines a dense retriever with a sparse retriever. The dense
retriever embeds the query and passages into vectors and compares them with cosine
similarity, capturing semantic meaning. The sparse retriever uses BM25, a keyword
ranking function, which is strong on exact terms, names, and rare tokens.

Reciprocal Rank Fusion (RRF) merges the two ranked lists without needing their scores
to be on the same scale. For each document it sums 1 / (k + rank) across the lists,
where k is a constant such as 60. Documents that rank high in either retriever rise to
the top of the fused list. RRF is simple, robust, and a strong default for hybrid search.

Re-ranking is the accuracy step that runs after fusion. A cross-encoder takes the query
and a candidate passage together and scores their relevance jointly, which is more
accurate than comparing independent embeddings. Because cross-encoders are expensive,
they run only on the shortlist of fused candidates, not the whole corpus.

Chunking splits documents into passages small enough to retrieve precisely. Sentence-aware
chunking with a small overlap keeps ideas intact across chunk boundaries. Deduplication
removes exact repeats by content hash and near-duplicates by embedding cosine similarity,
which keeps the index clean and avoids wasting the context window on redundant passages.
