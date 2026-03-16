"""Tests for sparse encoder: TokenEncoder and BM25SparseEncoder."""

from src.rag.sparse import tokenize, TokenEncoder, BM25SparseEncoder, SparseVector


class TestTokenize:
    def test_basic_tokenization(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert tokenize("hello, world! foo-bar.") == ["hello", "world", "foo", "bar"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_preserved(self):
        assert tokenize("version 3 release") == ["version", "3", "release"]


class TestTokenEncoder:
    def test_encode_tf_builds_vocab(self):
        enc = TokenEncoder()
        sv = enc.encode_tf("the quick brown fox")
        assert isinstance(sv, SparseVector)
        assert len(sv.indices) == 4
        assert all(v == 1.0 for v in sv.values)
        assert len(enc.vocab) == 4

    def test_encode_tf_repeated_tokens(self):
        enc = TokenEncoder()
        sv = enc.encode_tf("the the the fox")
        # "the" appears 3 times, "fox" 1 time
        assert len(sv.indices) == 2
        idx_the = enc.vocab["the"]
        pos = sv.indices.index(idx_the)
        assert sv.values[pos] == 3.0

    def test_vocab_is_append_only(self):
        enc = TokenEncoder()
        enc.encode_tf("alpha beta")
        first_vocab = dict(enc.vocab)
        enc.encode_tf("beta gamma")
        # alpha and beta keep their original IDs
        assert enc.vocab["alpha"] == first_vocab["alpha"]
        assert enc.vocab["beta"] == first_vocab["beta"]
        assert "gamma" in enc.vocab

    def test_encode_query_tf_ignores_unknown_tokens(self):
        enc = TokenEncoder(vocab={"hello": 0, "world": 1})
        sv = enc.encode_query_tf("hello unknown world")
        assert len(sv.indices) == 2
        assert 0 in sv.indices
        assert 1 in sv.indices

    def test_encode_query_tf_empty_for_all_unknown(self):
        enc = TokenEncoder(vocab={"hello": 0})
        sv = enc.encode_query_tf("unknown tokens only")
        assert sv.indices == []
        assert sv.values == []

    def test_init_from_existing_vocab(self):
        vocab = {"alpha": 0, "beta": 1, "gamma": 2}
        enc = TokenEncoder(vocab=vocab)
        assert enc.vocab == vocab
        # New tokens get IDs after the max existing
        enc.encode_tf("delta")
        assert enc.vocab["delta"] == 3

    def test_vocab_round_trip(self):
        enc1 = TokenEncoder()
        enc1.encode_tf("machine learning models")
        vocab = enc1.vocab

        enc2 = TokenEncoder(vocab=vocab)
        sv1 = enc1.encode_query_tf("machine learning")
        sv2 = enc2.encode_query_tf("machine learning")
        assert sv1.indices == sv2.indices
        assert sv1.values == sv2.values


class TestBM25SparseEncoder:
    def test_fit_and_get_scores(self):
        enc = BM25SparseEncoder()
        enc.fit(["the quick brown fox", "lazy dog sleeps", "quick fox jumps"])
        scores = enc.get_scores("quick fox")
        assert len(scores) == 3
        # "quick fox" should score highest for docs containing those terms
        assert scores[0] > 0 or scores[2] > 0

    def test_get_scores_empty_corpus(self):
        enc = BM25SparseEncoder()
        assert enc.get_scores("hello") == []
