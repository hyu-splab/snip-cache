import re
import string
from functools import lru_cache

import numpy as np


class SentenceComparer:
    _shared_model = None

    def __init__(self, model_name=None):
        if model_name is not None:
            from sentence_transformers import SentenceTransformer

            if SentenceComparer._shared_model is None:
                SentenceComparer._shared_model = SentenceTransformer(model_name)
        self.sbert = SentenceComparer._shared_model
        self.articles = ["the", "a", "an"]
        adverbs = [
            "just",
            "really",
            "actually",
            "basically",
            "literally",
            "seriously",
            "very",
        ]
        fillers = [
            "uh",
            "um",
            "hmm",
            "you know",
            "well",
            "okay",
            "right",
            "alright",
        ]
        vague_expressions = ["kind of", "sort of"]
        self.stop_words = set(self.articles + adverbs + fillers + vague_expressions)
        self.special_characters = list(string.punctuation)
        self.sbert = None

    def remove_custom_stopwords(
        self, text: str, stop_words: list[str] = None, special_characters=None  # type: ignore
    ) -> str:
        """
        Remove punctuation and stopwords from text.
        Maintains backward compatibility with previous implementation.
        """
        if stop_words is None:
            stop_words = self.stop_words
        if special_characters:
            for ch in special_characters:
                text = text.replace(ch, " ")
        words = [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in stop_words]
        return " ".join(words)

    #########################################################################
    # Tokenization and preprocessing
    #########################################################################
    def _tokenize(self, text: str):
        """Tokenize a sentence using regex (remove punctuation)."""
        return re.findall(r"\b\w+\b", text.lower())

    def preprocess_sentences(self, s1: str, s2: str) -> tuple[str, str]:
        """Remove common words between two sentences."""
        words1 = set(self._tokenize(s1))
        words2 = set(self._tokenize(s2))

        unique1 = words1 - words2
        unique2 = words2 - words1

        s1_filtered = " ".join(unique1) if unique1 else s1
        s2_filtered = " ".join(unique2) if unique2 else s2
        return s1_filtered, s2_filtered

    #########################################################################
    # Lexical similarities
    #########################################################################
    def jaccard_similarity(self, s1: str, s2: str) -> float:
        set1 = set(self._tokenize(s1))
        set2 = set(self._tokenize(s2))
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def subset_similarity(self, s1: str, s2: str) -> float:
        """Return 1.0 if one sentence's tokens are a subset of the other's."""
        set1 = set(self._tokenize(s1))
        set2 = set(self._tokenize(s2))
        return 1.0 if set1.issubset(set2) or set2.issubset(set1) else 0.0

    #########################################################################
    # Embedding-based similarities
    #########################################################################
    @lru_cache(maxsize=256)
    def _get_sentence_embedding(self, sentence: str):
        if self.sbert is None:
            return 0
        else:
            try:
                if not sentence.strip():
                    dim = self.sbert.get_sentence_embedding_dimension()
                    return np.zeros(dim)  # type: ignore
                return self.sbert.encode(sentence, show_progress_bar=False)
            except Exception as e:
                print(f"[EmbeddingError] {e}")
                dim = self.sbert.get_sentence_embedding_dimension()
                return np.zeros(dim)  # type: ignore

    def _compute_embedding_similarity(self, emb1, emb2) -> float:
        """(3) Fast numpy cosine similarity without sklearn overhead."""
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if denom == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / denom)

    def embedding_similarity(self, s1: str, s2: str) -> float:
        emb1 = self._get_sentence_embedding(s1)
        emb2 = self._get_sentence_embedding(s2)
        return self._compute_embedding_similarity(emb1, emb2)
        return 0.9

    #########################################################################
    # Hybrid semantic similarity
    #########################################################################
    def semantic_similarity(self, s1: str, s2: str, base_alpha: float = 0.3) -> float:
        """
        Combine lexical and embedding similarity with adaptive weighting.
        (5) Adjusts alpha dynamically by sentence length ratio.
        """
        lex_sim = max(self.jaccard_similarity(s1, s2), self.subset_similarity(s1, s2))
        emb_sim = self.embedding_similarity(s1, s2)
        len_ratio = min(len(s1), len(s2)) / max(len(s1), len(s2), 1)
        alpha = base_alpha * len_ratio
        return alpha * lex_sim + (1 - alpha) * emb_sim

    #########################################################################
    # Containment similarity (trigger detection)
    #########################################################################
    def _normalize_verb(self, word: str) -> str:
        w = word.lower()
        # -ed
        if w.endswith("ed") and len(w) > 3:
            return w[:-2]
        # -s but not -ss
        if w.endswith("s") and len(w) > 2 and not w.endswith("ss"):
            return w[:-1]
        return w

    def contain_similarity(self, full_sentence: str, sub_phrase: str) -> float:
        # Remove noise words
        full_sentence = self.remove_custom_stopwords(full_sentence, self.articles)
        sub_phrase = self.remove_custom_stopwords(sub_phrase, self.articles)

        # ========== 1. Word-boundary exact match (safe path) ==========
        # Prevent activate ⊄ deactivate type errors
        import re

        pattern = r"\b{}\b".format(re.escape(sub_phrase))
        if re.search(pattern, full_sentence):
            return 1.0

        # ========== 2. Sequential token matching (with verb normalization) ==========
        full_words = self._tokenize(full_sentence)
        sub_words = self._tokenize(sub_phrase)

        if not sub_words:
            return 0.0

        # Apply verb normalization
        full_words = [self._normalize_verb(w) for w in full_words]
        sub_words = [self._normalize_verb(w) for w in sub_words]

        # Sequential order check
        j = 0
        for word in full_words:
            if j < len(sub_words) and word == sub_words[j]:
                j += 1
                if j == len(sub_words):
                    return 1.0
        return 0.0
