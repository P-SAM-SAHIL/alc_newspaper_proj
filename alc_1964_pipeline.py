"""
English A La Carte (ALC) embedding pipeline for the 1964 historical newspaper data.

Input CSV columns expected:
article_id,date,newspaper_name,headline,article,LCCN,State,County,City

Example:
python alc_1964_pipeline.py ^
  --csv cleaned_1964.csv ^
  --fasttext cc.en.300.bin ^
  --dict-json dictionaries.json ^
  --group-col State ^
  --groups Mississippi "New York" California ^
  --out-dir outputs_1964 ^
  --window-size 5 ^
  --min-count 20

Dictionary JSON format:
{
  "BLACK": ["black", "negro", "..."],
  "WHITE": ["white", "..."],
  "RICH": ["rich", "wealthy", "elite"],
  "POOR": ["poor", "..."],
  "MEN": ["men", "man", "..."],
  "WOMEN": ["women", "woman", "..."],
  "POSITIVE": ["good", "..."],
  "NEGATIVE": ["bad", "..."]
}

If --dict-json is omitted, the script uses the built-in dictionaries from the
project prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import string
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm


TextRows = Iterator[Tuple[str, Optional[str]]]


BUILTIN_DICTIONARIES: Dict[str, List[str]] = {
    "BLACK": [
        "black", "african", "africans", "blacks", "colored", "coloreds", "negro", "negros",
        "n*er", "n*ers", "ni*a", "ni*as", "nigger", "niggers", "nigga", "niggas", "afro", "afros",
    ],
    "WHITE": [
        "white", "whites", "european", "europeans", "british", "english", "american",
        "americans", "caucasian", "caucasians", "englishman", "englishmen", "englishwoman",
        "englishwomen",
    ],
    "RICH": [
        "rich", "affluent", "wealthy", "moneyed", "wealth", "aristocrat", "aristocrats",
        "aristocracy", "prosperous", "privileged", "bourgeoisie", "bourgeois", "noble",
        "nobles", "nobility", "nobleman", "noblemen", "elite", "elites", "benefactor",
        "benefactors", "philanthropist", "philanthropists", "millionaire", "millionaires",
    ],
    "POOR": [
        "poor", "beggar", "beggars", "needy", "wretch", "wretches", "impoverished",
        "destitute", "penniless", "unaffluent", "underprivileged",
    ],
    "MEN": [
        "men", "man", "male", "males", "he", "him", "his", "himself", "mr", "mister",
        "boy", "boys", "guy", "guys", "fella", "fellas", "gent", "gents", "sir", "sirs",
        "bloke", "blokes", "gentleman", "gentlemen", "lad", "lads", "prince", "princes",
        "king", "kings",
    ],
    "WOMEN": [
        "women", "woman", "female", "females", "she", "her", "hers", "herself", "mrs",
        "ms", "girl", "girls", "gal", "gals", "lady", "ladies", "dame", "dames", "miss",
        "missus", "bride", "brides", "maiden", "maidens", "gentlewoman", "gentlewomen",
        "lass", "lassie", "madam", "princess", "princesses", "queen", "queens",
    ],
    "POSITIVE": [
        "good", "excellent", "pleasant", "wonderful", "honest", "kind", "friendly",
        "virtuous", "decent", "noble", "cheerful", "trustworthy",
    ],
    "NEGATIVE": [
        "bad", "horrible", "unpleasant", "horrid", "dishonest", "cruel", "hostile",
        "vicious", "mean", "nasty", "vile", "cowardly",
    ],
}


MASKED_TERM_EXPANSIONS = {
    "n*er": "nigger",
    "n*ers": "niggers",
    "ni*a": "nigga",
    "ni*as": "niggas",
}


BIAS_TESTS: List[Tuple[str, str, str, str]] = [
    ("BLACK_NEGATIVE_MINUS_WHITE_NEGATIVE", "BLACK", "WHITE", "NEGATIVE"),
    ("BLACK_POSITIVE_MINUS_WHITE_POSITIVE", "BLACK", "WHITE", "POSITIVE"),
    ("BLACK_RICH_MINUS_WHITE_RICH", "BLACK", "WHITE", "RICH"),
    ("BLACK_POOR_MINUS_WHITE_POOR", "BLACK", "WHITE", "POOR"),
    ("WOMEN_POSITIVE_MINUS_MEN_POSITIVE", "WOMEN", "MEN", "POSITIVE"),
    ("WOMEN_NEGATIVE_MINUS_MEN_NEGATIVE", "WOMEN", "MEN", "NEGATIVE"),
    ("POOR_NEGATIVE_MINUS_RICH_NEGATIVE", "POOR", "RICH", "NEGATIVE"),
    ("RICH_POSITIVE_MINUS_POOR_POSITIVE", "RICH", "POOR", "POSITIVE"),
]


def load_nltk_helpers():
    """
    Use NLTK for English stopwords, tokenization, and edit distance.
    """
    from nltk.corpus import stopwords
    from nltk.metrics.distance import edit_distance
    from nltk.tokenize import RegexpTokenizer

    try:
        words = set(stopwords.words("english"))
    except LookupError:
        import nltk

        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("english"))
    return words, RegexpTokenizer(r"[A-Za-z0-9]+"), edit_distance


def load_fasttext_model(path: Path, limit: Optional[int] = None):
    """
    Load a fastText model.

    Prefer .bin for true fastText subword OOV behavior. .vec also works, but OOV
    subword inference is not available from plain vector text files.
    """
    suffix = path.suffix.lower()
    if suffix == ".bin":
        from gensim.models.fasttext import load_facebook_vectors

        return load_facebook_vectors(str(path))
    if suffix in {".vec", ".txt"}:
        from gensim.models import KeyedVectors

        return KeyedVectors.load_word2vec_format(str(path), binary=False, limit=limit)
    if suffix == ".kv":
        from gensim.models import KeyedVectors

        return KeyedVectors.load(str(path), mmap="r")
    raise ValueError(f"Unsupported fastText file type: {path.suffix}. Use .bin, .vec, .txt, or .kv.")


class EnglishNormalizer:
    """
    Lowercase, remove punctuation, remove stop words, and optionally fuzzy-correct
    OCR-damaged core dictionary words.
    """

    OCR_DIGITS = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "7": "t"})

    def __init__(
        self,
        extra_stopwords: Optional[Set[str]] = None,
        fuzzy_terms: Optional[Set[str]] = None,
        protected_terms: Optional[Set[str]] = None,
        fuzzy_max_distance: int = 1,
        enable_fuzzy: bool = False,
    ) -> None:
        nltk_stopwords, tokenizer, edit_distance_fn = load_nltk_helpers()
        self.tokenizer = tokenizer
        self.edit_distance = edit_distance_fn
        self.stop_words = set(nltk_stopwords)
        self.stop_words.update(string.ascii_lowercase)
        if extra_stopwords:
            self.stop_words.update(w.lower() for w in extra_stopwords)
        self.protected_terms = {w.lower() for w in (protected_terms or set())}

        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_max_distance = fuzzy_max_distance
        self.fuzzy_terms = {w.lower() for w in (fuzzy_terms or set()) if w and w.isalpha()}
        self.fuzzy_by_len: Dict[int, List[str]] = defaultdict(list)
        for term in self.fuzzy_terms:
            self.fuzzy_by_len[len(term)].append(term)

    def tokenize(self, text: object) -> List[str]:
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return []

        raw_tokens = self.tokenizer.tokenize(str(text).lower())
        tokens: List[str] = []
        for raw in raw_tokens:
            token = self._repair_ocr_digits(raw)
            if not token.isalpha():
                continue
            if token in self.stop_words and token not in self.protected_terms:
                continue
            if self.enable_fuzzy:
                token = self._fuzzy_repair(token)
            if token and (token not in self.stop_words or token in self.protected_terms):
                tokens.append(token)
        return tokens

    @staticmethod
    def _has_alpha_and_digit(token: str) -> bool:
        return any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token)

    def _repair_ocr_digits(self, token: str) -> str:
        if self._has_alpha_and_digit(token):
            return token.translate(self.OCR_DIGITS)
        return token

    @lru_cache(maxsize=100_000)
    def _fuzzy_repair(self, token: str) -> str:
        if token in self.fuzzy_terms or len(token) < 4:
            return token

        best_term = token
        best_dist = self.fuzzy_max_distance + 1
        for length in range(len(token) - self.fuzzy_max_distance, len(token) + self.fuzzy_max_distance + 1):
            for candidate in self.fuzzy_by_len.get(length, []):
                if candidate[0] != token[0]:
                    continue
                try:
                    dist = self.edit_distance(token, candidate, substitution_cost=1, transpositions=True)
                except TypeError:
                    dist = self.edit_distance(token, candidate)
                if dist < best_dist:
                    best_term = candidate
                    best_dist = dist
                    if dist == 1:
                        return best_term
        return best_term if best_dist <= self.fuzzy_max_distance else token


def read_text_rows(
    csv_path: Path,
    article_col: str,
    group_col: Optional[str],
    chunksize: int,
    groups: Optional[Set[str]] = None,
) -> TextRows:
    del chunksize  # Kept in the CLI for compatibility with earlier pandas-based runs.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or article_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain article column '{article_col}'.")
        if group_col and group_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain group column '{group_col}'.")

        for row in reader:
            group = str(row.get(group_col, "")) if group_col else None
            if groups and group not in groups:
                continue
            yield row.get(article_col, ""), group


def iter_windows(tokens: Sequence[str], window_size: int) -> Iterator[Tuple[str, List[str]]]:
    for center_idx, target in enumerate(tokens):
        left = max(0, center_idx - window_size)
        right = min(len(tokens), center_idx + window_size + 1)
        context = [tokens[i] for i in range(left, right) if i != center_idx]
        if context:
            yield target, context


def l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def fit_weighted_linear_map(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Solve min_A sum_i weights_i || y_i - A x_i ||^2.

    With row-major matrices this fits y ~= X @ B, then returns A = B.T so
    transformed vectors are computed as context @ A.T.
    """
    sqrt_w = np.sqrt(weights).reshape(-1, 1).astype(np.float32)
    Xw = X * sqrt_w
    yw = y * sqrt_w
    B, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return B.T.astype(np.float32)


def model_has_index(model, word: str) -> bool:
    if hasattr(model, "has_index_for"):
        return bool(model.has_index_for(word))
    if hasattr(model, "key_to_index"):
        return word in model.key_to_index
    if hasattr(model, "vocab"):
        return word in model.vocab
    return word in model


class CachedVectorLookup:
    def __init__(self, model, vector_size: int, cache_size: int = 100_000) -> None:
        self.model = model
        self.vector_size = vector_size
        self.zero = np.zeros(vector_size, dtype=np.float32)
        self.get = lru_cache(maxsize=cache_size)(self._get_uncached)

    def _get_uncached(self, word: str) -> np.ndarray:
        try:
            return np.asarray(self.model.get_vector(word), dtype=np.float32)
        except KeyError:
            return self.zero


@dataclass
class GlobalALCResult:
    matrix: np.ndarray
    vocab: List[str]
    counts: Dict[str, int]
    context_vectors: np.ndarray
    alc_vectors: np.ndarray


class ALaCarteEnglish:
    def __init__(
        self,
        model,
        normalizer: EnglishNormalizer,
        window_size: int = 5,
        min_count: int = 20,
        alpha_k: float = 100.0,
        vector_cache_size: int = 100_000,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.window_size = window_size
        self.min_count = min_count
        self.alpha_k = alpha_k
        self.embedding_dim = int(model.vector_size)
        self.lookup = CachedVectorLookup(model, self.embedding_dim, vector_cache_size)
        self.A: Optional[np.ndarray] = None
        self.global_result: Optional[GlobalALCResult] = None

    def count_tokens(
        self,
        csv_path: Path,
        article_col: str,
        chunksize: int,
    ) -> Counter:
        counts: Counter = Counter()
        rows = read_text_rows(csv_path, article_col, group_col=None, chunksize=chunksize)
        for article, _ in tqdm(rows, desc="Phase 3A - counting global tokens"):
            counts.update(self.normalizer.tokenize(article))
        return counts

    def build_context_sums(
        self,
        csv_path: Path,
        article_col: str,
        eligible_words: Set[str],
        chunksize: int,
    ) -> Tuple[Dict[str, np.ndarray], Counter]:
        context_sums: Dict[str, np.ndarray] = {}
        target_counts: Counter = Counter()
        rows = read_text_rows(csv_path, article_col, group_col=None, chunksize=chunksize)

        for article, _ in tqdm(rows, desc="Phase 3B - building global contexts"):
            tokens = self.normalizer.tokenize(article)
            if len(tokens) < 2:
                continue
            for target, context in iter_windows(tokens, self.window_size):
                if target not in eligible_words:
                    continue
                context_vec = np.zeros(self.embedding_dim, dtype=np.float32)
                used = 0
                for token in context:
                    vec = self.lookup.get(token)
                    if vec is not self.lookup.zero:
                        context_vec += vec
                        used += 1
                if used == 0:
                    continue
                if target not in context_sums:
                    context_sums[target] = np.zeros(self.embedding_dim, dtype=np.float32)
                context_sums[target] += context_vec / used
                target_counts[target] += 1
        return context_sums, target_counts

    def fit_global_matrix(
        self,
        csv_path: Path,
        article_col: str,
        chunksize: int = 20_000,
        max_regression_words: Optional[int] = 100_000,
        max_global_alc_words: Optional[int] = 200_000,
    ) -> GlobalALCResult:
        token_counts = self.count_tokens(csv_path, article_col, chunksize)
        eligible = {word for word, count in token_counts.items() if count >= self.min_count}
        if max_global_alc_words:
            eligible = {
                word
                for word, _ in token_counts.most_common(max_global_alc_words)
                if token_counts[word] >= self.min_count
            }

        context_sums, context_counts = self.build_context_sums(csv_path, article_col, eligible, chunksize)

        regression_vocab = [
            word
            for word, _ in token_counts.most_common()
            if word in context_sums and model_has_index(self.model, word)
        ]
        if max_regression_words:
            regression_vocab = regression_vocab[:max_regression_words]
        if len(regression_vocab) < self.embedding_dim:
            raise ValueError(
                f"Only {len(regression_vocab)} regression words available for {self.embedding_dim} dimensions. "
                "Lower --min-count or increase corpus size."
            )

        X = np.vstack([context_sums[word] / context_counts[word] for word in regression_vocab]).astype(np.float32)
        y = np.vstack([self.model.get_vector(word) for word in regression_vocab]).astype(np.float32)
        weights = np.asarray(
            [min(1.0, float(context_counts[word]) / self.alpha_k) for word in regression_vocab],
            dtype=np.float32,
        )

        print(f"Fitting A on {len(regression_vocab):,} words with dimension {self.embedding_dim}.")
        self.A = fit_weighted_linear_map(X, y, weights)

        global_vocab = [word for word, _ in token_counts.most_common() if word in context_sums]
        if max_global_alc_words:
            global_vocab = global_vocab[:max_global_alc_words]
        C = np.vstack([context_sums[word] / context_counts[word] for word in global_vocab]).astype(np.float32)
        alc = l2_normalize_matrix(C @ self.A.T).astype(np.float32)

        result = GlobalALCResult(
            matrix=self.A,
            vocab=global_vocab,
            counts={word: int(context_counts[word]) for word in global_vocab},
            context_vectors=C,
            alc_vectors=alc,
        )
        self.global_result = result
        return result

    def load_matrix(self, matrix_path: Path) -> None:
        self.A = np.load(matrix_path).astype(np.float32)

    def local_context_vectors(
        self,
        csv_path: Path,
        article_col: str,
        group_col: str,
        groups: Sequence[str],
        target_words: Set[str],
        chunksize: int = 20_000,
    ) -> Dict[str, Dict[str, Tuple[np.ndarray, int]]]:
        if self.A is None:
            raise ValueError("Matrix A is not loaded or fitted.")

        group_set = set(groups)
        sums: Dict[str, Dict[str, np.ndarray]] = {g: {} for g in groups}
        counts: Dict[str, Counter] = {g: Counter() for g in groups}

        rows = read_text_rows(csv_path, article_col, group_col, chunksize, group_set)
        for article, group in tqdm(rows, desc=f"Phase 4 - local contexts by {group_col}"):
            if group not in group_set:
                continue
            tokens = self.normalizer.tokenize(article)
            for target, context in iter_windows(tokens, self.window_size):
                if target not in target_words:
                    continue
                context_vec = np.zeros(self.embedding_dim, dtype=np.float32)
                used = 0
                for token in context:
                    vec = self.lookup.get(token)
                    if vec is not self.lookup.zero:
                        context_vec += vec
                        used += 1
                if used == 0:
                    continue
                if target not in sums[group]:
                    sums[group][target] = np.zeros(self.embedding_dim, dtype=np.float32)
                sums[group][target] += context_vec / used
                counts[group][target] += 1

        output: Dict[str, Dict[str, Tuple[np.ndarray, int]]] = {}
        for group in groups:
            output[group] = {}
            for word, vec_sum in sums[group].items():
                n = int(counts[group][word])
                if n > 0:
                    output[group][word] = (vec_sum / n, n)
        return output

    def transform_context(self, context_vector: np.ndarray) -> np.ndarray:
        if self.A is None:
            raise ValueError("Matrix A is not loaded or fitted.")
        vec = context_vector.reshape(1, -1).astype(np.float32) @ self.A.T
        return l2_normalize_matrix(vec)[0].astype(np.float32)

    def nearest_neighbors(
        self,
        query_vector: np.ndarray,
        topn: int = 10,
        exclude: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        if self.global_result is None:
            raise ValueError("Global ALC vectors were not built; nearest-neighbor validation is unavailable.")
        exclude = exclude or set()
        q = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        sims = self.global_result.alc_vectors @ q
        if exclude:
            for word in exclude:
                try:
                    idx = self.global_result.vocab.index(word)
                    sims[idx] = -np.inf
                except ValueError:
                    pass
        top_idx = np.argpartition(-sims, kth=min(topn, len(sims) - 1))[:topn]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self.global_result.vocab[i], float(sims[i])) for i in top_idx]


def expand_masked_terms(terms: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for term in terms:
        lowered = str(term).strip().lower()
        if not lowered:
            continue
        expanded.append(lowered)
        if lowered in MASKED_TERM_EXPANSIONS:
            expanded.append(MASKED_TERM_EXPANSIONS[lowered])
    return expanded


def clean_dictionary_terms(
    raw: Mapping[str, Sequence[str]],
    normalizer: EnglishNormalizer,
) -> Dict[str, List[str]]:
    dictionaries: Dict[str, List[str]] = {}
    for concept, terms in raw.items():
        cleaned_terms: List[str] = []
        for term in expand_masked_terms(terms):
            toks = normalizer.tokenize(term)
            if len(toks) == 1:
                cleaned_terms.append(toks[0])
        dictionaries[concept.upper()] = sorted(set(cleaned_terms))
    return dictionaries


def load_dictionary_json(path: Path, normalizer: EnglishNormalizer) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return clean_dictionary_terms(raw, normalizer)


def flatten_dictionary_terms(dictionaries: Mapping[str, Sequence[str]]) -> Set[str]:
    return {term for terms in dictionaries.values() for term in terms}


def raw_dictionary_terms(raw: Mapping[str, Sequence[str]]) -> Set[str]:
    terms: Set[str] = set()
    for concept_terms in raw.values():
        terms.update(expand_masked_terms(concept_terms))
    return terms


def save_matrix_and_metadata(out_dir: Path, result: GlobalALCResult, args: argparse.Namespace) -> None:
    np.save(out_dir / "A_1964_news.npy", result.matrix)
    with (out_dir / "A_1964_news_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_dim": int(result.matrix.shape[0]),
                "window_size": args.window_size,
                "min_count": args.min_count,
                "alpha_k": args.alpha_k,
                "global_alc_vocab_size": len(result.vocab),
                "fasttext_model": str(args.fasttext),
                "csv": str(args.csv),
            },
            f,
            indent=2,
        )


def save_global_alc_vectors(out_dir: Path, result: GlobalALCResult) -> None:
    np.savez_compressed(
        out_dir / "global_alc_vectors_1964.npz",
        vocab=np.asarray(result.vocab),
        counts=np.asarray([result.counts[w] for w in result.vocab], dtype=np.int32),
        vectors=result.alc_vectors,
    )


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_local_word_embeddings(
    out_dir: Path,
    pipeline: ALaCarteEnglish,
    local_contexts: Mapping[str, Mapping[str, Tuple[np.ndarray, int]]],
    dictionaries: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    concept_by_word = {}
    for concept, words in dictionaries.items():
        for word in words:
            concept_by_word[word] = concept

    transformed: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    rows: List[Dict[str, object]] = []
    for group, word_map in local_contexts.items():
        transformed[group] = defaultdict(dict)
        for word, (context_vec, count) in word_map.items():
            concept = concept_by_word.get(word, "UNKNOWN")
            alc_vec = pipeline.transform_context(context_vec)
            transformed[group][concept][word] = alc_vec
            row = {"group": group, "concept": concept, "word": word, "count": count}
            row.update({f"dim_{i}": float(v) for i, v in enumerate(alc_vec)})
            rows.append(row)

    write_csv_rows(out_dir / "localized_word_alc_embeddings.csv", rows)
    return transformed


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def build_and_save_concept_vectors(
    out_dir: Path,
    transformed: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> Dict[str, Dict[str, np.ndarray]]:
    concept_vectors: Dict[str, Dict[str, np.ndarray]] = {}
    vector_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []
    bias_rows: List[Dict[str, object]] = []

    for group, concept_map in transformed.items():
        concept_vectors[group] = {}
        for concept, word_vecs in concept_map.items():
            if not word_vecs:
                continue
            mat = np.vstack(list(word_vecs.values()))
            concept_vec = l2_normalize_vector(mat.mean(axis=0)).astype(np.float32)
            concept_vectors[group][concept] = concept_vec
            row = {"group": group, "concept": concept, "n_words_present": len(word_vecs)}
            row.update({f"dim_{i}": float(v) for i, v in enumerate(concept_vec)})
            vector_rows.append(row)

        concepts = sorted(concept_vectors[group])
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i + 1 :]:
                score_rows.append(
                    {
                        "group": group,
                        "concept_a": concept_a,
                        "concept_b": concept_b,
                        "cosine_similarity": cosine(
                            concept_vectors[group][concept_a],
                            concept_vectors[group][concept_b],
                        ),
                    }
                )

        for test_name, concept_a, concept_b, attribute in BIAS_TESTS:
            if (
                concept_a not in concept_vectors[group]
                or concept_b not in concept_vectors[group]
                or attribute not in concept_vectors[group]
            ):
                continue
            sim_a = cosine(concept_vectors[group][concept_a], concept_vectors[group][attribute])
            sim_b = cosine(concept_vectors[group][concept_b], concept_vectors[group][attribute])
            bias_rows.append(
                {
                    "group": group,
                    "bias_test": test_name,
                    "concept_a": concept_a,
                    "concept_b": concept_b,
                    "attribute": attribute,
                    "cosine_a_attribute": sim_a,
                    "cosine_b_attribute": sim_b,
                    "difference_a_minus_b": sim_a - sim_b,
                    "interpretation": (
                        f"positive means {concept_a} is closer to {attribute} than {concept_b}"
                    ),
                }
            )

    write_csv_rows(out_dir / "localized_concept_vectors.csv", vector_rows)
    write_csv_rows(out_dir / "concept_cosine_similarity_scores.csv", score_rows)
    write_csv_rows(out_dir / "bias_contrast_scores.csv", bias_rows)
    return concept_vectors


def save_validation_neighbors(
    out_dir: Path,
    pipeline: ALaCarteEnglish,
    transformed: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    validation_words: Sequence[str],
    topn: int,
) -> None:
    rows: List[Dict[str, object]] = []
    validation_set = set(validation_words)
    for group, concept_map in transformed.items():
        for concept, word_map in concept_map.items():
            for word, vec in word_map.items():
                if validation_set and word not in validation_set:
                    continue
                neighbors = pipeline.nearest_neighbors(vec, topn=topn, exclude={word})
                for rank, (neighbor, similarity) in enumerate(neighbors, start=1):
                    rows.append(
                        {
                            "group": group,
                            "concept": concept,
                            "query_word": word,
                            "rank": rank,
                            "neighbor": neighbor,
                            "cosine_similarity": similarity,
                        }
                    )
    if rows:
        write_csv_rows(out_dir / "validation_nearest_neighbors.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1964 English ALC embedding pipeline.")
    parser.add_argument("--csv", type=Path, required=True, help="Cleaned 1964 CSV file.")
    parser.add_argument("--fasttext", type=Path, required=True, help="English fastText .bin/.vec/.txt/.kv path.")
    parser.add_argument(
        "--dict-json",
        type=Path,
        default=None,
        help="Optional concept dictionary JSON file. Uses built-in project dictionaries if omitted.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("alc_outputs_1964"))
    parser.add_argument("--article-col", default="article")
    parser.add_argument("--group-col", default="State", choices=["State", "County", "City"])
    parser.add_argument("--groups", nargs="+", required=True, help="Geographic values to analyze.")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--chunksize", type=int, default=20_000)
    parser.add_argument("--alpha-k", type=float, default=100.0)
    parser.add_argument("--max-regression-words", type=int, default=100_000)
    parser.add_argument("--max-global-alc-words", type=int, default=200_000)
    parser.add_argument("--fasttext-limit", type=int, default=None, help="Only applies to .vec/.txt models.")
    parser.add_argument("--matrix", type=Path, default=None, help="Existing A_1964_news.npy to reuse.")
    parser.add_argument("--enable-fuzzy", action="store_true", help="Fuzzy-repair OCR variants of dictionary words.")
    parser.add_argument("--fuzzy-max-distance", type=int, default=1)
    parser.add_argument("--validation-words", nargs="*", default=[], help="Dictionary words for nearest-neighbor checks.")
    parser.add_argument("--neighbor-topn", type=int, default=10)
    parser.add_argument("--save-global-alc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dict_json:
        with args.dict_json.open("r", encoding="utf-8") as f:
            raw_dict = json.load(f)
    else:
        raw_dict = BUILTIN_DICTIONARIES
    raw_terms = raw_dictionary_terms(raw_dict)

    normalizer = EnglishNormalizer(
        fuzzy_terms=raw_terms,
        protected_terms=raw_terms,
        fuzzy_max_distance=args.fuzzy_max_distance,
        enable_fuzzy=args.enable_fuzzy,
    )
    dictionaries = (
        load_dictionary_json(args.dict_json, normalizer)
        if args.dict_json
        else clean_dictionary_terms(BUILTIN_DICTIONARIES, normalizer)
    )
    target_words = flatten_dictionary_terms(dictionaries)
    with (args.out_dir / "dictionaries_used.json").open("w", encoding="utf-8") as f:
        json.dump(dictionaries, f, indent=2)

    print("Phase 2 - loading base fastText model.")
    model = load_fasttext_model(args.fasttext, limit=args.fasttext_limit)

    pipeline = ALaCarteEnglish(
        model=model,
        normalizer=normalizer,
        window_size=args.window_size,
        min_count=args.min_count,
        alpha_k=args.alpha_k,
    )

    if args.matrix:
        print(f"Phase 3 - loading existing matrix: {args.matrix}")
        pipeline.load_matrix(args.matrix)
    else:
        print("Phase 3 - fitting custom 1964 news transformation matrix A.")
        result = pipeline.fit_global_matrix(
            csv_path=args.csv,
            article_col=args.article_col,
            chunksize=args.chunksize,
            max_regression_words=args.max_regression_words,
            max_global_alc_words=args.max_global_alc_words,
        )
        save_matrix_and_metadata(args.out_dir, result, args)
        if args.save_global_alc or args.validation_words:
            save_global_alc_vectors(args.out_dir, result)

    print("Phase 4 - generating localized ALC embeddings.")
    local_contexts = pipeline.local_context_vectors(
        csv_path=args.csv,
        article_col=args.article_col,
        group_col=args.group_col,
        groups=args.groups,
        target_words=target_words,
        chunksize=args.chunksize,
    )
    transformed = save_local_word_embeddings(args.out_dir, pipeline, local_contexts, dictionaries)

    print("Phase 5 - aggregating dictionaries and computing cosine scores.")
    build_and_save_concept_vectors(args.out_dir, transformed)

    if args.validation_words:
        print("Phase 6 - saving nearest-neighbor validation.")
        if pipeline.global_result is None:
            raise ValueError("Validation needs a newly fitted global ALC vocabulary. Omit --matrix or rerun without it.")
        cleaned_validation = []
        for word in args.validation_words:
            toks = normalizer.tokenize(word)
            cleaned_validation.extend(toks[:1])
        save_validation_neighbors(args.out_dir, pipeline, transformed, cleaned_validation, args.neighbor_topn)

    print(f"Done. Outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
