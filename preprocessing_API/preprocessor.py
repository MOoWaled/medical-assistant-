import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


FALLBACK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
}


def _has_resource(path: str) -> bool:
    """Return whether a local NLTK resource exists; never download at runtime."""
    try:
        nltk.data.find(path)
    except LookupError:
        return False
    return True


class MedicalTextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.stop_words = FALLBACK_STOP_WORDS

        self.can_tokenize_with_nltk = _has_resource("tokenizers/punkt") and _has_resource(
            "tokenizers/punkt_tab"
        )
        self.can_lemmatize = _has_resource("corpora/wordnet") or _has_resource("corpora/wordnet.zip")
        self.lemmatizer = WordNetLemmatizer() if self.can_lemmatize else None

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        if self.can_tokenize_with_nltk:
            tokens = word_tokenize(text)
        else:
            # Keeps preprocessing available offline when the NLTK downloader is rate-limited.
            tokens = re.findall(r"[a-zA-Z0-9]+", text)
        
        filtered = [w for w in tokens if w not in self.stop_words]
        lemmatized = [self.lemmatizer.lemmatize(w) for w in filtered] if self.lemmatizer else filtered
        
        return " ".join(lemmatized)
