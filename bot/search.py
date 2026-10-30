"""Keyword search (BM25) over document chunks.

The corpus is small (hundreds of short documents), so a local index is enough
and keeps Groq tokens for answering. Ukrainian words are cut to a short prefix
so "внесок" and "внески", "гриби" and "грибів" meet each other.
"""

import math
import re
import textwrap
from collections import Counter
from dataclasses import dataclass

from .storage import Document

STOPWORDS = {
    "і", "й", "та", "а", "але", "чи", "що", "як", "де", "коли", "хто", "це", "той", "ті", "на", "в", "у",
    "з", "із", "зі", "до", "від", "для", "по", "за", "про", "під", "над", "біля", "не", "ні", "так", "же", "ж",
    "бо", "щоб", "його", "її", "їх", "ми", "ви", "ти", "я", "він", "вона", "воно", "вони", "мені", "нам", "вам",
    "можна", "треба", "є", "був", "була", "буде", "яка", "який", "яке", "які", "скільки", "чому", "тут", "там",
    "наш", "наша", "наше", "наші", "свій", "своя", "свої", "все", "всі", "усі", "також", "ще", "вже", "лише",
    # common in questions and titles, they pull unrelated documents up: "поруч" is in 16 titles
    "поруч", "робити", "зробити", "якщо", "потрібно",
}

CHUNK_CHARS = 700


# Words every document shares; they say nothing about what a document is about.
COMMON_STEMS = ("рукави", "лісу", "лісі", "лісом", "ліс", "мешкан", "звір", "порад")

RELATIVE_CUTOFF = 0.35

# Words animals ask with and words the documents are written with: "що їсти" is answered by
# "Чим поживитись", "хто живе" by the statute's "живуть", "скільки платити" by "внесок".
SYNONYMS = [
    {"їсти", "їжа", "їжу", "їжі", "поживитись", "пожива", "поживу", "харчі", "смачне", "підкріпитися", "годуватися"},
    {"живе", "живуть", "мешкає", "мешкають", "проживає"},
    {"внесок", "внески", "платити", "сплачувати", "здавати", "плата"},
    {"чергує", "чергування", "обов'язки", "доглядає"},
    {"небезпека", "небезпеки", "небезпечно", "боятися", "остерігатися", "загроза"},
    {"стежка", "стежки", "дорога", "шлях", "дійти", "пройти"},
]


def stems(word: str) -> list[str]:
    """A short and a longer prefix: "внесок"/"внески" meet on "внес", "чергує"/"чергування" on "черг",
    while words that share the longer prefix count twice."""
    if len(word) <= 4:
        return [word]
    return [word[:4], word[:6]] if len(word) > 6 else [word[:4], word]


def tokens(text: str) -> list[str]:
    words = re.findall(r"[а-щьюяіїєґa-z0-9']+", text.lower().replace("’", "'").replace("ʼ", "'"))
    result = []
    for word in words:
        if word in STOPWORDS or len(word) < 2 or word.startswith(COMMON_STEMS):
            continue
        result.extend(stems(word))
    return result


def expand(query_terms: set[str]) -> set[str]:
    for group in SYNONYMS:
        group_terms = {term for word in group for term in tokens(word)}
        if query_terms & group_terms:
            query_terms = query_terms | group_terms
    return query_terms


def split_chunks(text: str) -> list[str]:
    """Pieces of up to about CHUNK_CHARS. Word and PDF files separate paragraphs with a single line break,
    so a whole guide can come without a blank line: such text is cut by lines, then by sentences."""
    pieces = []
    for paragraph in re.split(r"\n\s*\n|\n(?=[-–•])", text):
        pieces += short_pieces(paragraph.strip())
    return pack(glue_headings(pieces), "\n") or [text]


def short_pieces(paragraph: str) -> list[str]:
    if len(paragraph) <= CHUNK_CHARS:
        return [paragraph] if paragraph else []
    lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
    if len(lines) > 1:
        return [piece for line in lines for piece in short_pieces(line)]
    sentences = re.split(r"(?<=[.!?…])\s+", paragraph)
    return pack([piece for sentence in sentences for piece in textwrap.wrap(sentence, CHUNK_CHARS)], " ")


def looks_like_heading(line: str) -> bool:
    """ "Під дубом" or "Біля річки:" over a section, not a sentence."""
    return len(line) <= 80 and not line.endswith((".", "!", "?", "…", ";", ","))


def glue_headings(pieces: list[str]) -> list[str]:
    """A heading stays with the text under it, so a chunk never ends with the next section's title."""
    glued, headings = [], []
    for piece in pieces:
        if looks_like_heading(piece) and len(headings) < 3:
            headings.append(piece)
            continue
        glued.append("\n".join(headings + [piece]))
        headings = []
    return glued + (["\n".join(headings)] if headings else [])


def pack(pieces: list[str], separator: str) -> list[str]:
    """Join pieces into chunks of up to CHUNK_CHARS."""
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(separator) + len(piece) > CHUNK_CHARS:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}{separator}{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


@dataclass
class Hit:
    doc: Document
    score: float


class Index:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.chunks: list[tuple[Document, Counter, int]] = []
        for doc in documents:
            title_tokens = tokens(doc.title) * 2
            for chunk in split_chunks(doc.text):
                terms = Counter(title_tokens + tokens(chunk))
                self.chunks.append((doc, terms, sum(terms.values())))
        self.avg_len = sum(length for *_, length in self.chunks) / max(len(self.chunks), 1)
        self.df = Counter()
        for _, terms, _ in self.chunks:
            self.df.update(terms.keys())

    def search(self, query: str, allowed=None, limit: int = 5, cutoff: float = RELATIVE_CUTOFF) -> list[Hit]:
        """Best documents for the query; `allowed(doc)` filters by place, `cutoff` drops
        documents scoring under that share of the best one."""
        query_terms = expand(set(tokens(query)))
        if not query_terms:
            return []
        n = len(self.chunks)
        best: dict[int, Hit] = {}
        for doc, terms, length in self.chunks:
            if allowed and not allowed(doc):
                continue
            score = 0.0
            for term in query_terms:
                tf = terms.get(term)
                if not tf:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / self.avg_len))
            if score > 0 and (doc.id not in best or best[doc.id].score < score):
                best[doc.id] = Hit(doc, score)
        hits = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
        return [hit for hit in hits if hit.score >= cutoff * hits[0].score][:limit] if hits else []
