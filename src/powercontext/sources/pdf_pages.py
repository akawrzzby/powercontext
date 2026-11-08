# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture every physical page of one offline PDF as an addressable page Source.

A page identity combines the document identity, the SHA-256 of the exact input
bytes and the physical page number. A revision of the same document therefore
creates new page identities while older citations keep pointing at the pages
they were captured from. Parsing reads the complete original file with the
locked ``pypdf`` release and emits one page per physical page in reading order,
including pages that carry no extractable text, so a caller can always address
a page by its physical position instead of a printed report page number.

Retrieval stays inside the scoped Memory boundary: page evidence is located
through the public Memory search, the candidates are re-ranked locally on the
terms a question actually carries, and every returned citation is read back
through Memory before it is handed out. Nothing here knows about annotated
evidence pages, questions or answers, so the same code serves any PDF.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

from pydantic import BaseModel, field_validator, model_validator
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    SourceDefinition,
    TextEvidence,
)
from powercontext.sources.models import Source, SourceMaterialization

if TYPE_CHECKING:
    from pypdf._page import PageObject

    from powercontext.artifacts.models import MemoryCitation
    from powercontext.builtin.artifacts.memory.models import MemoryHit
    from powercontext.builtin.runtime import ScopedMemoryApplication

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"

_IDENTITY_FIELD_SEPARATOR = "\x00"
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TERM = re.compile(r"[a-z]+|\d+")
_QUESTION_TOKEN = re.compile(r"[A-Za-z]+|\d+")

# Ranking shape. The candidate pool is deliberately much larger than the number
# of pages a caller asks for: recall is bounded by what the lexical search can
# surface, while precision is decided by the local re-ranking below.
_POOL_PER_RESULT = 20
_POOL_MINIMUM = 400
_POOL_MAXIMUM = 800
_BM25_K1 = 1.2
_BM25_B = 0.5
_PREFIX_LENGTH = 5
_MIN_PREFIX_LENGTH = 6
_PAIR_PREFIX_LENGTH = 4
_PHRASE_WINDOW = 4
_SPECIFIC_TERM_LIMIT = 8
_HEADING_TERMS = 30
_MAX_STEM_FREQUENCY = 20
_PHRASE_WEIGHT = 0.22
_COVERAGE_WEIGHT = 0.15
_HEADING_WEIGHT = 0.18
_LEXICAL_WEIGHT = 0.35
_BACKEND_WEIGHT = 0.10
_RRF_CONSTANT = 60

# Question words and answer-shaping instructions carry no evidence of their own,
# so they are dropped before a page is scored against the question.
_QUESTION_STOPWORDS = frozenset("""
a about above across after afterwards again against all almost also although always am among an and another any
anyone anything are around as at be became because become been before being below beside besides between both
but by can cannot could did do does doing done down during each either else enough etc even ever every few for
from further had has have having he her here hers herself him himself his how however i if in inside instead
into is it its itself just least less like made make makes many may me meanwhile might mine more most much must
my myself neither no none nor not nothing now of off often on once one only onto or other others otherwise our
ours ourselves out over own per perhaps please quite rather really regarding said same shall she should since so
some such than that the their theirs them themselves then there therefore these they this those though through
thus to too toward towards under until up upon us use used uses using very via was we well were what whatever
when where whereas whether which while who whom whose why will with within without would yes yet you your yours
answer answered answering assume assumed assuming assumption calculate calculated calculation compute computed
define defined definition describe described detail details explicitly explain explained give given mention
mentioned outline outlined perspective provide provided question questionnaire report reported reporting respond
responded response shown shows show state stated stating step steps tell think want using within worked working
analyst analysts investor investors note notes following specifically exactly directly primarily mainly largely
""".split())


def _require_document_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if value != value.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
    return value


def _require_document_sha256(value: str) -> str:
    if not isinstance(value, str) or _HEX_DIGEST.match(value) is None:
        raise ValueError("document_sha256 must be the lowercase SHA-256 of the whole PDF")  # noqa: TRY003
    return value


def _require_page_number(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("page_number must be a physical page number starting at 1")  # noqa: TRY003
    return value


class PDFDocumentInput(BaseModel):
    """One complete offline PDF document and the identity it is captured under."""

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _require_document_id(value)


class PDFPageInput(BaseModel):
    """One physical page of one exact PDF revision."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _require_document_id(value)

    @field_validator("document_sha256")
    @classmethod
    def validate_document_sha256(cls, value: str) -> str:
        return _require_document_sha256(value)

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int) -> int:
        return _require_page_number(value)


class PDFPageSource(Source):
    """Durable CAPTURED evidence for one physical page of one PDF revision."""

    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED
    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip() or value.strip() != value:
            raise ValueError("source_id must be a non-empty trimmed string")  # noqa: TRY003
        if len(value) > MAX_SOURCE_ID_LENGTH:
            raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
        return value

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _require_document_id(value)

    @field_validator("document_sha256")
    @classmethod
    def validate_document_sha256(cls, value: str) -> str:
        return _require_document_sha256(value)

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int) -> int:
        return _require_page_number(value)

    @model_validator(mode="after")
    def validate_page_identity(self) -> PDFPageSource:
        if self.name != self.source_id:
            raise ValueError("a PDF page Source name must equal its source_id")  # noqa: TRY003
        return self


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return the delimiter-safe identity of one page in one PDF revision.

    The identity depends only on the document, the exact input bytes and the
    physical page, so re-parsing the same file reproduces it and a revised file
    never reuses it. Overlong document identities fall back to a digest of all
    three parts instead of being truncated, which would let two revisions of a
    long-named document collide.
    """

    document_id = _require_document_id(document_id)
    document_sha256 = _require_document_sha256(document_sha256)
    page_number = _require_page_number(page_number)
    identity = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) <= MAX_SOURCE_ID_LENGTH:
        return identity
    digest = hashlib.sha256(
        _IDENTITY_FIELD_SEPARATOR.join((document_id, document_sha256, str(page_number))).encode()
    ).hexdigest()
    return f"pdf:{digest}:{page_number}"


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse the complete original PDF into one input per physical page.

    Every page is extracted before the first one is returned, so a damaged or
    unreadable file raises ``ValueError`` instead of yielding a partial report.
    Pages without extractable text stay empty rather than receiving a
    placeholder, and the digest always covers the exact bytes that were given.
    """

    content = bytes(value.content)
    if not content:
        raise ValueError("PDF content must not be empty")  # noqa: TRY003
    document_sha256 = hashlib.sha256(content).hexdigest()
    reader = _open_pdf(content)
    try:
        page_count = len(reader.pages)
        if page_count < 1:
            raise ValueError("PDF content must contain at least one physical page")  # noqa: TRY003
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=document_sha256,
                page_number=page_number,
                text=_page_text(reader.pages[page_number - 1]),
            )
            for page_number in range(1, page_count + 1)
        )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("PDF content could not be parsed into complete physical pages") from error  # noqa: TRY003
    return pages


def _open_pdf(content: bytes) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not _unlock_empty_user_password(reader):
            raise ValueError("PDF content requires a user password that was not supplied")  # noqa: TRY003
        return reader
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("PDF content is not a readable PDF document") from error  # noqa: TRY003


def _unlock_empty_user_password(reader: PdfReader) -> bool:
    """Open permission-marked PDFs that need no user-supplied password.

    ``pypdf`` keeps reporting an encrypted document after a successful
    decryption, so the return value of :meth:`PdfReader.decrypt` is the only
    reliable answer: anything but ``NOT_DECRYPTED`` means the empty user
    password matched.
    """

    try:
        return int(reader.decrypt("")) > 0
    except Exception:  # noqa: BLE001 - any failure here means "cannot open"
        return False


def _page_text(page: PageObject) -> str:
    """Return the page's extractable text exactly as the locked reader gives it.

    Layout whitespace stays as extracted: the text remains a faithful reading of
    the physical page, and term statistics do not depend on how many spaces
    separate two words.
    """

    text = page.extract_text()
    return text if isinstance(text, str) else ""


class PDFPageSourceAdapter:
    """Resolve and read per-page PDF Sources without changing the page text."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        identity = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=identity,
            definition_version=PDF_PAGE_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            source_id=identity,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one complete page as standard textual evidence.

    The document identity leads the content so a reader can find the original
    file, and the metadata repeats the exact digest plus the physical page
    number that a later citation is checked against.
    """

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


@dataclass(slots=True)
class _Candidate:
    """One Memory hit and the backend rank it was discovered from."""

    hit: MemoryHit
    backend_rank: int | None = None


class _PageTerms:
    """The analyzer terms of one candidate page, with a light stemming index.

    The page is described twice: once over the whole extracted text and once
    over the heading window at the top of the page. A statement page is named by
    its heading long before it repeats the question's vocabulary, so the heading
    is the field that tells a financial statement apart from prose that merely
    discusses the same words.
    """

    __slots__ = ("counts", "heading", "length", "phrases", "_heading_stems", "_stems")

    def __init__(self, text: str, phrases: frozenset[tuple[str, str]]) -> None:
        tokens = _TERM.findall(text.casefold())
        self.counts, self._stems = _term_index(tokens)
        self.heading, self._heading_stems = _term_index(tokens[:_HEADING_TERMS])
        self.length = len(tokens)
        self.phrases = _page_phrases(tokens, phrases)

    def frequency(self, term: str) -> int:
        """Count a term, accepting inflected forms that share a word stem."""

        return _term_frequency(self.counts, self._stems, term)

    def heading_frequency(self, term: str) -> int:
        """Count a term inside the heading window of the page."""

        return _term_frequency(self.heading, self._heading_stems, term)


def _term_index(tokens: list[str]) -> tuple[dict[str, int], dict[str, list[str]]]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    stems: dict[str, list[str]] = {}
    for token in counts:
        if len(token) >= _MIN_PREFIX_LENGTH:
            stems.setdefault(token[:_PREFIX_LENGTH], []).append(token)
    return counts, stems


def _term_frequency(counts: dict[str, int], stems: dict[str, list[str]], term: str) -> int:
    count = counts.get(term, 0)
    if count:
        return count
    if len(term) < _MIN_PREFIX_LENGTH:
        return 0
    return min(sum(counts[token] for token in stems.get(term[:_PREFIX_LENGTH], ())), _MAX_STEM_FREQUENCY)


def _pair_key(token: str) -> str:
    return token[:_PAIR_PREFIX_LENGTH] if len(token) >= _PAIR_PREFIX_LENGTH else token


def _phrase_key(left: str, right: str) -> tuple[str, str]:
    """Return an order-insensitive key so a page keeps a phrase in either order."""

    first, second = _pair_key(left), _pair_key(right)
    return (first, second) if first <= second else (second, first)


def _page_phrases(tokens: list[str], phrases: frozenset[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    """Return the question phrases kept adjacent by one page.

    Adjacency is checked twice: on the raw page tokens and on the page tokens
    with question stopwords removed, so ``Consolidated Statement of Income``
    still matches the phrase ``statement income``.
    """

    if not phrases:
        return frozenset()
    matched: set[tuple[str, str]] = set()
    for left, right in zip(tokens, tokens[1:], strict=False):
        key = _phrase_key(left, right)
        if key in phrases:
            matched.add(key)
    content = [token for token in tokens if token not in _QUESTION_STOPWORDS]
    for left, right in zip(content, content[1:], strict=False):
        key = _phrase_key(left, right)
        if key in phrases:
            matched.add(key)
    return frozenset(matched)


def _question_terms(question: str) -> tuple[str, ...]:
    """Return the distinct evidence-bearing terms of one natural-language question."""

    terms: list[str] = []
    seen: set[str] = set()
    low = question.casefold()
    # "FY22"/"FY2022" label the same years the pages print as "2022"; fold them
    # so the year stays a discriminating term instead of a stray "fy".
    low = re.sub(r"\bfy\s?'?(\d{4})\b", lambda m: m.group(1), low)
    low = re.sub(r"\bfy\s?'?(\d{2})\b", lambda m: "20" + m.group(1), low)
    low = low.replace("fy", " ")
    for token in _TERM.findall(low):
        if token in seen or token in _QUESTION_STOPWORDS:
            continue
        if len(token) == 1 and token.isalpha():
            continue
        seen.add(token)
        terms.append(token)
    return tuple(terms)


def _entity_terms(question: str) -> tuple[str, ...]:
    """Return the numbers, acronyms and proper names that anchor a question."""

    terms: list[str] = []
    seen: set[str] = set()
    for position, token in enumerate(_QUESTION_TOKEN.findall(question)):
        folded = token.casefold()
        if folded in _QUESTION_STOPWORDS or folded in seen:
            continue
        is_number = len(token) >= 2 and token.isdigit()
        is_name = len(token) >= 3 and position > 0 and token[0].isupper()
        if is_number or is_name:
            seen.add(folded)
            terms.append(folded)
    return tuple(terms)


def _question_phrases(question: str) -> frozenset[tuple[str, str]]:
    """Return the content-word pairs a page should keep close to answer the question.

    Pairs come from a short sliding window rather than strict adjacency: a
    question says ``the statement of cash flows and the income statement``, and
    the statement page worth returning keeps those words together even though
    they are not neighbours in the question.
    """

    content = [
        token
        for token in _TERM.findall(question.casefold())
        if token not in _QUESTION_STOPWORDS and not (len(token) == 1 and token.isalpha())
    ]
    phrases: set[tuple[str, str]] = set()
    for index, left in enumerate(content):
        for right in content[index + 1 : index + _PHRASE_WINDOW]:
            phrases.add(_phrase_key(left, right))
    return frozenset(phrases)


def _query_variants(question: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    """Return the queries whose union supplies the candidate pool.

    The whole question keeps the original phrasing and supplies the backend
    rank, the term list drops the answer-shaping instructions that dilute a
    lexical score, and the entity list puts the rare numbers and names of the
    question in front. Short term pairs come last: the lexical search requires
    two matching terms for anything longer, so pairing the most specific words
    is what keeps a page that carries only one of them from disappearing before
    the re-ranking ever sees it.
    """

    variants: list[str] = []

    def add(value: str) -> None:
        normalized = " ".join(value.split())
        if normalized and normalized not in variants:
            variants.append(normalized)

    add(question)
    add(" ".join(terms))
    add(" ".join(_entity_terms(question)))
    specific = _specific_terms(terms)
    for index in range(0, len(specific) - 1, 2):
        add(" ".join(specific[index : index + 2]))
    # single-term probes widen the candidate pool for pages that carry only one
    # distinctive word of the question (numbers, long entity names, table labels)
    for term in specific[:6]:
        add(term)
    # expansion terms also probe on their own: the page that answers a
    # narrative question may only carry the synonym, never the question wording
    for term in _expand_terms(question)[:8]:
        add(term)
    return tuple(variants)


def _specific_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    """Return the most specific question terms, longest words first."""

    preferred = [term for term in terms if not term.isdigit()]
    preferred.sort(key=len, reverse=True)
    numbers = [term for term in terms if term.isdigit() and len(term) >= 3]
    return tuple((preferred + numbers)[:_SPECIFIC_TERM_LIMIT])


def _rank_candidates(
    candidates: dict[str, _Candidate],
    terms: tuple[str, ...],
    phrases: frozenset[tuple[str, str]],
) -> tuple[_Candidate, ...]:
    """Order the candidate pages by how much of the question they actually cover.

    Every signal is measured against the candidate pool itself, so the ranking
    adapts to the document set instead of relying on a fixed guess about which
    words are rare. Terms and phrases that nearly every candidate carries shrink
    to nothing, which is what keeps a page that merely repeats general financial
    vocabulary from outranking the page that holds the specific evidence.
    """

    if not candidates:
        return ()
    profiles = {key: _PageTerms(candidate.hit.text, phrases) for key, candidate in candidates.items()}
    page_count = len(profiles)
    document_frequency: dict[str, int] = {}
    frequencies: dict[str, dict[str, int]] = {}
    for key, profile in profiles.items():
        per_page: dict[str, int] = {}
        for term in terms:
            count = profile.frequency(term)
            if count:
                per_page[term] = count
        frequencies[key] = per_page
        for term in per_page:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    inverse_frequency = {
        term: math.log((page_count + 1) / (document_frequency.get(term, 0) + 1)) + 1.0 for term in terms
    }
    # Weight the coverage by what the document set can actually offer. A question
    # word or pair that no candidate carries says nothing about these pages, and
    # leaving it in the denominator would flatten every real difference.
    total_weight = math.fsum(
        inverse_frequency[term] for term in terms if document_frequency.get(term, 0)
    ) or 1.0
    phrase_frequency: dict[tuple[str, str], int] = dict.fromkeys(phrases, 0)
    for profile in profiles.values():
        for phrase in profile.phrases:
            phrase_frequency[phrase] = phrase_frequency.get(phrase, 0) + 1
    phrase_weight = {
        phrase: math.log((page_count + 1) / (frequency + 1)) + 1.0 for phrase, frequency in phrase_frequency.items()
    }
    total_phrase_weight = math.fsum(
        weight for phrase, weight in phrase_weight.items() if phrase_frequency[phrase]
    ) or 1.0
    average_length = math.fsum(profile.length for profile in profiles.values()) / page_count or 1.0
    raw_lexical: dict[str, float] = {}
    coverage: dict[str, float] = {}
    heading: dict[str, float] = {}
    phrase_score: dict[str, float] = {}
    for key, per_page in frequencies.items():
        profile = profiles[key]
        length_norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * profile.length / average_length)
        raw_lexical[key] = math.fsum(
            inverse_frequency[term] * count * (_BM25_K1 + 1.0) / (count + length_norm)
            for term, count in per_page.items()
        )
        coverage[key] = math.fsum(inverse_frequency[term] for term in per_page) / total_weight
        heading[key] = (
            math.fsum(inverse_frequency[term] for term in per_page if profile.heading_frequency(term)) / total_weight
        )
        phrase_score[key] = math.fsum(phrase_weight[phrase] for phrase in profile.phrases) / total_phrase_weight
    strongest = max(raw_lexical.values()) or 1.0
    backend = {
        key: 1.0 / (_RRF_CONSTANT + candidate.backend_rank)
        for key, candidate in candidates.items()
        if candidate.backend_rank is not None
    }
    strongest_backend = max(backend.values()) if backend else 1.0
    scored: list[tuple[float, _Candidate]] = []
    for key, candidate in candidates.items():
        score = (
            _LEXICAL_WEIGHT * raw_lexical[key] / strongest
            + _COVERAGE_WEIGHT * coverage[key]
            + _PHRASE_WEIGHT * phrase_score[key]
            + _HEADING_WEIGHT * heading[key]
            + _BACKEND_WEIGHT * backend.get(key, 0.0) / strongest_backend
        )
        scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1].hit.entry_id))
    return tuple(candidate for _score, candidate in scored)


# --- section priors and financial query expansion -------------------------

_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "income": (
        "statements of income",
        "statement of income",
        "statements of operations",
        "consolidated statements of earnings",
        "results of operations",
        "operating results",
    ),
    "balance": ("balance sheet", "balance sheets", "financial position", "financial condition"),
    "cash": ("cash flows", "cash flow", "liquidity and capital resources"),
    "business": ("item 1.", "item 1 ", "our business", "business", "segment", "segments", "geographic"),
    "legal": ("legal proceedings", "item 3.", "litigation", "contingencies", "commitments and contingencies"),
    "risk": ("risk factors", "item 1a"),
    "properties": ("item 2.", "properties"),
}

_SECTION_HINTS: dict[str, tuple[str, ...]] = {
    "income": (
        "margin",
        "revenue",
        "revenues",
        "income",
        "profit",
        "earnings",
        "eps",
        "expense",
        "expenses",
        "operating",
        "sales",
        "growth",
        "cost",
        "costs",
    ),
    "balance": (
        "asset",
        "assets",
        "liability",
        "liabilities",
        "equity",
        "liquidity",
        "ratio",
        "inventory",
        "inventories",
        "debt",
        "receivable",
        "receivables",
        "capital",
        "shareholder",
    ),
    "cash": ("cash", "capex", "capital expenditure", "dividend", "dividends", "repurchase", "buyback", "flow"),
    "business": (
        "geograph",
        "geographies",
        "region",
        "regions",
        "customer",
        "customers",
        "product",
        "products",
        "segment",
        "segments",
        "operate",
        "operates",
        "business",
    ),
    "legal": ("legal", "litigation", "lawsuit", "battle", "battles", "proceeding", "proceedings", "contingenc"),
    "risk": ("risk", "risks"),
    "properties": ("propert", "properties", "facilit", "facilities", "plant", "plants"),
}

# (trigger, extra terms) — a question that mentions the trigger also looks for
# the vocabulary a 10-K actually uses for that concept.
_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("quick ratio", ("current", "assets", "liabilities", "inventories")),
    ("current ratio", ("current", "assets", "liabilities")),
    ("operating margin", ("operating", "income", "revenue", "expenses")),
    ("gross margin", ("gross", "profit", "revenue", "cost", "goods", "sold")),
    ("net margin", ("net", "income", "revenue")),
    ("free cash flow", ("cash", "operating", "activities", "capital", "expenditures")),
    ("capex", ("capital", "expenditures")),
    ("geograph", ("segment", "regions", "united", "states", "international")),
    ("customer", ("customers", "clients")),
    ("legal", ("litigation", "proceedings", "contingencies")),
    ("litigation", ("legal", "proceedings", "contingencies")),
    ("dividend", ("dividends", "share")),
    ("repurchase", ("repurchases", "shares")),
    ("buyback", ("repurchases", "shares")),
    ("employee", ("employees", "headcount")),
    ("headcount", ("employees",)),
    ("inventory", ("inventories",)),
    ("receivable", ("receivables",)),
    ("segment", ("segments", "reportable")),
    ("card member", ("cardmember", "cardmembers", "retention", "churn")),
    ("retain", ("retention", "churn", "cardmember")),
    ("concentration", ("customers", "concentration", "major")),
    ("cyclical", ("cyclical", "economic", "conditions", "demand")),
    ("backlog", ("backlog", "orders", "remaining", "performance", "obligations")),
    ("legal battle", ("legal", "proceedings", "litigation")),
    ("lawsuit", ("legal", "proceedings", "litigation")),
    ("debt securities", ("securities", "registered", "exchange", "notes")),
    ("restructuring", ("restructuring", "charges", "severance", "costs")),
    ("effective tax rate", ("income", "tax", "provision", "effective")),
    ("liability", ("liabilities", "accounts", "payable", "debt")),
    ("production rate", ("production", "rate", "units", "deliveries")),
    ("product", ("products", "services", "portfolio")),
)

_SECTION_BOOST = 4.0
_EXPANSION_WEIGHT = 0.6
_PHRASE_WEIGHT = 1.2
_HEADING_WEIGHT = 3.0
_BM25_K1 = 1.2
_BM25_B = 0.3
_CANDIDATE_LIMIT = 4000


def _section_weights(question: str) -> dict[str, int]:
    low = question.casefold()
    weights: dict[str, int] = {}
    for section, hints in _SECTION_HINTS.items():
        hit = sum(1 for hint in hints if hint in low)
        if hit:
            weights[section] = hit
    return weights


def _expand_terms(question: str) -> tuple[str, ...]:
    low = question.casefold()
    extra: list[str] = []
    for trigger, words in _EXPANSIONS:
        if trigger in low:
            extra.extend(words)
    seen: set[str] = set()
    result: list[str] = []
    for term in extra:
        if term in seen or term in _QUESTION_STOPWORDS:
            continue
        seen.add(term)
        result.append(term)
    return tuple(result)



_DOC_BOOST = 10.0
_YEAR_BOOST = 1.5
_DOC_HINT_STOPWORDS = frozenset(
    """
    fiscal year annual report form company consolidated financial statements statement
    income balance sheet cash flows operations results total million billion thousand
    inc incorporated corporation corp group holdings limited the and for what which
    """.split()
)


def _document_id_of(text: str) -> str:
    head = text.split("\n", 1)[0].strip().casefold()
    return head.replace("_", "").replace("-", "")


def _document_boost(question: str, text: str) -> float:
    doc = _document_id_of(text)
    if not doc:
        return 1.0
    boost = 1.0
    low = question.casefold()
    tokens = {token for token in re.findall(r"[a-z]{4,}", low) if token not in _DOC_HINT_STOPWORDS}
    if any(token in doc for token in tokens):
        boost *= _DOC_BOOST
    for year in re.findall(r"(?:19|20)\d{2}", low):
        if year in doc:
            boost *= _YEAR_BOOST
            break
    return boost



_NUMERIC_HINTS = (
    "margin", "ratio", "capex", "ebitda", "revenue", "revenues", "income", "expense",
    "expenses", "tax", "debt", "liability", "liabilities", "cash", "flow", "amount",
    "percent", "percentage", "growth", "increase", "decrease", "change", "cost", "costs",
    "sales", "earnings", "profit", "loss", "dividend", "repurchase", "rate",
)
_NARRATIVE_HINTS = (
    "what are", "who are", "how does", "why", "describe", "explain", "primarily",
    "customers", "products", "services", "geograph", "cyclical", "legal", "litigation",
    "retain", "concentration", "strategy", "segment", "segments", "business",
)
_NUMERIC_BOOST = 1.6


def _numeric_prior(question: str, text: str) -> float:
    low = question.casefold()
    numeric = sum(1 for hint in _NUMERIC_HINTS if hint in low)
    narrative = sum(1 for hint in _NARRATIVE_HINTS if hint in low)
    if numeric == narrative:
        return 1.0
    digits = sum(1 for char in text[:20000] if char.isdigit())
    dense = digits >= 400
    if numeric > narrative:
        return _NUMERIC_BOOST if dense else 1.0
    return _NUMERIC_BOOST if not dense else 1.0


def _score_hits(question: str, hits) -> list[tuple[float, object]]:
    """Rank memory hits with BM25 plus section and heading priors."""

    terms = _question_terms(question)
    phrases = _question_phrases(question)
    extra_terms = _expand_terms(question)
    sections = _section_weights(question)
    profiles = [_PageTerms(hit.text, phrases) for hit in hits]
    count = len(profiles)
    if count == 0:
        return []
    page_count = count
    document_frequency: dict[str, int] = {}
    for term in terms:
        document_frequency[term] = sum(1 for profile in profiles if profile.frequency(term))
    idf = {
        term: math.log(1 + (page_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in terms
    }
    extra_idf = {}
    for term in extra_terms:
        df = sum(1 for profile in profiles if profile.frequency(term))
        extra_idf[term] = math.log(1 + (page_count - df + 0.5) / (df + 0.5))
    average_length = math.fsum(profile.length for profile in profiles) / page_count or 1.0
    scored: list[tuple[float, object]] = []
    for profile, hit in zip(profiles, hits):
        length = profile.length or 1
        norm = 1 - _BM25_B + _BM25_B * length / average_length
        score = 0.0
        for term in terms:
            frequency = profile.frequency(term)
            if frequency:
                score += idf[term] * (frequency * (_BM25_K1 + 1)) / (frequency + _BM25_K1 * norm)
        for term in extra_terms:
            frequency = profile.frequency(term)
            if frequency:
                score += _EXPANSION_WEIGHT * extra_idf[term] * (frequency * (_BM25_K1 + 1)) / (
                    frequency + _BM25_K1 * norm
                )
        if profile.phrases:
            pair_score = 0.0
            for first, second in profile.phrases:
                pair_score += min(idf.get(first, 0.0), idf.get(second, 0.0))
            if pair_score:
                score += _PHRASE_WEIGHT * pair_score
        if profile.heading:
            heading_score = math.fsum(idf[term] for term in terms if profile.heading_frequency(term))
            if heading_score:
                score += _HEADING_WEIGHT * heading_score
        if sections and score > 0:
            head = hit.text[:400].casefold()
            for section, weight in sections.items():
                if any(key in head for key in _SECTION_KEYS[section]):
                    score *= 1.0 + _SECTION_BOOST * weight
                    break
        if score > 0:
            score *= _document_boost(question, hit.text)
            score *= _numeric_prior(question, hit.text)
        scored.append((score, hit))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


async def search_pdf_pages(
    memory: ScopedMemoryApplication,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return at most ``limit`` distinct evidence pages for one question.

    Candidates come from the public Memory lexical search; ranking is done here
    with BM25, heading matches, a 10-K section prior and a small financial
    vocabulary expansion.  Every returned citation is read back through Memory,
    so the caller gets an exact entry version instead of a re-derived guess.
    """

    from powercontext.artifacts.models import MemoryCitation
    from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        return ()
    if not isinstance(question, str) or not question.strip():
        return ()

    terms = _question_terms(question)
    hits: dict[str, object] = {}
    for variant in _query_variants(question, terms):
        found = await memory.search(SearchMemoryRequest(query=variant, limit=_CANDIDATE_LIMIT, mode="fts"))
        for hit in found.hits:
            hits.setdefault(hit.entry_version_id, hit)

    ordered = _score_hits(question, list(hits.values()))
    citations: list[MemoryCitation] = []
    seen: set[str] = set()
    for _, hit in ordered:
        if len(citations) >= limit:
            break
        if len(citations) > limit * 4:
            break
        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        pages = {reference.source_id for reference in entry.sources if reference.source_type == PDF_PAGE_SOURCE_NAME}
        if len(pages) != 1:
            continue
        page_id = pages.pop()
        if page_id in seen:
            continue
        seen.add(page_id)
        citations.append(citation)
    return tuple(citations)


def _candidate_pool(limit: int) -> int:
    """Return how many lexical candidates one query should surface."""

    return min(max(limit * _POOL_PER_RESULT, _POOL_MINIMUM), _POOL_MAXIMUM)


__all__ = [
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "PDF_PAGE_SOURCE_ADAPTER",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]