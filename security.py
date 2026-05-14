"""
security.py — ATS Analyzer Security Module
Privacy-by-Design: All file handling is in-memory only (io.BytesIO).
No user data is persisted to disk after processing.
"""

import re
import html
import unicodedata
import io
import hashlib
import logging
from typing import Optional

# ── Logging (no PII in logs) ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

# Characters that could be used in injection attacks
_INJECTION_PATTERN = re.compile(
    r"[<>{}\[\]\\;`$|&]|javascript:|data:|vbscript:|on\w+=",
    re.IGNORECASE,
)

# PII patterns (used for masking in logs/debug output only — never strip from
# the actual resume text that is being analyzed)
_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(
        r"(\+?\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{4}"
    ),
    "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),          # Indian PAN
    "aadhaar": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),  # Aadhaar
}


# ── File Validation ───────────────────────────────────────────────────────────

class FileValidationError(ValueError):
    """Raised when an uploaded file fails security validation."""


def validate_upload(file_bytes: bytes, filename: str) -> None:
    """
    Validate an uploaded file before any processing.

    Checks:
        1. File size does not exceed MAX_FILE_SIZE_MB
        2. File extension is in ALLOWED_EXTENSIONS
        3. File magic bytes match the declared extension

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename provided by the user.

    Raises:
        FileValidationError: If any check fails.
    """
    # 1. Size check
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise FileValidationError(
            f"File exceeds {MAX_FILE_SIZE_MB} MB limit "
            f"({len(file_bytes) / 1_048_576:.1f} MB uploaded)."
        )

    # 2. Extension check
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type '{ext}'. "
            f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} are accepted."
        )

    # 3. Magic-byte check (prevent extension spoofing)
    if ext == ".pdf":
        if not file_bytes.startswith(b"%PDF"):
            raise FileValidationError(
                "File does not appear to be a valid PDF (magic bytes mismatch)."
            )
    elif ext == ".docx":
        # DOCX is a ZIP archive — PK magic bytes
        if not file_bytes[:2] == b"PK":
            raise FileValidationError(
                "File does not appear to be a valid DOCX (magic bytes mismatch)."
            )

    logger.info(
        "File validated OK | ext=%s size=%d hash=%s",
        ext,
        len(file_bytes),
        _file_fingerprint(file_bytes),
    )


def to_memory_stream(file_bytes: bytes) -> io.BytesIO:
    """
    Wrap raw bytes in an in-memory BytesIO buffer.
    Parsers read from this buffer; nothing touches the filesystem.
    """
    buf = io.BytesIO(file_bytes)
    buf.seek(0)
    return buf


def _file_fingerprint(file_bytes: bytes) -> str:
    """Return a short SHA-256 fingerprint for audit logging (not PII)."""
    return hashlib.sha256(file_bytes).hexdigest()[:12]


# ── Text Sanitization ─────────────────────────────────────────────────────────

def sanitize_text_input(raw: str, max_length: int = 50_000) -> str:
    """
    Sanitize free-text inputs (job description, resume text) to prevent
    injection attacks and normalise encoding.

    Steps:
        1. HTML-escape potential markup
        2. Strip injection-like characters
        3. Normalize Unicode to NFC
        4. Collapse excessive whitespace
        5. Enforce max length

    Args:
        raw:        Raw string from user input or parsed document.
        max_length: Hard cap on character count.

    Returns:
        Sanitized string safe for downstream NLP processing.
    """
    if not isinstance(raw, str):
        raw = str(raw)

    # 1. HTML-escape (neutralises <script>, etc.)
    text = html.escape(raw, quote=False)
    # Unescape common punctuation that html.escape over-escapes for our use case
    text = text.replace("&amp;", "&").replace("&#x27;", "'").replace("&quot;", '"')

    # 2. Remove injection patterns
    text = _INJECTION_PATTERN.sub(" ", text)

    # 3. Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # 4. Normalise whitespace (keep newlines for structure)
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal space
    text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 consecutive newlines

    # 5. Length cap
    if len(text) > max_length:
        logger.warning("Input truncated from %d to %d chars.", len(text), max_length)
        text = text[:max_length]

    return text.strip()


# ── PII Utilities (for display / logging only) ────────────────────────────────

def mask_pii_for_display(text: str) -> str:
    """
    Return a copy of `text` with PII replaced by placeholders.
    Used ONLY for audit logs or debug display — never applied to analysis data.
    """
    masked = text
    masked = _PII_PATTERNS["email"].sub("[EMAIL REDACTED]", masked)
    masked = _PII_PATTERNS["phone"].sub("[PHONE REDACTED]", masked)
    masked = _PII_PATTERNS["ssn"].sub("[SSN REDACTED]", masked)
    masked = _PII_PATTERNS["pan"].sub("[PAN REDACTED]", masked)
    masked = _PII_PATTERNS["aadhaar"].sub("[AADHAAR REDACTED]", masked)
    return masked


def extract_pii_inventory(text: str) -> dict:
    """
    Detect which PII categories are present in the text.
    Returns a dict of {category: count} for the privacy report shown to the user.
    Does NOT return the actual PII values.
    """
    inventory = {}
    for label, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            inventory[label.upper()] = len(matches)
    return inventory
