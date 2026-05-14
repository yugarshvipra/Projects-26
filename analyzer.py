"""
analyzer.py — ATS Analyzer Core Engine
Computes match score, keyword gap analysis, and actionable ATS feedback.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import ParsedResume
from security import sanitize_text_input

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class ATSScore:
    """Breakdown of the overall ATS compatibility score."""
    overall: float = 0.0           # 0–100
    content_match: float = 0.0     # cosine similarity (0–100)
    keyword_coverage: float = 0.0  # % of JD keywords found in resume
    formatting: float = 0.0        # formatting quality (0–100)
    completeness: float = 0.0      # key sections present (0–100)
    contact_info: float = 0.0      # contact details score (0–100)

    # Score weights (must sum to 1.0)
    W_CONTENT: float = field(default=0.40, repr=False)
    W_KEYWORDS: float = field(default=0.30, repr=False)
    W_FORMAT: float = field(default=0.15, repr=False)
    W_COMPLETE: float = field(default=0.10, repr=False)
    W_CONTACT: float = field(default=0.05, repr=False)

    def compute_overall(self) -> None:
        self.overall = round(
            self.content_match   * self.W_CONTENT
            + self.keyword_coverage * self.W_KEYWORDS
            + self.formatting       * self.W_FORMAT
            + self.completeness     * self.W_COMPLETE
            + self.contact_info     * self.W_CONTACT,
            1,
        )


@dataclass
class AnalysisResult:
    """Complete analysis output returned to the UI layer."""
    score: ATSScore = field(default_factory=ATSScore)

    # Keyword analysis
    jd_keywords: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    resume_top_terms: list[str] = field(default_factory=list)

    # Feedback
    formatting_issues: list[str] = field(default_factory=list)
    formatting_positives: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    improvement_tips: list[str] = field(default_factory=list)

    # Summary sentence
    verdict: str = ""
    error: Optional[str] = None


# ── Tokenization Helpers ──────────────────────────────────────────────────────

# Stop words relevant to resume/JD context (supplement sklearn's list)
_EXTRA_STOPWORDS = {
    "experience", "work", "job", "position", "role", "company", "team",
    "years", "year", "month", "skill", "skills", "ability", "responsibilities",
    "responsible", "duties", "required", "requirements", "preferred",
    "seeking", "looking", "candidate", "applicant", "employee",
    "strong", "good", "excellent", "great", "plus", "bonus",
    "must", "will", "also", "etc", "eg", "ie", "including", "include",
}

_NGRAM_RANGE = (1, 3)   # uni-, bi-, trigrams for richer matching


def _clean_for_tfidf(text: str) -> str:
    """Lowercase, remove punctuation (keep hyphens inside words), strip digits-only tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    # Remove standalone digit tokens
    text = re.sub(r"\b\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_top_ngrams(text: str, top_n: int = 30) -> list[str]:
    """
    Use TF-IDF to pull the most important n-grams from a single document.
    Returns a list of terms sorted by importance.
    """
    cleaned = _clean_for_tfidf(text)
    if len(cleaned.split()) < 5:
        return []

    vec = TfidfVectorizer(
        ngram_range=_NGRAM_RANGE,
        stop_words="english",
        max_features=500,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vec.fit_transform([cleaned])
        scores = dict(zip(vec.get_feature_names_out(),
                          tfidf_matrix.toarray()[0]))
        # Filter extra stop words and very short tokens
        filtered = {
            k: v for k, v in scores.items()
            if k not in _EXTRA_STOPWORDS and len(k) > 2 and v > 0
        }
        sorted_terms = sorted(filtered, key=filtered.get, reverse=True)  # type: ignore[arg-type]
        return sorted_terms[:top_n]
    except ValueError:
        return []


# ── Cosine Similarity ─────────────────────────────────────────────────────────

def _cosine_match(text_a: str, text_b: str) -> float:
    """
    Compute TF-IDF cosine similarity between two texts.
    Returns a value in [0, 100].
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    cleaned_a = _clean_for_tfidf(text_a)
    cleaned_b = _clean_for_tfidf(text_b)

    vec = TfidfVectorizer(
        ngram_range=_NGRAM_RANGE,
        stop_words="english",
        sublinear_tf=True,
        max_features=10_000,
    )
    try:
        matrix = vec.fit_transform([cleaned_a, cleaned_b])
        sim = cosine_similarity(matrix[0], matrix[1])[0][0]
        return round(float(sim) * 100, 1)
    except ValueError as exc:
        logger.warning("Cosine similarity failed: %s", exc)
        return 0.0


# ── Keyword Gap Analysis ──────────────────────────────────────────────────────

def _keyword_gap(
    resume_text: str,
    jd_text: str,
    jd_top_n: int = 40,
) -> tuple[list[str], list[str], list[str]]:
    """
    Extract important keywords from the JD and check which ones the resume
    addresses.

    Returns:
        (jd_keywords, matched, missing)
    """
    jd_keywords = _extract_top_ngrams(jd_text, top_n=jd_top_n)
    resume_lower = resume_text.lower()

    matched, missing = [], []
    for kw in jd_keywords:
        # A keyword "matches" if all words of the phrase appear in the resume
        words = kw.split()
        if len(words) == 1:
            found = bool(re.search(rf"\b{re.escape(kw)}\b", resume_lower))
        else:
            # For phrases: check if phrase appears (with possible minor gaps)
            found = kw in resume_lower
        (matched if found else missing).append(kw)

    return jd_keywords, matched, missing


# ── Section Completeness ──────────────────────────────────────────────────────

_IDEAL_SECTIONS = {
    "Summary": ["summary", "objective", "profile", "about"],
    "Experience": ["experience", "work history", "professional experience", "employment"],
    "Education": ["education", "academic", "qualification"],
    "Skills": ["skills", "technical skills", "core competencies", "proficiencies"],
    "Projects": ["projects", "portfolio", "work samples"],
    "Certifications": ["certifications", "certificates", "credentials", "licenses"],
    "Contact": ["email", "phone", "linkedin", "contact"],
}


def _section_completeness(parsed: ParsedResume) -> tuple[float, list[str]]:
    """
    Check how many of the ideal resume sections are present.
    Returns (score_0_to_100, list_of_missing_section_names).
    """
    text_lower = parsed.clean_text.lower()
    contact_present = bool(parsed.emails or parsed.phones or parsed.linkedin_urls)

    found, missing_names = [], []
    for section_name, keywords in _IDEAL_SECTIONS.items():
        if section_name == "Contact":
            present = contact_present
        else:
            present = any(k in text_lower for k in keywords)
        if present:
            found.append(section_name)
        else:
            missing_names.append(section_name)

    score = round(len(found) / len(_IDEAL_SECTIONS) * 100, 1)
    return score, missing_names


# ── Formatting Audit ──────────────────────────────────────────────────────────

def _formatting_audit(parsed: ParsedResume) -> tuple[float, list[str], list[str]]:
    """
    Return (score_0_to_100, issues_list, positives_list).
    Deductions are applied for ATS-hostile formatting choices.
    """
    issues: list[str] = []
    positives: list[str] = []
    deductions = 0

    # Tables
    if parsed.has_tables:
        issues.append(
            "⚠️ Tables detected — many ATS systems cannot parse table content "
            "correctly, causing critical information to be missed. "
            "Convert tables to plain bullet points."
        )
        deductions += 20

    # Multi-column layout
    if parsed.has_multi_columns:
        issues.append(
            "⚠️ Multi-column layout detected — ATS parsers typically read "
            "left-to-right across the full page width, scrambling the order "
            "of content in side-by-side columns. Use a single-column layout."
        )
        deductions += 20

    # Image-based
    if parsed.is_image_based:
        issues.append(
            "🚫 Resume appears to be image-based or scanned — ATS systems "
            "cannot read images. Use a text-based PDF or DOCX."
        )
        deductions += 40

    # Very short resume
    if parsed.word_count < 200:
        issues.append(
            f"⚠️ Resume is very short ({parsed.word_count} words). "
            "ATS systems score on content density. Aim for 400–700 words."
        )
        deductions += 10
    elif parsed.word_count > 800:
        issues.append(
            f"ℹ️ Resume is long ({parsed.word_count} words). "
            "Consider trimming to 1–2 pages for most roles."
        )
        deductions += 5

    # Headers/footers
    if parsed.has_headers_footers:
        issues.append(
            "ℹ️ Possible headers/footers detected. "
            "Some ATS systems skip or duplicate header/footer text. "
            "Keep contact info in the main body of the document."
        )
        deductions += 5

    # Positives
    if not parsed.has_tables:
        positives.append("✅ No tables detected — good for ATS parsing.")
    if not parsed.has_multi_columns:
        positives.append("✅ Single-column layout — ATS-friendly structure.")
    if not parsed.is_image_based:
        positives.append("✅ Text-based document — fully parsable by ATS.")
    if 300 <= parsed.word_count <= 750:
        positives.append(f"✅ Good length ({parsed.word_count} words) — appropriate content density.")

    score = max(0.0, 100.0 - deductions)
    return score, issues, positives


# ── Contact Info Score ────────────────────────────────────────────────────────

def _contact_score(parsed: ParsedResume) -> float:
    score = 0.0
    if parsed.emails:
        score += 40
    if parsed.phones:
        score += 30
    if parsed.linkedin_urls:
        score += 20
    if parsed.name:
        score += 10
    return min(score, 100.0)


# ── Improvement Tips ──────────────────────────────────────────────────────────

def _build_tips(result: AnalysisResult, parsed: ParsedResume) -> list[str]:
    tips: list[str] = []

    if result.score.overall < 40:
        tips.append(
            "🔴 Your resume needs significant work to pass ATS screening. "
            "Focus on tailoring the content to the job description."
        )
    elif result.score.overall < 65:
        tips.append(
            "🟡 Your resume is a moderate ATS match. "
            "Adding missing keywords and fixing formatting will boost your score."
        )
    else:
        tips.append(
            "🟢 Your resume has a strong ATS match. "
            "Fine-tuning keyword placement and formatting can push it further."
        )

    if result.missing_keywords:
        top_missing = result.missing_keywords[:8]
        tips.append(
            f"🔑 Add these high-value JD keywords naturally into your resume: "
            f"{', '.join(top_missing)}."
        )

    if not parsed.emails:
        tips.append("📧 Add a professional email address to your contact section.")
    if not parsed.phones:
        tips.append("📞 Include a phone number for recruiters to reach you directly.")
    if not parsed.linkedin_urls:
        tips.append("🔗 Add your LinkedIn URL — many ATS systems and recruiters expect it.")

    if result.missing_sections:
        tips.append(
            f"📋 Consider adding these sections: "
            f"{', '.join(result.missing_sections)}."
        )

    if result.score.content_match < 40:
        tips.append(
            "📝 The language in your resume differs significantly from the job "
            "description. Mirror the JD's phrasing and terminology where appropriate."
        )

    if parsed.word_count < 250:
        tips.append(
            "📄 Expand your resume content — detail your achievements with "
            "quantifiable metrics (e.g., 'Reduced latency by 30%')."
        )

    return tips


# ── Main Analyzer ─────────────────────────────────────────────────────────────

def analyze(parsed: ParsedResume, job_description: str) -> AnalysisResult:
    """
    Run the full ATS analysis pipeline.

    Args:
        parsed:           Output from resume_parser.parse_resume()
        job_description:  Raw JD text from the user (will be sanitized here)

    Returns:
        AnalysisResult with scores, gaps, and recommendations.
    """
    result = AnalysisResult()

    if parsed.error:
        result.error = parsed.error
        return result

    # Sanitize JD
    jd_clean = sanitize_text_input(job_description)
    if len(jd_clean.strip()) < 30:
        result.error = "Job description is too short. Please provide more detail."
        return result

    resume_text = parsed.clean_text

    # ── Sub-scores ────────────────────────────────────────────────────────────
    content_sim = _cosine_match(resume_text, jd_clean)
    jd_kws, matched, missing = _keyword_gap(resume_text, jd_clean)
    keyword_cov = round(len(matched) / len(jd_kws) * 100, 1) if jd_kws else 0.0
    fmt_score, fmt_issues, fmt_positives = _formatting_audit(parsed)
    complete_score, missing_secs = _section_completeness(parsed)
    contact_score = _contact_score(parsed)

    # ── Populate result ───────────────────────────────────────────────────────
    result.score = ATSScore(
        content_match=content_sim,
        keyword_coverage=keyword_cov,
        formatting=fmt_score,
        completeness=complete_score,
        contact_info=contact_score,
    )
    result.score.compute_overall()

    result.jd_keywords = jd_kws
    result.matched_keywords = matched
    result.missing_keywords = missing
    result.resume_top_terms = _extract_top_ngrams(resume_text, top_n=20)

    result.formatting_issues = fmt_issues
    result.formatting_positives = fmt_positives
    result.missing_sections = missing_secs

    result.improvement_tips = _build_tips(result, parsed)

    # ── Verdict ───────────────────────────────────────────────────────────────
    score = result.score.overall
    if score >= 80:
        verdict = "Excellent match — your resume is well-optimised for this role."
    elif score >= 65:
        verdict = "Good match — a few tweaks could make your resume stand out."
    elif score >= 45:
        verdict = "Moderate match — targeted improvements are recommended."
    else:
        verdict = "Low match — significant tailoring is needed to pass ATS filters."
    result.verdict = verdict

    logger.info(
        "Analysis complete | overall=%.1f content=%.1f keywords=%.1f "
        "format=%.1f complete=%.1f contact=%.1f",
        score, content_sim, keyword_cov, fmt_score, complete_score, contact_score,
    )
    return result
