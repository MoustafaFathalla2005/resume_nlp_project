<<<<<<< HEAD
"""
jd_generator.py
Generate a Job Description from a resume text.

Two modes:
    Rule-based (use_ai=False)  — fast, offline, keyword-driven
    AI-powered  (use_ai=True)  — calls Claude API for a richer, unique JD

The rule-based mode extracts skills found in the resume and fills a
structured template with role-specific content. The AI mode sends the
resume to Claude and returns whatever it writes — no repetition, no
copy-paste phrasing.
"""

=======
>>>>>>> e1aa699cdab3c7216389cd8e9a3ed0bcd3b14a69
import re
import random
import pandas as pd

from db_reader import (
    get_skills_db,
    get_category_title_map,
    get_summaries,
    get_responsibilities,
    get_offers,
)
from matcher import ResumeMatcher


class JobDescriptionGenerator:

    SAMPLE_LIMITS: dict[str, tuple[int, int]] = {
        "responsibilities": (3, 5),
        "offers"          : (2, 4),
        "skills"          : (3, 8),
    }

    def __init__(self, db_path: str = None, sample_limits: dict = None):
        self._db_path = db_path

<<<<<<< HEAD
    Methods
    generate_from_resume(resume_text, category) -> str  : main entry point
    extract_skills(text)                        -> list : matched skills
    extract_experience_years(text)              -> int  : max years found
    """

    def __init__(self, use_ai=False):
        """
        Parameters
        use_ai : bool, default False
            If True, call the Claude API. Falls back to rule-based on failure.
        """
        self.use_ai = use_ai

=======
        # per-instance override of sampling limits (falls back to class defaults)
        self._sample_limits = {**self.SAMPLE_LIMITS, **(sample_limits or {})}

        # ── lazy DB tables ─────────────────────────────────────────────────
        self._skills_db          = None
        self._category_title_map = None
        self._summaries          = None
        self._responsibilities   = None
        self._offers             = None

        self._category_matcher: ResumeMatcher | None = None
        self._summary_matcher:  ResumeMatcher | None = None
        self._resp_matcher:     ResumeMatcher | None = None
        self._offers_matcher:   ResumeMatcher | None = None
>>>>>>> e1aa699cdab3c7216389cd8e9a3ed0bcd3b14a69

    # ── DB properties (load once from file, then cached) ──────────────────

<<<<<<< HEAD
        Parameters
        resume_text : str       — raw or cleaned resume
        category    : str|None  — predicted job category (used to pick title)

        Returns
        str  — formatted job description
        """
        if self.use_ai:
            return self._generate_ai(resume_text, category)
        return self._generate_rule_based(resume_text, category)

    def extract_skills(self, text):
        """
        Find all skills from SKILLS_DB present in the resume text.

        Parameters
        text : str

        Returns
        list of str  — unique skill strings found, in order of discovery
        """
=======
    def _load_all(self):
        kw = {"filepath": self._db_path} if self._db_path else {}
        self._skills_db          = get_skills_db(**kw)
        self._category_title_map = get_category_title_map(**kw)
        self._summaries          = get_summaries(**kw)
        self._responsibilities   = get_responsibilities(**kw)
        self._offers             = get_offers(**kw)

    @property
    def skills_db(self) -> dict:
        if self._skills_db is None:
            self._load_all()
        return self._skills_db

    @property
    def category_title_map(self) -> dict:
        if self._category_title_map is None:
            self._load_all()
        return self._category_title_map

    @property
    def summaries(self) -> dict:
        if self._summaries is None:
            self._load_all()
        return self._summaries

    @property
    def responsibilities(self) -> dict:
        if self._responsibilities is None:
            self._load_all()
        return self._responsibilities

    @property
    def offers(self) -> dict:
        if self._offers is None:
            self._load_all()
        return self._offers

    # ── corpus builders ───────────────────────────────────────────────────

    def _build_category_corpus(self) -> pd.DataFrame:
        norm = {k.lower().replace("_", " "): v for k, v in self.skills_db.items()}
        rows = []
        for category in self.category_title_map:
            skills = (
                self.skills_db.get(category.lower().replace(" ", "_"))
                or norm.get(category.lower())
                or []
            )
            resps = self.responsibilities.get(
                category,
                self.responsibilities.get("_default", [])
            )
            doc = " ".join(skills) + " " + " ".join(resps)
            rows.append({"Category": category, "Resume": doc.strip()})
        return pd.DataFrame(rows)

    def _build_section_corpus(self, section: dict) -> pd.DataFrame:
        rows = []
        for key, value in section.items():
            text = value if isinstance(value, str) else " ".join(value)
            rows.append({"Category": key, "Resume": text})
        return pd.DataFrame(rows)

    # ── matcher properties (fit once, then cached) ────────────────────────

    @property
    def category_matcher(self) -> ResumeMatcher:
        if self._category_matcher is None:
            self._category_matcher = ResumeMatcher()
            self._category_matcher.fit(self._build_category_corpus(), text_col="Resume")
        return self._category_matcher

    @property
    def summary_matcher(self) -> ResumeMatcher:
        if self._summary_matcher is None:
            self._summary_matcher = ResumeMatcher()
            self._summary_matcher.fit(
                self._build_section_corpus(self.summaries), text_col="Resume"
            )
        return self._summary_matcher

    @property
    def resp_matcher(self) -> ResumeMatcher:
        if self._resp_matcher is None:
            self._resp_matcher = ResumeMatcher()
            self._resp_matcher.fit(
                self._build_section_corpus(self.responsibilities), text_col="Resume"
            )
        return self._resp_matcher

    @property
    def offers_matcher(self) -> ResumeMatcher:
        if self._offers_matcher is None:
            self._offers_matcher = ResumeMatcher()
            self._offers_matcher.fit(
                self._build_section_corpus(self.offers), text_col="Resume"
            )
        return self._offers_matcher

    # ── matcher-driven data fetchers ──────────────────────────────────────

    def _infer_category(self, resume_text: str, hint: str = None) -> str:
        if hint and hint in self.category_title_map:
            return hint
        result = self.category_matcher.match(resume_text, top_n=1)
        return (
            result.iloc[0]["Category"]
            if not result.empty
            else next(iter(self.category_title_map))
        )

    def _fetch_title(self, category: str) -> str:
        return self.category_title_map[category]

    def _fetch_summary(self, resume_text: str, category: str) -> str:
        if category in self.summaries:
            return self.summaries[category]
        result = self.summary_matcher.match(resume_text, top_n=1)
        key    = result.iloc[0]["Category"] if not result.empty else "_default"
        return self.summaries.get(key, "")

    def _fetch_responsibilities(self, resume_text: str, category: str) -> list:
        if category in self.responsibilities:
            pool = self.responsibilities[category]
        else:
            result = self.resp_matcher.match(resume_text, top_n=1)
            key    = result.iloc[0]["Category"] if not result.empty else "_default"
            pool   = self.responsibilities.get(key, [])
        return self._sample(pool, "responsibilities")

    def _fetch_offers(self, resume_text: str, category: str) -> list:
        if category in self.offers:
            pool = self.offers[category]
        else:
            result = self.offers_matcher.match(resume_text, top_n=1)
            key    = result.iloc[0]["Category"] if not result.empty else "_default"
            pool   = self.offers.get(key, [])
        return self._sample(pool, "offers")

    # ── public ────────────────────────────────────────────────────────────

    def generate_from_resume(self, resume_text: str, category: str = None) -> str:
        return self._generate_rule_based(resume_text, category)

    def extract_skills(self, text: str) -> list:
        """Extract skills using word-boundary matching to avoid false positives."""
>>>>>>> e1aa699cdab3c7216389cd8e9a3ed0bcd3b14a69
        text_lower = text.lower()
        found = []
        for skill_list in self.skills_db.values():
            for skill in skill_list:
                if len(skill) < 2:
                    continue
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower) and skill not in found:
                    found.append(skill)
        return self._sample(found, "skills")

<<<<<<< HEAD
    def extract_experience_years(self, text):
        """
        Parse the largest number of years mentioned in the resume.

        Parameters
        text : str

        Returns
        int  — maximum years found; defaults to 1 if none mentioned
        """
        matches = re.findall(r"(\d+)\+?\s*(years?|yrs?)", text.lower())
        return max((int(m[0]) for m in matches), default=1)

=======
    def extract_experience_years(self, text: str) -> int:
        """Return the largest year count mentioned; defaults to 1."""
        matches = re.findall(r"(\d+)\+?\s*(years?|yrs?)", text.lower())
        return max((int(m[0]) for m in matches), default=1)

    # ── private helpers ───────────────────────────────────────────────────
>>>>>>> e1aa699cdab3c7216389cd8e9a3ed0bcd3b14a69

    def _group_skills(self, skills: list) -> dict:
        """Group found skills by their DB domain."""
        groups = {}
        for domain, domain_skills in self.skills_db.items():
            matched = [s for s in skills if s in domain_skills]
            if matched:
                groups[domain.replace("_", " ").title()] = matched
        return groups

    def _sample(self, items: list, section: str) -> list:
        if not items:
            return items

        lo, hi = self._sample_limits.get(section, (3, len(items)))
        hi     = min(hi, len(items))

        if len(items) <= lo:
            return items

        k       = random.randint(lo, hi)
        indices = sorted(random.sample(range(len(items)), k))
        return [items[i] for i in indices]

    def _generate_rule_based(self, resume_text: str, category: str) -> str:
        category    = self._infer_category(resume_text, hint=category)
        title       = self._fetch_title(category)
        summary_tpl = self._fetch_summary(resume_text, category)
        resp_lines  = self._fetch_responsibilities(resume_text, category)
        offer_lines = self._fetch_offers(resume_text, category)

        skills = self.extract_skills(resume_text)
        years  = self.extract_experience_years(resume_text)

<<<<<<< HEAD
        summary_tpl = self._SUMMARIES.get(
            category,
            "We are looking for a talented {title} with {years}+ years of experience "
            "to join our growing engineering team."
        )
        summary = summary_tpl.format(title=title, years=years)

        
        resp_lines = self._RESPONSIBILITIES.get(category, self._RESPONSIBILITIES["_default"])
=======
        summary    = summary_tpl.format(title=title, years=years)
>>>>>>> e1aa699cdab3c7216389cd8e9a3ed0bcd3b14a69
        resp_block = "\n".join(f"  - {r}" for r in resp_lines)

        skill_groups = self._group_skills(skills)
        req_lines    = [f"  - {years}+ years of hands-on professional experience"]
        for domain, domain_skills in skill_groups.items():
            req_lines.append(f"  - {domain}: {', '.join(domain_skills)}")
        if not skill_groups:
            req_lines.append("  - Strong software engineering fundamentals")
        req_lines += [
            "  - Solid understanding of software design principles",
            "  - Experience working in agile / scrum teams",
        ]
<<<<<<< HEAD
        req_block = "\n".join(req_lines)

        offers      = self._OFFERS.get(category, self._OFFERS["_default"])
        offer_block = "\n".join(f"  - {o}" for o in offers)

        tech_stack = ", ".join(skills) if skills else "General technical skills"
=======
        req_block   = "\n".join(req_lines)
        offer_block = "\n".join(f"  - {o}" for o in offer_lines)
        tech_stack  = ", ".join(skills) if skills else "General technical skills"
>>>>>>> e1aa699cdab3c7216389cd8e9a3ed0bcd3b14a69

        sep = "─" * 50
        return (
            f"{sep}\n"
            f"JOB DESCRIPTION: {title}\n"
            f"{sep}\n\n"
            f"SUMMARY\n{summary}\n\n"
            f"RESPONSIBILITIES\n{resp_block}\n\n"
            f"REQUIREMENTS\n{req_block}\n\n"
            f"TECH STACK\n  {tech_stack}\n\n"
            f"WHAT WE OFFER\n{offer_block}\n"
            f"{sep}"
        )

<<<<<<< HEAD

    def _generate_ai(self, resume_text, category):
        """Call Claude API to write a unique JD. Falls back to rule-based on error."""
        try:
            cat_hint = f"Predicted category: {category}." if category else ""
            prompt = (
                "You are an expert technical recruiter.\n"
                "Write a professional job description for the role this resume fits best.\n"
                f"{cat_hint}\n\n"
                "Include: Job Title, Summary (2-3 sentences), Responsibilities "
                "(5 bullet points), Requirements (5 bullet points), Tech Stack, "
                "and Benefits. Be specific — do not repeat any phrase twice.\n\n"
                f"Resume:\n{resume_text[:3000]}"
            )
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model"     : "claude-sonnet-4-20250514",
                    "max_tokens": 1500,
                    "messages"  : [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data   = resp.json()
            blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            return "\n".join(blocks).strip()

        except Exception as exc:
            print(f"[JDGenerator] AI call failed ({exc}). Using rule-based fallback.")
            return self._generate_rule_based(resume_text, category)

=======
>>>>>>> e1aa699cdab3c7216389cd8e9a3ed0bcd3b14a69
    def __repr__(self):
        return f"JobDescriptionGenerator()"