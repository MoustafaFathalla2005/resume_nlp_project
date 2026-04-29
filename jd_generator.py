"""
jd_generator.py
---------------
Generate a Job Description from a resume text.

Two modes:
    Rule-based (use_ai=False)  — fast, offline, keyword-driven
    AI-powered  (use_ai=True)  — calls Claude API for a richer, unique JD

The rule-based mode extracts skills found in the resume and fills a
structured template with role-specific content. The AI mode sends the
resume to Claude and returns whatever it writes — no repetition, no
copy-paste phrasing.
"""

import re
import requests


# ── skills database ──────────────────────────────────────────────────────────
SKILLS_DB = {
    "data_science"     : ["python", "r", "machine learning", "deep learning",
                          "tensorflow", "pytorch", "scikit-learn", "pandas",
                          "numpy", "statistics", "nlp", "computer vision",
                          "sql", "tableau", "spark"],
    "web"              : ["javascript", "typescript", "react", "angular",
                          "vue", "node", "html", "css", "webpack", "graphql", "redux"],
    "backend"          : ["python", "java", "django", "flask", "fastapi",
                          "spring", "microservices", "docker", "kubernetes",
                          "aws", "azure", "postgresql", "mysql", "mongodb",
                          "redis", "kafka"],
    "devops"           : ["docker", "kubernetes", "jenkins", "terraform",
                          "ansible", "aws", "azure", "linux", "bash",
                          "prometheus", "grafana"],
    "java"             : ["java", "spring", "hibernate", "maven", "gradle",
                          "junit", "jpa"],
    "testing"          : ["selenium", "pytest", "junit", "cucumber",
                          "jira", "postman"],
}

# map predicted category → job title shown in the JD
CATEGORY_TITLE_MAP = {
    "Data Science"             : "Senior Data Scientist / ML Engineer",
    "Python Developer"         : "Python Software Engineer",
    "Java Developer"           : "Java Backend Engineer",
    "Web Designing"            : "Frontend / Full-Stack Developer",
    "DevOps Engineer"          : "DevOps / Platform Engineer",
    "Network Security Engineer": "Cybersecurity Engineer",
    "Database"                 : "Database Engineer / DBA",
    "DotNet Developer"         : ".NET Software Engineer",
    "Blockchain"               : "Blockchain Developer",
    "Testing"                  : "QA Automation Engineer",
    "Automation Testing"       : "SDET / Automation Engineer",
    "ETL Developer"            : "Data / ETL Engineer",
    "Hadoop"                   : "Big Data Engineer",
    "HR"                       : "HR Business Partner",
    "Sales"                    : "Sales Account Executive",
    "Mechanical Engineer"      : "Mechanical Design Engineer",
    "Civil Engineer"           : "Civil / Structural Engineer",
    "Electrical Engineering"   : "Electrical Systems Engineer",
}


class JobDescriptionGenerator:
    """
    Build a job description from a resume string.

    Rule-based mode extracts skills + years of experience from the resume
    text and assembles a structured JD with role-specific content.

    AI mode sends the resume to the Claude API and returns whatever
    Claude writes — guaranteed to be unique and non-repetitive.

    Methods
    -------
    generate_from_resume(resume_text, category) -> str  : main entry point
    extract_skills(text)                        -> list : matched skills
    extract_experience_years(text)              -> int  : max years found
    """

    def __init__(self, use_ai=False):
        """
        Parameters
        ----------
        use_ai : bool, default False
            If True, call the Claude API. Falls back to rule-based on failure.
        """
        self.use_ai = use_ai

    # ── public ──────────────────────────────────────────────────────────────

    def generate_from_resume(self, resume_text, category=None):
        """
        Generate a job description.

        Parameters
        ----------
        resume_text : str       — raw or cleaned resume
        category    : str|None  — predicted job category (used to pick title)

        Returns
        -------
        str  — formatted job description
        """
        if self.use_ai:
            return self._generate_ai(resume_text, category)
        return self._generate_rule_based(resume_text, category)

    def extract_skills(self, text):
        """
        Find all skills from SKILLS_DB present in the resume text.

        Parameters
        ----------
        text : str

        Returns
        -------
        list of str  — unique skill strings found, in order of discovery
        """
        text_lower = text.lower()
        found = []
        for skill_list in SKILLS_DB.values():
            for skill in skill_list:
                if skill in text_lower and skill not in found:
                    found.append(skill)
        return found

    def extract_experience_years(self, text):
        """
        Parse the largest number of years mentioned in the resume.

        Parameters
        ----------
        text : str

        Returns
        -------
        int  — maximum years found; defaults to 1 if none mentioned
        """
        matches = re.findall(r"(\d+)\+?\s*(years?|yrs?)", text.lower())
        return max((int(m[0]) for m in matches), default=1)

    # ── private rule-based ───────────────────────────────────────────────────

    # role-specific summaries (one per category)
    _SUMMARIES = {
        "Data Science":
            "We are looking for a data-driven {title} who turns raw data into "
            "products. You will own the full ML lifecycle: data, experiments, "
            "models in production, and monitoring.",
        "Python Developer":
            "We need a sharp {title} who writes clean, tested Python and cares "
            "about code quality, performance, and maintainability.",
        "Java Developer":
            "We are hiring a {title} to build high-throughput backend services. "
            "You will own microservices, APIs, and data pipelines in a "
            "cloud-native environment.",
        "Web Designing":
            "We want a creative {title} who turns designs into fast, accessible "
            "web experiences — from wireframe to production CSS.",
        "DevOps Engineer":
            "We need a {title} who lives and breathes automation: CI/CD, "
            "infrastructure-as-code, and observability from day one.",
        "Testing":
            "We are looking for a quality-obsessed {title} to build test suites "
            "that give the whole team confidence to ship at speed.",
        "HR":
            "We are looking for a people-first {title} to drive talent "
            "acquisition, onboarding, and employee engagement.",
    }

    _RESPONSIBILITIES = {
        "Data Science": [
            "Design, train, and evaluate ML/DL models end-to-end",
            "Build and maintain feature pipelines and data workflows",
            "Collaborate with product and engineering on experiment design",
            "Monitor model drift and own retraining cycles",
            "Document findings and present results to stakeholders",
        ],
        "Python Developer": [
            "Write clean, tested, and well-documented Python code",
            "Design and maintain REST APIs and backend services",
            "Review pull requests and mentor junior developers",
            "Identify and fix performance bottlenecks",
            "Participate in system design discussions",
        ],
        "Java Developer": [
            "Build and maintain high-performance Java microservices",
            "Design RESTful APIs consumed by web and mobile clients",
            "Write unit and integration tests with high coverage",
            "Optimise database queries and caching strategies",
        ],
        "DevOps Engineer": [
            "Design and maintain CI/CD pipelines for multiple teams",
            "Manage cloud infrastructure using Terraform / Ansible",
            "Own observability stack: metrics, logs, and alerting",
            "Harden security posture and manage access controls",
        ],
        "_default": [
            "Design, develop, and deliver features in a cross-functional team",
            "Write clean, well-tested, and documented code",
            "Participate in architecture discussions and code reviews",
            "Mentor junior engineers and share technical knowledge",
        ],
    }

    _OFFERS = {
        "Data Science": [
            "Access to cutting-edge GPU compute",
            "Conference and publication budget",
            "Flexible remote / hybrid schedule",
            "Competitive salary and equity",
        ],
        "_default": [
            "Competitive, market-benchmarked salary and annual bonus",
            "Flexible remote / hybrid work",
            "Annual learning and development budget",
            "Health, dental, and vision coverage",
        ],
    }

    def _pick_title(self, skills, category):
        if category and category in CATEGORY_TITLE_MAP:
            return CATEGORY_TITLE_MAP[category]
        skill_set = set(skills)
        if skill_set & {"machine learning", "deep learning", "tensorflow"}:
            return "Data Scientist / ML Engineer"
        if skill_set & {"react", "angular", "vue", "javascript"}:
            return "Frontend / Full-Stack Developer"
        if skill_set & {"django", "flask", "fastapi", "spring"}:
            return "Backend Software Engineer"
        if skill_set & {"docker", "kubernetes", "terraform"}:
            return "DevOps / Platform Engineer"
        return "Software Engineer"

    def _group_skills(self, skills):
        """Group found skills by their SKILLS_DB domain."""
        groups = {}
        for domain, domain_skills in SKILLS_DB.items():
            matched = [s for s in skills if s in domain_skills]
            if matched:
                groups[domain.replace("_", " ").title()] = matched
        return groups

    def _generate_rule_based(self, resume_text, category):
        skills = self.extract_skills(resume_text)
        years  = self.extract_experience_years(resume_text)
        title  = self._pick_title(skills, category)

        # summary
        summary_tpl = self._SUMMARIES.get(
            category,
            "We are looking for a talented {title} with {years}+ years of experience "
            "to join our growing engineering team."
        )
        summary = summary_tpl.format(title=title, years=years)

        # responsibilities
        resp_lines = self._RESPONSIBILITIES.get(category, self._RESPONSIBILITIES["_default"])
        resp_block = "\n".join(f"  - {r}" for r in resp_lines)

        # requirements — experience + grouped skills + generic lines
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
        req_block = "\n".join(req_lines)

        # what we offer
        offers      = self._OFFERS.get(category, self._OFFERS["_default"])
        offer_block = "\n".join(f"  - {o}" for o in offers)

        # tech stack
        tech_stack = ", ".join(skills) if skills else "General technical skills"

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

    # ── AI mode ─────────────────────────────────────────────────────────────

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
                    "max_tokens": 1000,
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

    def __repr__(self):
        return f"JobDescriptionGenerator(use_ai={self.use_ai})"
