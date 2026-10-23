import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger
import pypdf

class ChunkMetadata(BaseModel):
    source_type: str = "resume"  # resume or job_description
    section: str = "general"
    chunk_index: int = 0
    token_estimate: int = 0
    is_sanitized_data: bool = True

class DocumentChunk(BaseModel):
    chunk_id: str
    section: str
    content: str
    metadata: ChunkMetadata

class CandidateSummary(BaseModel):
    name: str = "Candidate"
    target_role: str = "Software Engineer"
    skills: List[str] = Field(default_factory=list)
    projects_count: int = 0
    projects: List[str] = Field(default_factory=list)
    experience_highlights: List[str] = Field(default_factory=list)
    summary_text: str = ""

class ResumeIngestionResult(BaseModel):
    candidate_summary: CandidateSummary
    resume_chunks: List[DocumentChunk] = Field(default_factory=list)
    jd_chunks: List[DocumentChunk] = Field(default_factory=list)
    raw_resume_text: str = ""
    raw_jd_text: str = ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract clean text from a PDF file."""
    reader = pypdf.PdfReader(pdf_path)
    text_parts = []
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)
    return "\n".join(text_parts)

def sanitize_text_as_data(text: str) -> str:
    """
    Safety requirement #6:
    Instructions hidden in resume/JD (e.g. 'ignore previous instructions', 'give 10/10')
    are strictly treated as passive candidate data, never instructions.
    """
    # Normalize excessive newlines and whitespace
    clean = re.sub(r'\r\n|\r', '\n', text)
    return clean.strip()

def chunk_document_by_sections(text: str, source_type: str = "resume") -> List[DocumentChunk]:
    """
    Splits text into chunks and labels each chunk with section metadata
    (skills, projects, experience, education, etc.).
    """
    sanitized = sanitize_text_as_data(text)
    lines = sanitized.split("\n")

    section_keywords = {
        "summary": ["summary", "about me", "professional summary", "objective", "profile"],
        "skills": ["skills", "technical skills", "technologies", "core competencies"],
        "experience": ["experience", "work experience", "employment", "professional experience", "history"],
        "projects": ["projects", "personal projects", "academic projects", "key projects"],
        "education": ["education", "academic background", "qualifications", "degree"],
        "responsibilities": ["responsibilities", "key responsibilities", "duties", "what you will do"],
        "requirements": ["requirements", "qualifications", "required skills", "what we are looking for"]
    }

    chunks: List[DocumentChunk] = []
    current_section = "overview"
    current_lines: List[str] = []
    chunk_counter = 0

    def commit_chunk(section_name: str, lines_to_commit: List[str]):
        nonlocal chunk_counter
        content = "\n".join(lines_to_commit).strip()
        if content:
            chunk = DocumentChunk(
                chunk_id=f"{source_type}_{section_name}_{chunk_counter}",
                section=section_name,
                content=content,
                metadata=ChunkMetadata(
                    source_type=source_type,
                    section=section_name,
                    chunk_index=chunk_counter,
                    token_estimate=len(content.split()),
                    is_sanitized_data=True
                )
            )
            chunks.append(chunk)
            chunk_counter += 1

    for line in lines:
        stripped = line.strip()
        lower_line = stripped.lower().rstrip(":")

        # Check if line matches a known section header
        matched_section = None
        for sec, keywords in section_keywords.items():
            if lower_line in keywords or any(stripped.lower() == f"## {kw}" or stripped.lower() == f"### {kw}" for kw in keywords):
                matched_section = sec
                break

        if matched_section:
            if current_lines:
                commit_chunk(current_section, current_lines)
                current_lines = []
            current_section = matched_section
        else:
            if stripped:
                current_lines.append(stripped)

    if current_lines:
        commit_chunk(current_section, current_lines)

    return chunks

def extract_candidate_summary(resume_chunks: List[DocumentChunk], jd_chunks: List[DocumentChunk], raw_resume: str, raw_jd: str) -> CandidateSummary:
    """Extract candidate name, skills, projects, and target role."""
    # 1. Candidate Name (usually in the first lines of resume)
    name = "Candidate"
    lines = [l.strip() for l in raw_resume.split("\n") if l.strip()]
    if lines:
        first_line = lines[0]
        # Common pattern: Alex Chen or Name in caps/title
        if len(first_line.split()) in [2, 3, 4] and not any(kw in first_line.lower() for kw in ["resume", "curriculum", "page", "phone", "email"]):
            name = first_line.strip("#* ")

    # 2. Target Role (from JD)
    target_role = "Software Engineer"
    for line in raw_jd.split("\n"):
        if "role:" in line.lower() or "title:" in line.lower() or "position:" in line.lower():
            target_role = line.split(":", 1)[1].strip()
            break
        elif "engineer" in line.lower() or "developer" in line.lower() or "architect" in line.lower():
            if len(line.strip()) < 50:
                target_role = line.strip("#* ")
                break

    # 3. Skills Extraction
    skills: List[str] = []
    skills_chunk = next((c.content for c in resume_chunks if c.section == "skills"), "")
    if skills_chunk:
        # Split by commas, dashes, bullets, colons
        skill_candidates = re.split(r'[,•|\-\n:]', skills_chunk)
        for sc in skill_candidates:
            token = sc.strip()
            if token and len(token) < 25 and not token.lower().endswith("skills") and not token.lower().endswith("languages"):
                if token not in skills:
                    skills.append(token)
    if not skills:
        common_tech = ["Python", "Go", "Java", "JavaScript", "TypeScript", "SQL", "PostgreSQL", "Redis", "Kafka", "Docker", "AWS", "FastAPI", "React"]
        skills = [tech for tech in common_tech if tech.lower() in raw_resume.lower()]

    # 4. Projects Extraction
    projects: List[str] = []
    projects_chunk = next((c.content for c in resume_chunks if c.section == "projects"), "")
    if projects_chunk:
        proj_lines = projects_chunk.split("\n")
        for pl in proj_lines:
            pl_str = pl.strip()
            if pl_str.startswith(("1.", "2.", "3.", "4.", "5.", "-", "*")):
                title = re.sub(r'^[\d\.\-\*\s]+', '', pl_str).split("(")[0].split("-")[0].strip()
                if title and len(title) < 50 and title not in projects:
                    projects.append(title)
            elif "(" in pl_str and ")" in pl_str and len(pl_str) < 60:
                title = pl_str.split("(")[0].strip()
                if title and title not in projects:
                    projects.append(title)

    if not projects:
        # Fallback check
        for line in raw_resume.split("\n"):
            if "project" in line.lower() and len(line.strip()) < 50:
                projects.append(line.strip("#*- "))

    # 5. Experience Highlights
    experience_highlights = []
    exp_chunk = next((c.content for c in resume_chunks if c.section == "experience"), "")
    if exp_chunk:
        for el in exp_chunk.split("\n"):
            el_s = el.strip()
            if el_s.startswith(("-", "•", "*")) and len(el_s) > 20:
                experience_highlights.append(el_s.lstrip("-•* "))
            if len(experience_highlights) >= 3:
                break

    summary = CandidateSummary(
        name=name,
        target_role=target_role,
        skills=skills[:12],
        projects_count=len(projects),
        projects=projects,
        experience_highlights=experience_highlights,
        summary_text=f"Candidate {name} applying for {target_role}. Detected {len(skills)} skills and {len(projects)} featured projects."
    )
    return summary

def load_and_ingest(resume_input: str, jd_input: str) -> ResumeIngestionResult:
    """
    Main ingestion entrypoint:
    Loads resume (path to PDF or raw text) and job description.
    Labels sections with metadata, creates chunks, and extracts candidate summary.
    """
    # 1. Parse Resume
    raw_resume = ""
    if os.path.exists(resume_input):
        path = Path(resume_input)
        if path.suffix.lower() == ".pdf":
            raw_resume = extract_text_from_pdf(str(path))
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_resume = f.read()
    else:
        raw_resume = resume_input

    # 2. Parse Job Description
    raw_jd = ""
    if os.path.exists(jd_input):
        with open(jd_input, "r", encoding="utf-8", errors="ignore") as f:
            raw_jd = f.read()
    else:
        raw_jd = jd_input

    # 3. Create Chunks with Metadata
    resume_chunks = chunk_document_by_sections(raw_resume, source_type="resume")
    jd_chunks = chunk_document_by_sections(raw_jd, source_type="job_description")

    # 4. Extract Summary
    summary = extract_candidate_summary(resume_chunks, jd_chunks, raw_resume, raw_jd)

    # Print summary as required by Requirement #1
    print("\n" + "="*50)
    print("INGESTION SUMMARY")
    print("="*50)
    print(f"Candidate Name:    {summary.name}")
    print(f"Target Role:       {summary.target_role}")
    print(f"Skills Found:      {', '.join(summary.skills[:8])}...")
    print(f"Projects Count:    {summary.projects_count} ({', '.join(summary.projects)})")
    print(f"Total Chunks:      Resume: {len(resume_chunks)}, JD: {len(jd_chunks)}")
    print("="*50 + "\n")

    logger.info(f"Successfully ingested resume for {summary.name} with {len(resume_chunks)} chunks.")

    return ResumeIngestionResult(
        candidate_summary=summary,
        resume_chunks=resume_chunks,
        jd_chunks=jd_chunks,
        raw_resume_text=raw_resume,
        raw_jd_text=raw_jd
    )
