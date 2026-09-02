import os
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, Any, List, Tuple

# Configuration
RESUMES_DIR = os.path.join("data", "resumes")
JOB_DESCS_DIR = os.path.join("data", "job_descriptions")
PROCESSED_DIR = os.path.join("data", "processed")

class RecruitmentAgent:
    """Recruitment agent that matches candidates to job descriptions using skill embeddings."""
    
    def __init__(self):
        """Initialize the recruitment agent with embedding model."""
        print("Loading sentence transformer model for recruitment matching...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load existing skills for matching (reuse from skill-gap engine)
        self.role_skills = pd.read_csv(os.path.join(PROCESSED_DIR, "role_skills.csv"))
        self.skill_embeddings = self._embed_existing_skills()
        
    def _embed_existing_skills(self) -> Dict[str, any]:
        """Pre-embed existing skills from role_skills for matching."""
        unique_skills = self.role_skills["Skill Name"].unique().tolist()
        embeddings = self.model.encode(unique_skills)
        
        return {
            "skills": unique_skills,
            "embeddings": embeddings
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF resume."""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""
        
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers (simple heuristic)
        text = re.sub(r'\d+\s*\n', '\n', text)
        return text.strip()
    
    def parse_resume(self, resume_path: str) -> Dict[str, Any]:
        """Parse a resume and extract key information."""
        text = self.extract_text_from_pdf(resume_path)
        
        if not text:
            return {"error": f"Could not extract text from {resume_path}"}
        
        # Extract candidate name (simplified - first line or common patterns)
        lines = text.split('\n')
        candidate_name = lines[0] if lines else "Unknown"
        
        # Extract skills by matching against known skills
        extracted_skills = self._extract_skills_from_text(text)
        
        return {
            "resume_file": os.path.basename(resume_path),
            "candidate_name": candidate_name,
            "full_text": text,
            "extracted_skills": extracted_skills,
            "skill_count": len(extracted_skills)
        }
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text by matching against known skills."""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skill_embeddings["skills"]:
            skill_lower = skill.lower()
            # Check if skill appears in text (with word boundaries)
            if re.search(r'\b' + re.escape(skill_lower) + r'\b', text_lower):
                found_skills.append(skill)
        
        return found_skills
    
    def parse_job_description(self, job_desc_path: str) -> Dict[str, Any]:
        """Parse a job description and extract key information."""
        try:
            with open(job_desc_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            return {"error": f"Could not read {job_desc_path}: {e}"}
        
        text = self._clean_text(text)
        
        # Extract job title (first line or from content)
        lines = text.split('\n')
        job_title = lines[0] if lines else "Unknown"
        
        # Extract required skills
        required_skills = self._extract_skills_from_text(text)
        
        return {
            "job_file": os.path.basename(job_desc_path),
            "job_title": job_title,
            "full_text": text,
            "required_skills": required_skills,
            "skill_count": len(required_skills)
        }
    
    def calculate_match_score(self, candidate_skills: List[str], 
                             job_skills: List[str]) -> Tuple[float, List[str]]:
        """
        Calculate match score between candidate and job description.
        Returns (score, matched_skills).
        """
        if not job_skills:
            return 0.0, []
        
        # Calculate Jaccard similarity (intersection over union)
        candidate_set = set(candidate_skills)
        job_set = set(job_skills)
        
        intersection = candidate_set.intersection(job_set)
        union = candidate_set.union(job_set)
        
        if not union:
            return 0.0, []
        
        jaccard_score = len(intersection) / len(union)
        
        # Also calculate coverage (how many job skills the candidate has)
        coverage_score = len(intersection) / len(job_set) if job_set else 0.0
        
        # Combined score: 60% coverage, 40% Jaccard
        combined_score = 0.6 * coverage_score + 0.4 * jaccard_score
        
        return combined_score, list(intersection)
    
    def match_candidate_to_jobs(self, resume_path: str) -> Dict[str, Any]:
        """Match a single candidate to all available job descriptions."""
        # Parse candidate resume
        candidate = self.parse_resume(resume_path)
        
        if "error" in candidate:
            return candidate
        
        # Get all job descriptions
        job_files = [f for f in os.listdir(JOB_DESCS_DIR) if f.endswith('.txt')]
        
        matches = []
        for job_file in job_files:
            job_path = os.path.join(JOB_DESCS_DIR, job_file)
            job = self.parse_job_description(job_path)
            
            if "error" in job:
                continue
            
            # Calculate match score
            score, matched_skills = self.calculate_match_score(
                candidate["extracted_skills"],
                job["required_skills"]
            )
            
            matches.append({
                "job_title": job["job_title"],
                "job_file": job_file,
                "match_score": score,
                "matched_skills": matched_skills,
                "candidate_skills": candidate["extracted_skills"],
                "required_skills": job["required_skills"]
            })
        
        # Sort by match score
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        
        return {
            "candidate_name": candidate["candidate_name"],
            "resume_file": candidate["resume_file"],
            "candidate_skills": candidate["extracted_skills"],
            "skill_count": candidate["skill_count"],
            "job_matches": matches
        }
    
    def match_all_candidates(self) -> Dict[str, Any]:
        """Match all candidates to all job descriptions."""
        # Get all resume files
        resume_files = [f for f in os.listdir(RESUMES_DIR) if f.endswith('.pdf')]
        
        all_matches = []
        for resume_file in resume_files:
            resume_path = os.path.join(RESUMES_DIR, resume_file)
            candidate_match = self.match_candidate_to_jobs(resume_path)
            
            if "error" not in candidate_match:
                all_matches.append(candidate_match)
        
        return {
            "total_candidates": len(all_matches),
            "matches": all_matches
        }

# Global agent instance
_recruitment_agent = None

def get_recruitment_agent() -> RecruitmentAgent:
    """Get or create the recruitment agent singleton."""
    global _recruitment_agent
    if _recruitment_agent is None:
        _recruitment_agent = RecruitmentAgent()
    return _recruitment_agent
