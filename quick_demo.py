#!/usr/bin/env python
"""
quick_start.py — ATS Resume Analyzer Quick Demo

This script demonstrates the core analysis pipeline in action.
No UI — just raw functionality for testing.

Usage: python quick_start.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from resume_parser import parse_resume
from analyzer import analyze
from security import sanitize_text_input

print("=" * 80)
print("🎯 ATS Resume Analyzer — Quick Demo")
print("=" * 80)
print()

# ── Demo Resume Text (as if parsed from a PDF) ────────────────────────────────
demo_resume = """
JOHN SMITH
john.smith@email.com | +1-555-123-4567 | linkedin.com/in/johnsmith | github.com/johnsmith

PROFESSIONAL SUMMARY
Full-stack software engineer with 5+ years of experience building scalable web 
applications using Python, React, and AWS. Strong background in system design and 
cloud architecture.

EXPERIENCE

Senior Software Engineer | Tech Corp, San Francisco, CA | 2021 - Present
• Architected and deployed microservices using Docker and Kubernetes, reducing 
  deployment time by 60%
• Led team of 3 engineers in redesigning legacy monolith to event-driven architecture
• Optimised database queries using indexing strategies, improving query performance 
  by 45%
• Implemented CI/CD pipeline using Jenkins and GitLab, reducing manual deployments 
  by 80%

Software Engineer | StartUp Inc, San Jose, CA | 2019 - 2021
• Developed RESTful APIs using Django and FastAPI serving 100K+ daily requests
• Built real-time data visualisation dashboard using React and D3.js
• Implemented authentication and authorization using OAuth 2.0 and JWT
• Collaborated with product team to design and ship 5 major features

Junior Developer | Agency LLC, Los Angeles, CA | 2018 - 2019
• Developed responsive web applications using React and TypeScript
• Integrated third-party APIs and payment gateways
• Mentored 2 interns in Python best practices and code review process

EDUCATION
Bachelor of Science in Computer Science | State University | 2018
GPA: 3.8/4.0 | Relevant Coursework: Algorithms, Databases, Systems Design

TECHNICAL SKILLS
Languages: Python, JavaScript, TypeScript, SQL, Bash
Backend: Django, FastAPI, Node.js, Flask, PostgreSQL, MongoDB
Frontend: React, Redux, HTML5, CSS3, Tailwind CSS
Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, Kubernetes, Jenkins, Terraform
Other: Git, REST APIs, GraphQL, System Design, Agile/Scrum

CERTIFICATIONS
AWS Certified Solutions Architect – Associate (2021)
Professional Scrum Master (PSM I) (2020)

PROJECTS
Open Source Contribution: Contributed 15+ commits to popular Python data library
Personal Portfolio: Full-stack project management app with React + Django (100 stars on GitHub)
"""

# ── Demo Job Description ──────────────────────────────────────────────────────
demo_jd = """
We are seeking a Senior Full-Stack Engineer to join our growing team.

About the Role:
You will architect and develop scalable cloud-native applications serving millions 
of users. You'll work with Python, JavaScript, and modern cloud infrastructure.

Responsibilities:
• Design and implement microservices using Python and FastAPI
• Build responsive frontend applications with React and TypeScript
• Deploy and manage applications on AWS using Docker and Kubernetes
• Establish best practices for code quality, testing, and documentation
• Mentor junior engineers and lead technical discussions
• Participate in on-call rotation for production support

Requirements:
• 4+ years of professional full-stack development experience
• Strong proficiency in Python and JavaScript/TypeScript
• Experience with relational databases (PostgreSQL, MySQL)
• Hands-on experience with AWS (EC2, S3, Lambda, RDS)
• Proficiency with Docker, Kubernetes, and CI/CD pipelines
• Experience with REST APIs and GraphQL
• Bachelor's degree in Computer Science or equivalent

Nice to Have:
• Experience with system design and architecture
• Open source contributions
• AWS or Kubernetes certifications
• Experience with machine learning frameworks

Benefits:
Competitive salary, health insurance, 401k, remote work options, professional 
development budget, stock options.
"""

# ── Parse Resume ───────────────────────────────────────────────────────────────
print("📄 Parsing resume...")
print("-" * 80)

# Create synthetic bytes (in real app, this comes from uploaded PDF/DOCX)
resume_bytes = demo_resume.encode("utf-8")
parsed = parse_resume(resume_bytes, "demo_resume.docx")

print(f"✅ Resume parsed successfully")
print(f"   • Words: {parsed.word_count:,}")
print(f"   • Name: {parsed.name or '(not detected)'}")
print(f"   • Emails: {', '.join(parsed.emails) if parsed.emails else '(none)'}")
print(f"   • Phones: {', '.join(parsed.phones) if parsed.phones else '(none)'}")
print(f"   • LinkedIn: {', '.join(parsed.linkedin_urls) if parsed.linkedin_urls else '(none)'}")
print(f"   • Skills detected: {len(parsed.skills)}")
print(f"   • Organizations: {', '.join(parsed.organizations[:3])}...")
print()

# ── Run Analysis ───────────────────────────────────────────────────────────────
print("🧠 Running ATS analysis...")
print("-" * 80)

result = analyze(parsed, demo_jd)

if result.error:
    print(f"❌ Error: {result.error}")
    sys.exit(1)

print(f"✅ Analysis complete")
print()

# ── Display Results ───────────────────────────────────────────────────────────
print("📊 RESULTS")
print("=" * 80)
print()

# Overall score
score = result.score.overall
print(f"🏆 Overall ATS Score: {score:.1f}%")
print(f"   Verdict: {result.verdict}")
print()

# Breakdown
print("📈 Score Breakdown:")
print(f"   • Content Match:        {result.score.content_match:>6.1f}%")
print(f"   • Keyword Coverage:     {result.score.keyword_coverage:>6.1f}%")
print(f"   • Formatting:           {result.score.formatting:>6.1f}%")
print(f"   • Completeness:         {result.score.completeness:>6.1f}%")
print(f"   • Contact Info:         {result.score.contact_info:>6.1f}%")
print()

# Keywords
print(f"✅ Matched Keywords: {len(result.matched_keywords)}/{len(result.jd_keywords)}")
if result.matched_keywords:
    print(f"   {', '.join(result.matched_keywords[:10])}...")
print()

print(f"❌ Missing Keywords: {len(result.missing_keywords)}")
if result.missing_keywords:
    print(f"   {', '.join(result.missing_keywords[:5])}...")
print()

# Tips
print("💡 Improvement Tips:")
for i, tip in enumerate(result.improvement_tips[:3], 1):
    print(f"   {i}. {tip}")
print()

# Formatting
print("⚠️ Formatting Issues:")
if result.formatting_issues:
    for issue in result.formatting_issues:
        print(f"   • {issue}")
else:
    print("   ✅ None detected")
print()

print("=" * 80)
print("✅ Demo complete! Ready to use the web app?")
print()
print("   streamlit run app.py")
print()
