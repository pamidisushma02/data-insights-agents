
"""
Resume Screening with LangSmith - Using Google Colab Secrets
"""

# ============================================================================
# CELL 1: Install Packages
# ============================================================================
!pip install -q langchain langchain-google-genai langsmith
# print("✅ Installation complete!")

# ============================================================================
# CELL 2: Import Libraries and Load Secrets
# ============================================================================

import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client, traceable
from google.colab import userdata

print("📦 All imports successful!")
print("\n🔑 Loading API keys from Colab secrets...")

# Load API keys from Colab secrets
# Secret names in your Colab should be:
# - GOOGLE_API_KEY
# - LANGCHAIN_API_KEY

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
LANGCHAIN_API_KEY = userdata.get('LANGCHAIN_API_KEY')
print("✅ API keys loaded from secrets")


LANGCHAIN_PROJECT = "resume-screening-demo"

# Configure environment
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

print("✅ Environment configured successfully!")
print(f"📊 Project Name: {LANGCHAIN_PROJECT}")

# ============================================================================
# CELL 3: Initialize Clients
# ============================================================================

# Initialize LangSmith Client
try:
    langsmith_client = Client()
    print("✅ LangSmith client initialized")
except Exception as e:
    print(f"❌ Error initializing LangSmith: {e}")
    print("Check your LANGCHAIN_API_KEY secret")

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )
    print("✅ Google Gemini LLM initialized")
    print("\n🎉 All systems ready!")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    print("Check your GOOGLE_API_KEY secret")

# ============================================================================
# CELL 4: Define Traceable Functions
# ============================================================================

@traceable(name="extract_resume_text")
def extract_resume_text(resume_content: str) -> dict:
    """
    Extracts and processes resume text
    Returns structured data with metadata
    """
    return {
        "text": resume_content,
        "text_length": len(resume_content),
        "timestamp": datetime.now().isoformat()
    }

@traceable(name="analyze_resume_complete")
def analyze_resume(job_requirements: str, resume_text: str) -> dict:
    """
    Analyzes resume against job requirements using LLM
    Returns analysis with suitability score
    """
    prompt = f"""
    Analyze the following resume against the job requirements and provide a suitability score (0-100).

    Job Requirements:
    {job_requirements}

    Resume:
    {resume_text}

    Provide your analysis in the following format:
    1. Skills Match: [Your analysis]
    2. Experience Relevance: [Your analysis]
    3. Suitability Score: [0-100]

    Be specific and concise.
    """

    # Call LLM (automatically traced by LangChain)
    response = llm.invoke(prompt)
    analysis = response.content

    # Extract score (basic parsing)
    score = 50  # default
    for line in analysis.split('\n'):
        if 'Suitability Score:' in line or 'Score:' in line:
            try:
                # Extract score (IMPROVED - FIXES THE 3/100 BUG)
                import re
                score = 50  # default

                # Look for patterns like "Score: 95" or "Suitability Score: 95"
                score_patterns = [
                    r'Suitability Score:\s*(\d+)',
                    r'Score:\s*(\d+)',
                    r'score:\s*(\d+)',
                    r'suitability score:\s*(\d+)'
                ]

                for pattern in score_patterns:
                    matches = re.findall(pattern, analysis, re.IGNORECASE)
                    if matches:
                        score = int(matches[0])
                        # Ensure score is between 0-100
                        score = max(0, min(100, score))
                        break
            except:
                pass

    return {
        "analysis": analysis,
        "suitability_score": score,
        "job_requirements_length": len(job_requirements),
        "analysis_length": len(analysis),
        "processing_metadata": {
            "model": "gemini-2.0-flash",
            "temperature": 0.2,
            "timestamp": datetime.now().isoformat()
        }
    }

@traceable(name="process_resume_screening")
def process_resume_screening(job_requirements: str, resume_content: str) -> dict:
    """
    Main orchestration function for complete resume screening workflow
    """
    # Step 1: Extract text
    resume_data = extract_resume_text(resume_content)

    # Step 2: Analyze resume
    analysis_result = analyze_resume(job_requirements, resume_data["text"])

    # Step 3: Return complete results
    return {
        "status": "success",
        "resume_data": resume_data,
        "analysis": analysis_result["analysis"],
        "suitability_score": analysis_result["suitability_score"],
        "metadata": analysis_result["processing_metadata"]
    }

print("✅ All functions defined successfully!")
print("📋 Functions ready:")
print("   - extract_resume_text()")
print("   - analyze_resume()")
print("   - process_resume_screening()")

# ============================================================================
# CELL 5: Run Single Demo Test
# ============================================================================

# Sample data
job_requirements = """
Required: Python, Machine Learning, 3+ years experience
Preferred: TensorFlow, NLP, Deep Learning
"""

resume_content = """
John Doe
Software Engineer

Skills: Python, Machine Learning, TensorFlow, Deep Learning

Experience:
- 4 years as ML Engineer at Tech Corp
- Developed NLP models for text classification
- Built recommendation systems using Python and TensorFlow

Education: MS in Computer Science
"""

print("🚀 Starting Resume Screening with LangSmith Observability...")
print(f"📊 Project: {LANGCHAIN_PROJECT}")
print("=" * 70)

# Process the resume
result = process_resume_screening(job_requirements, resume_content)

print("\n✅ Analysis Complete!")
print("=" * 70)
print(f"\n📈 Suitability Score: {result['suitability_score']}/100")
print("\n📝 Detailed Analysis:")
print("-" * 70)
print(result['analysis'])
print("-" * 70)
print("\n💾 Metadata:")
print(f"   Resume Length: {result['resume_data']['text_length']} characters")
print(f"   Analysis Length: {len(result['analysis'])} characters")
print(f"   Timestamp: {result['metadata']['timestamp']}")
print("=" * 70)
print("\n🔍 View detailed traces in LangSmith Dashboard:")
print("   👉 https://smith.langchain.com")
print("   📁 Project: resume-screening-demo")
