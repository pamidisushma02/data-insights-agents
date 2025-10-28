

# ============================================
# Installation
# ============================================
!pip install -q crewai crewai-tools litellm

print("✅ Installation complete!")

# ============================================
# Simple Single Agent Example
# ============================================

from crewai import Agent, Task, Crew, LLM
import os
import time
from getpass import getpass

# Get Groq API Key
print("🔑 Enter your Groq API Key")
print("Get one free at: https://console.groq.com/keys")
groq_api_key = getpass("Groq API Key: ")

# Set environment variable
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize Groq LLM
llm = LLM(
    model="groq/llama-3.1-8b-instant",  # Fast model, low token usage
    temperature=0.7,
    max_tokens=1000,
    api_key=groq_api_key
)

print("✅ LLM initialized!")

# Create a simple research agent
researcher = Agent(
    role="Research Analyst",
    goal="Research topics and provide clear, concise insights",
    backstory="You are a skilled researcher who provides accurate information.",
    verbose=True,
    llm=llm
)

# Create a simple task
research_task = Task(
    description="""Research the topic: {topic}

    Provide a brief overview in 3-4 sentences.
    Focus on the most important points.""",
    agent=researcher,
    expected_output="A concise 3-4 sentence overview of the topic"
)

# Create crew with single agent
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

# Execute the task
print("\n🚀 Starting research...")
print("=" * 50)

result = crew.kickoff(
    inputs={
        "topic": "What is artificial intelligence and how does it work?"
    }
)

print("\n" + "=" * 50)
print("📊 RESULT")
print("=" * 50)
print(result)

# ============================================
# Try Your Own Topic (Optional)
# ============================================

# Change the topic and run this cell to research something else
custom_topic = "The benefits of renewable energy"

print(f"\n🔍 Researching: {custom_topic}")
print("=" * 50)

result = crew.kickoff(
    inputs={
        "topic": custom_topic
    }
)

print("\n" + "=" * 50)
print("📊 RESULT")
print("=" * 50)
print(result)

# ============================================
#Installation
# ============================================
!pip install -q crewai crewai-tools openai yfinance google-search-results

import warnings
warnings.filterwarnings('ignore')

print("Installation complete!")

# ============================================
# Import Libraries
# ============================================

import os
from getpass import getpass
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import yfinance as yf
from serpapi import GoogleSearch

print("Libraries imported!")

# ============================================
# Setup API Keys
# ============================================

# Get OpenAI API Key
print("Enter your OpenAI API Key")
print("Get one at: https://platform.openai.com/api-keys")
openai_api_key = getpass("OpenAI API Key: ")

# Get SerpAPI Key
print("\nEnter your SerpAPI Key")
print("Get one free at: https://serpapi.com/")
serpapi_key = getpass("SerpAPI Key: ")

# Set environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPAPI_API_KEY"] = serpapi_key

print("API keys configured!")

# ============================================
# Configure OpenAI LLM
# ============================================

# Initialize OpenAI
llm = LLM(
    model="gpt-3.5-turbo",
    api_key=openai_api_key,
    temperature=0.7
)

print("OpenAI LLM initialized!")

# ============================================
#Define Custom Tools
# ============================================

# Define input schemas
class StockSearchInput(BaseModel):
    query: str = Field(..., description="Search query for stock news")

class YahooFinanceInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")

class StockSearchTool(BaseTool):
    name: str = "StockNewsSearcher"
    description: str = "Search for latest stock news using SerpAPI Google News"
    args_schema: Type[BaseModel] = StockSearchInput

    def _run(self, query: str) -> str:
        """Search for stock news using SerpAPI"""
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": os.environ.get("SERPAPI_API_KEY"),
                "tbm": "nws",
                "num": 3
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            news = results.get("news_results", results.get("organic_results", []))

            if not news:
                return "No recent news found"

            output = []
            for item in news[:3]:
                title = item.get("title", "")
                snippet = item.get("snippet", item.get("description", ""))
                output.append(f"• {title}: {snippet[:100]}")

            return "\n".join(output)

        except Exception as e:
            return f"Error: {str(e)}"

class YahooFinanceTool(BaseTool):
    name: str = "YahooFinanceFetcher"
    description: str = "Get stock price data from Yahoo Finance"
    args_schema: Type[BaseModel] = YahooFinanceInput

    def _run(self, ticker: str) -> str:
        """Fetch stock price data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")

            if hist.empty:
                return f"No data for {ticker}"

            latest = hist.tail(5)
            current = latest['Close'].iloc[-1]
            change = ((latest['Close'].iloc[-1] - latest['Close'].iloc[0]) / latest['Close'].iloc[0]) * 100

            return f"""Stock: {ticker}
Price: ${current:.2f}
5-day Change: {change:+.2f}%
High: ${latest['High'].max():.2f}
Low: ${latest['Low'].min():.2f}"""

        except Exception as e:
            return f"Error: {str(e)}"

# Instantiate tools
search_tool = StockSearchTool()
finance_tool = YahooFinanceTool()

print("Tools created!")

# ============================================
# Define Agents
# ============================================

# Stock Analyst
analyst = Agent(
    role='Stock Analyst',
    goal='Analyze stock data and news',
    backstory='Expert financial analyst',
    verbose=False,
    allow_delegation=False,
    llm=llm,
    tools=[search_tool, finance_tool]
)

# Report Writer
writer = Agent(
    role='Report Writer',
    goal='Write investment reports',
    backstory='Professional financial writer',
    verbose=False,
    allow_delegation=False,
    llm=llm
)

print("Agents created!")

# ============================================
# Define Tasks
# ============================================

# Task 1: Search news
news_task = Task(
    description="""Search for latest news about Apple Inc. (AAPL) stock.
    Use query 'AAPL stock news'.
    Summarize the top 3 news items.""",
    expected_output="Summary of recent AAPL news",
    agent=analyst
)

# Task 2: Analyze prices
price_task = Task(
    description="""Analyze Apple stock (AAPL) price trends.
    Get price data and identify key trends.""",
    expected_output="Price analysis for AAPL",
    agent=analyst
)

# Task 3: Write report
report_task = Task(
    description="""Write a professional investor report for AAPL.
    Include:
    1. Executive Summary
    2. News Highlights
    3. Price Analysis
    4. Investment Outlook

    Keep under 300 words.""",
    expected_output="Investment report for AAPL",
    agent=writer
)

print("Tasks defined!")

# ============================================
# Run the Crew
# ============================================

# Create crew
crew = Crew(
    agents=[analyst, writer],
    tasks=[news_task, price_task, report_task],
    verbose=False
)

print("\nStarting Stock Analysis...")
print("=" * 60)

# Execute
result = crew.kickoff()

print("\n" + "=" * 60)
print("INVESTMENT REPORT - APPLE INC. (AAPL)")
print("=" * 60)
print(result)
print("=" * 60)
