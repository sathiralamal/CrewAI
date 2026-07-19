import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

# Initialize Large Language Model (LLM) using Groq's OpenAI-compatible endpoint.
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

llm = LLM(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    provider="openai",
    base_url="https://api.groq.com/openai/v1",
)

# Create your CrewAI agents with role, main goal/objective, and backstory/personality
summarizer = Agent(
    role='Documentation Summarizer', # Agent's job title/function
    goal='Create concise summaries of technical documentation', # Agent's main objective
    backstory='Technical writer who excels at simplifying complex concepts', # Agent's background/expertise
    llm=llm, # LLM that powers your agent
    verbose=True # Show agent's thought process as it completes its task
)

translator = Agent(
    role='Technical Translator',
    goal='Translate technical documentation to other languages',
    backstory='Technical translator specializing in software documentation',
    llm=llm,
    verbose=True
)

# Define your agents' tasks
summary_task = Task(
    description='Summarize this React hook documentation:\n\nuseFetch(url) is a custom hook for making HTTP requests. It returns { data, loading, error } and automatically handles loading states.',
    expected_output="A clear, concise summary of the hook's functionality",
    agent=summarizer # Agent assigned to task
)

translation_task = Task(
    description='Translate the summary to Turkish',
    expected_output="Turkish translation of the hook documentation",
    agent=translator,
    dependencies=[summary_task] # Must run after the summary task
)

# Create crew to manage agents and task workflow
crew = Crew(
    agents=[summarizer, translator], # Agents to include in your crew
    tasks=[summary_task, translation_task], # Tasks in execution order
    verbose=True
)

result = crew.kickoff()
print(result)