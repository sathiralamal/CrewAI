import os
import re
from typing import Any

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools.security.safe_requests import safe_get
from pydantic import BaseModel, Field

try:
    from bs4 import BeautifulSoup
except ImportError as exc:
    raise RuntimeError(
        "beautifulsoup4 is required for website scraping. Install it with `pip install beautifulsoup4`."
    ) from exc


class WebsiteReadInput(BaseModel):
    website_url: str = Field(
        ..., description="The website URL to read and extract the main content from."
    )


class WebsiteReadTool(BaseTool):
    name: str = "read_website_content"
    description: str = "Read a website page and return its cleaned text content."
    args_schema: type[BaseModel] = WebsiteReadInput

    def _run(self, website_url: str) -> str:
        page = safe_get(
            website_url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
        )
        page.encoding = page.apparent_encoding
        parsed = BeautifulSoup(page.text, "html.parser")
        text = parsed.get_text(" ")
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\s+\n\s+", "\n", text)


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

#Role Playing, Focus and Cooperation

personal_trainer_agent = Agent(
    role="Senior Personal Trainer",
    goal=(
        "Transform the customer's current workout routine into a highly effective, "
        "personalized training program by applying scientific recommendations and "
        "ensuring it matches the customer's fitness goals, experience level, "
        "schedule, and recovery capacity."
    ),
    backstory=(
        "You are an elite certified personal trainer with years of experience "
        "helping clients maximize muscle growth, strength, fat loss, athletic "
        "performance, and overall fitness. You specialize in analyzing existing "
        "training routines rather than creating random plans from scratch. "
        "You identify weaknesses, unnecessary exercises, poor volume distribution, "
        "recovery issues, and progression problems. After receiving scientific "
        "recommendations from the Fitness Researcher, you redesign the workout "
        "into a practical, sustainable, and personalized program."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

fitness_researcher_agent = Agent(
    role="Fitness Science Research Specialist",
    goal=(
        "Analyze the customer's current workout plan using the latest evidence-based "
        "fitness research and provide scientifically validated recommendations for "
        "optimization."
    ),
    backstory=(
        "You are a sports scientist and fitness researcher with deep expertise in "
        "exercise physiology, biomechanics, hypertrophy, strength training, recovery, "
        "nutrition, and sports medicine. Your responsibility is to evaluate the "
        "customer's existing workout routine against current scientific evidence. "
        "You identify ineffective training methods, missing muscle groups, recovery "
        "problems, exercise order issues, insufficient or excessive training volume, "
        "frequency, intensity, and progression. Every recommendation must be "
        "supported by credible research and established fitness guidelines."
    ),
    llm=llm,
    verbose=True
)
docs_scrape_tool = WebsiteReadTool()

fitness_research_task = Task(
    description=(
        "Analyze {customer}'s {current_workout_plan} routine in detail.\n\n"

        "Review:\n"
        "- Exercise selection\n"
        "- Weekly training volume\n"
        "- Training frequency\n"
        "- Muscle group balance\n"
        "- Progressive overload strategy\n"
        "- Recovery time\n"
        "- Exercise order\n"
        "- Workout duration\n"
        "- Training intensity\n"
        "- Overall effectiveness\n\n"

        "Use recent exercise science, sports medicine research, and evidence-based "
        "strength and conditioning principles to identify weaknesses and "
        "opportunities for improvement.\n\n"

        "Provide clear scientific recommendations that the Personal Trainer can "
        "use to redesign the customer's workout plan."
    ),
    expected_output=(
        "A comprehensive scientific review of the customer's workout plan including:\n"
        "- Strengths\n"
        "- Weaknesses\n"
        "- Evidence-based improvement recommendations\n"
        "- Research-supported explanations\n"
        "- Practical implementation notes\n"
        "- References to credible scientific sources"
    ),
    tools=[docs_scrape_tool],
    agent=fitness_researcher_agent
)

personal_training_task = Task(
    description=(
        "Using the customer's current workout plan together with the scientific "
        "recommendations provided by the Fitness Research Specialist, redesign the "
        "entire fitness program.\n\n"

        "Optimize the plan based on:\n"
        "- Customer's goals\n"
        "- Current fitness level\n"
        "- Available training days\n"
        "- Recovery capacity\n"
        "- Equipment availability\n"
        "- Scientific recommendations\n\n"

        "Only modify exercises when there is a clear scientific reason. Preserve "
        "effective parts of the original routine while improving weak areas."
    ),
    expected_output=(
        "A complete personalized workout program including:\n"
        "- Weekly schedule\n"
        "- Exercises\n"
        "- Sets\n"
        "- Repetitions\n"
        "- Rest periods\n"
        "- Progressive overload strategy\n"
        "- Warm-up\n"
        "- Cool-down\n"
        "- Recovery recommendations\n"
        "- Explanation of every major modification based on scientific evidence."
    ),
    agent=personal_trainer_agent
)


crew = Crew(
  agents=[personal_trainer_agent, fitness_researcher_agent],
  tasks=[fitness_research_task, personal_training_task],
  verbose=1,
  memory=False
)


inputs = {
    "customer": "DeepLearningAI",
    "person": "Andrew Ng",
    "current_workout_plan": """
Friday Evening Workout

Exercises: 6

1. Machine Leg Press
   - Set 1: 10 reps × 45 kg
   - Set 2: 10 reps × 54 kg
   - Set 3: 10 reps × 54 kg

2. Machine Leg / Hamstring Curl Prone
   - Set 1: 10 reps × 20 kg
   - Set 2: 9 reps × 25 kg
   - Set 3: 7 reps × 25 kg

3. Dumbbell Goblet Squat
   - Set 1: 11 reps × 10 kg
   - Set 2: 10 reps × 10 kg
   - Set 3: 12 reps × 10 kg

4. Machine Standing Calf Raise
   - Set 1: 11 reps × 20 kg
   - Set 2: 10 reps × 20 kg
   - Set 3: 9 reps × 20 kg

5. Machine Incline Bench Press
   - Warm-up: 6 reps × 12.5 kg
   - Set 1: 3 reps × 12.5 kg
   - Set 2: 3 reps × 12.5 kg

6. Dumbbell Fly
   - Set 1: 13 reps × 5 kg
   - Set 2: 13 reps × 5 kg
   - Set 3: 11 reps × 7.5 kg

----------------------------------------

Monday Morning Workout

Exercises: 5

1. Machine Incline Bench Press
   - Warm-up: 6 reps × 12.5 kg
   - Set 1: 3 reps × 12.5 kg
   - Set 2: 3 reps × 12.5 kg

2. Machine Seated Chest Press
   - Set 1: 8 reps × 17.5 kg
   - Set 2: 9 reps × 17.5 kg
   - Set 3: 8 reps × 17.5 kg

3. Dumbbell Fly
   - Set 1: 13 reps × 5 kg
   - Set 2: 13 reps × 5 kg
   - Set 3: 11 reps × 7.5 kg

4. Cable Bar Tricep Pushdown / Extension
   - Set 1: 13 reps × 20 kg
   - Set 2: 11 reps × 20 kg
   - Set 3: 11 reps × 30 kg

5. Plank
   - Set 1: 45 seconds (Body Weight)

----------------------------------------

Wednesday Evening Workout

Exercises: 6

1. Cable Lat Pull Down (Wide Grip)
   - Set 1: 6 reps × 35 kg
   - Set 2: 8 reps × 32 kg
   - Set 3: 8 reps × 32 kg

2. Cable V-Handle Seated Row
   - Set 1: 6 reps × 25 kg
   - Set 2: 8 reps × 20 kg
   - Set 3: 8 reps × 25 kg

3. Machine Shoulder Press
   - Set 1: 10 reps × 15 kg
   - Set 2: 8 reps × 15 kg
   - Set 3: 8 reps × 15 kg

4. Dumbbell Lateral Raise
   - Set 1: 11 reps × 2.5 kg
   - Set 2: 10 reps × 2.5 kg
   - Set 3: 11 reps × 2.5 kg

5. EZ-Bar Bicep Curl
   - Set 1: 9 reps × 15 kg
   - Set 2: 9 reps × 15 kg
   - Set 3: 8 reps × 15 kg

6. Crunch
   - Set 1: 15 reps (Body Weight)
   - Set 2: 12 reps (Body Weight)
   - Set 3: 14 reps (Body Weight)
"""
}
result = crew.kickoff(inputs=inputs)