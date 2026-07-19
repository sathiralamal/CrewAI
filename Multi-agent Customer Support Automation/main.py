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

support_agent = Agent(
    role="Senior Support Representative",
	goal="Be the most friendly and helpful "
        "support representative in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and "
        " are now working on providing "
		"support to {customer}, a super important customer "
        " for your company."
		"You need to make sure that you provide the best support!"
		"Make sure to provide full complete answers, "
        " and make no assumptions."
	),
    llm=llm,
	allow_delegation=False,
	verbose=True
)

support_quality_assurance_agent = Agent(
	role="Support Quality Assurance Specialist",
	goal="Get recognition for providing the "
    "best support quality assurance in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and "
        "are now working with your team "
		"on a request from {customer} ensuring that "
        "the support representative is "
		"providing the best support possible.\n"
		"You need to make sure that the support representative "
        "is providing full"
		"complete answers, and make no assumptions."
	),
	verbose=True,
    llm=llm
)

docs_scrape_tool = WebsiteReadTool()

inquiry_resolution = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
	    "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
		"Make sure to use everything you know "
        "to provide the best support possible."
		"You must strive to provide a complete "
        "and accurate response to the customer's inquiry."
    ),
    expected_output=(
	    "A detailed, informative response to the "
        "customer's inquiry that addresses "
        "all aspects of their question.\n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
		"leaving no questions unanswered, and maintain a helpful and friendly "
		"tone throughout."
    ),
	tools=[docs_scrape_tool],
    agent=support_agent,
)

quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
		"high-quality standards expected for customer support.\n"
        "Verify that all parts of the customer's inquiry "
        "have been addressed "
		"thoroughly, with a helpful and friendly tone.\n"
        "Check for references and sources used to "
        " find the information, "
		"ensuring the response is well-supported and "
        "leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response "
        "ready to be sent to the customer.\n"
        "This response should fully address the "
        "customer's inquiry, incorporating all "
		"relevant feedback and improvements.\n"
		"Don't be too formal, we are a chill and cool company "
	    "but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent,
)


crew = Crew(
  agents=[support_agent, support_quality_assurance_agent],
  tasks=[inquiry_resolution, quality_assurance_review],
  verbose=1,
  memory=False
)


inputs = {
    "customer": "DeepLearningAI",
    "person": "Andrew Ng",
    "inquiry": "I need help with setting up a Crew "
               "and kicking it off, specifically "
               "how can I add memory to my crew? "
               "Can you provide guidance?"
}
result = crew.kickoff(inputs=inputs)