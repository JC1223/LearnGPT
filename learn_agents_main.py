import asyncio
import os

import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv

# Disable tracing since we're not using OpenAI.com models
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "github":
    client = openai.AsyncOpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")
else:
    exit

agent = Agent(
    name="LearnGPT",
    instructions="You are a general tutor.  Help the user learn a chosen topic.",
    model=OpenAIChatCompletionsModel(model = MODEL_NAME, openai_client=client)
)

async def main():
    result = await Runner.run(agent, input="Hello World")
    print(result.final_output)


if __name__ == "main":
    asyncio.run(main())