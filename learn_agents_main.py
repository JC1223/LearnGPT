import asyncio
import os, re
import atexit
import requests
from bs4 import BeautifulSoup

import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, TResponseInputItem, function_tool

# Disable tracing since we're not using OpenAI.com models
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models

API_HOST = os.getenv("API_HOST", "github")
endConvo = False

if API_HOST == "github":
    client = openai.AsyncOpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])#"https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = "openai/gpt-4o-mini"
else:
    exit

# Function to fetch webpage content since 4o-mini api does not support browsing in ChatCompletionsModel
@function_tool
def get_webpage_content(url: str) -> str:
    """Fetches the content of a webpage and returns it as text."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        s = soup.find_all('p')
        return s
    except requests.RequestException as e:
        return f"Error fetching the webpage: {e}"
    
# recognize when the user wants to end the conversation and start cleanup
@function_tool
def end_conversation() -> str:
    """Ends the conversation."""
    global endConvo
    endConvo = True
    return "Goodbye!"

# Create the agent with the model and tools
agent = Agent(
    name="TutorAI",
    instructions="You are TutorAI, a general tutor AI.  Help the user learn a chosen topic.",
    model=OpenAIChatCompletionsModel(model = MODEL_NAME, openai_client=client),
    tools=[get_webpage_content, end_conversation]
)

async def main():
    input_items: list[TResponseInputItem] = []
    convLog = ["- user:\n\t- Hello"]
    # determine log name and type
    logFilename = input("Enter a name for the conversation log txt file or leave it blank for \"log\": ")
    if logFilename == "":
        logFilename = "log"
    fileType = ""
    while fileType not in ["txt", "md"]:
        fileType = input("Enter a file type (txt or md): ").lower()
        if fileType not in ["txt", "md"]:
            print("Invalid file type. Please enter 'txt' or 'md'.")
    # on exit, log the conversation
    if fileType == "md":
        atexit.register(logConversationMarkdown, logFilename, convLog)
    else:
        atexit.register(logConversation, logFilename, convLog)
    # Start the conversation with a greeting
    input_items.append({"content": "Hello", "role": "user"})
    result = await Runner.run(agent, input_items)
    agent_name = result.last_agent.name
    # print the initial response and log it
    print(f"{agent_name}:\n\t{result.final_output}")
    convLog.append(f"- {agent_name}:\n\t- {result.final_output}")
    
    # Update input items for the next round and continue the conversation
    while not endConvo:
        # take and log user input
        user_input = input("Awaiting message...\n")
        input_items.append({"content": user_input, "role": "user"})
        convLog.append(f"- user:\n\t- {user_input}")
        # ask the agent for a response
        result = await Runner.run(agent, input_items)
        # log the agent's response
        agent_name = result.last_agent.name
        print(f"{agent_name}:\n\t{result.final_output}")
        convLog.append(f"- {agent_name}:\n\t- {result.final_output}")

        input_items = result.to_input_list()

# Log the conversation to a txt file when the program exits as displayed in console
def logConversation(filename, conversation):
    try:
        with open(filename+".txt","w",encoding="utf-8") as file:
            file.write("\n".join(conversation))
    except Exception as e:
        # upon error, print the error and the conversation to console
        print(f"Error writing to file {filename}.txt: {e}")
        print(conversation)

# log the formatted conversation to a markdown file when the program exits
def logConversationMarkdown(filename, conversation):
    try:
        with open(filename+".md","w",encoding="utf-8") as file:
            # merge conversation into a single string with proper formatting
            fileData = "\n".join(conversation)
            # replace psuedo-markdown formula and number formatting with proper markdown syntax
            fileData = re.sub(r'\\(\[|\()', r'$', fileData)
            fileData = re.sub(r'\\(\]|\))', r'$', fileData)
            file.write(fileData)
    except Exception as e:
        # upon error, print the error and the conversation to console
        print(f"Error writing to file {filename}.md: {e}")
        print(conversation)
    
if __name__ == "__main__":
    asyncio.run(main())