import asyncio
import os, re
import sys
import atexit
import requests
from bs4 import BeautifulSoup

import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, TResponseInputItem, function_tool
from dotenv import load_dotenv

# Disable tracing since we're not using OpenAI.com models
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
# load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
endConvo = False

if API_HOST == "github":
    client = openai.AsyncOpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])#"https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = "openai/gpt-4o-mini"#os.getenv("GITHUB_MODEL", "gpt-4o")
else:
    # print("Exit: api_host")
    exit

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
    
@function_tool
def end_conversation() -> str:
    """Ends the conversation."""
    global endConvo
    endConvo = True
    return "Goodbye!"

agent = Agent(
    name="TutorAI",
    instructions="You are a general tutor.  Help the user learn a chosen topic.",
    model=OpenAIChatCompletionsModel(model = MODEL_NAME, openai_client=client),
    tools=[get_webpage_content, end_conversation]
)
# print("B")
async def main():
    input_items: list[TResponseInputItem] = []
    convLog = ["- user:\n\t- Hello"]
    logFilename = input("Enter a name for the conversation log txt file or leave it blank for \"log\": ")
    if logFilename == "":
        logFilename = "log"
    fileType = ""
    while fileType not in ["txt", "md"]:
        fileType = input("Enter a file type (txt or md): ").lower()
        if fileType not in ["txt", "md"]:
            print("Invalid file type. Please enter 'txt' or 'md'.")
    if fileType == "md":
        atexit.register(logConversationMarkdown, logFilename, convLog)
    else:
        atexit.register(logConversation, logFilename, convLog)
    input_items.append({"content": "Hello", "role": "user"})
    result = await Runner.run(agent, input_items)
    agent_name = result.last_agent.name
    print(f"{agent_name}:\n\t{result.final_output}")
    convLog.append(f"- {agent_name}:\n\t- {result.final_output}")
    # for new_item in result.new_items:
    #     agent_name = new_item.agent.name
    #     # print(new_item)
    #     print(f"{agent_name}:\n\t{new_item.raw_item.content[-1].text}")
    #     convLog.append(f"{agent_name}:\n\t{new_item.raw_item.content[-1].text}")
    
    while not endConvo:
        user_input = input("Awaiting message...\n")
        input_items.append({"content": user_input, "role": "user"})
        convLog.append(f"- user:\n\t- {user_input}")
        result = await Runner.run(agent, input_items)
        agent_name = result.last_agent.name
        print(f"{agent_name}:\n\t{result.final_output}")
        convLog.append(f"- {agent_name}:\n\t- {result.final_output}")
        # for new_item in result.new_items:
        #     agent_name = new_item.agent.name
        #     # print(new_item)
        #     print(f"{agent_name}:\n\t{new_item.raw_item.content[-1].text}")
        #     convLog.append(f"{agent_name}:\n\t{new_item.raw_item.content[-1].text}")

        input_items = result.to_input_list()


    # response
    # terminate = False
    # asyncio.run(print,"00000000000")
    # userIn = await asyncInput("Awaiting Input...")
    # userIn = "Hello"
    # print("C")
    # result = await Runner.run(agent, input="Hello how are you?")
    # print("D")
    # print(result.final_output)#, flush=True)
    # print("E")
    # print("AAAA",flush=True)
    
    # userIn = await asyncInput("Awaiting Input...")
    # userIn = "Exit"
    # terminate = False if userIn.lower() != "exit" else True
    # new_input = result.to_input_list() + [{"role":"user", "content":userIn}]
    # while terminate == False:
    #     result = await Runner.run(agent, input=new_input)
    #     print(result.final_output)
    #     # userIn = await asyncInput("Awaiting Input...")
    #     terminate = False if userIn.lower() != "exit" else True
    #     new_input = result.to_input_list() + [{"role":"user", "content":userIn}]
    #     # asyncio.run(print,"AAAAAAAA")
    # asyncio.run(print,"BBBBBBBBBB")
    
# async def asyncInput(string: str) -> str:
#     await asyncio.to_thread(sys.stdout.write, f'{string} ')
#     return await asyncio.to_thread(sys.stdin.readLine)

def logConversation(filename, conversation):
    try:
        with open(filename+".txt","w",encoding="utf-8") as file:
            file.write("\n".join(conversation))
    except Exception as e:
        print(f"Error writing to file {filename}.txt: {e}")
        print(conversation)

def logConversationMarkdown(filename, conversation):
    try:
        with open(filename+".md","w",encoding="utf-8") as file:
            fileData = "\n".join(conversation)
            fileData = re.sub(r'\\(\[|\()', r'$', fileData)
            fileData = re.sub(r'\\(\]|\))', r'$', fileData)
            file.write(fileData)
    except Exception as e:
        print(f"Error writing to file {filename}.md: {e}")
        print(conversation)
    
# print("F")
if __name__ == "__main__":
    asyncio.run(main())
# print("G")