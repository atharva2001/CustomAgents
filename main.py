from langchain.tools import StructuredTool
from langchain.agents import create_json_agent, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.tools import tool

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key="You key here") 

@tool 
def getSum(n1: int, n2: int) -> int:
    """Return the sum of two numbers"""
    return n1 + n2 

toolkit = [getSum]

template = """
        You are an mathematical Bot. Your task is to reutrn the sum of two numbers given to you.

        Begin!

        Number 1: {input1}
        Number 2: {input2}
        """

agent = initialize_agent(
    llm=llm,
    tools=toolkit,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

prompt = PromptTemplate(input_variables=['input1', 'input2'], template=template)
response = agent.invoke(prompt.format(input1="-1", input2="-2"))

print("---------------------------------------------------------------------")

print(response["output"])
