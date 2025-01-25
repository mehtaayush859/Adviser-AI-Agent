from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool

from langchain import hub
import os

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
loader = DirectoryLoader("/Users/ankitach/Downloads/datasets")
docs = loader.load_and_split(text_splitter=text_splitter)

load_dotenv()
os.environ['OPENAI_API_KEY'] = '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'

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()
tool = create_retriever_tool(
    retriever,
    "search_quality_of_life",
    "Searches and returns best city to live",
)
tools = [tool]
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.messages

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke(
    {
        "input": "if I move to California as a smoker, what will my insurance cost be compared to New York?"
    }
)
print(result["output"])