import streamlit as st
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import tempfile
import json
# Load environment variables
# load_dotenv()

os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_API_KEY"]

service_account_info = st.secrets["gcp_service_account"]

service_account_dict = dict(service_account_info)

# Write to a temporary file
with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
    json.dump(service_account_dict, tmp)
    tmp_path = tmp.name

# Set the environment variable for Google libraries
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_path

# Set Streamlit page config
st.set_page_config(page_title="LangChain SQL Agent", layout="wide")

# Initialize LLM and databases (cache to avoid repeated loading)
@st.cache_resource(show_spinner=False)
def load_agent():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    sqlalchemy_url_1 = f'bigquery://testing-ai-agents-457413/testing_agents'
    sqlalchemy_url_2 = f'bigquery://testing-ai-agents-457413/newyork'
    sqlalchemy_url_3 = f'bigquery://testing-ai-agents-457413/san_francisco'

    db1 = SQLDatabase.from_uri(sqlalchemy_url_1)
    db2 = SQLDatabase.from_uri(sqlalchemy_url_2)
    db3 = SQLDatabase.from_uri(sqlalchemy_url_3)

    databases = {
        "testing_agents": (db1, "use this tool to get data about population of different cities"),
        "newyork": (db2, "use this tool to get data about newyork city"),
        "san_francisco": (db3, "use this tool to get data about san francisco city")
    }

    tools = []
    for name, kit in databases.items():
        toolkit = SQLDatabaseToolkit(db=kit[0], llm=llm)
        internal_tools = toolkit.get_tools()
        for i in internal_tools:
            i.name = f"{name}_{i.name}"
            i.description = kit[1] + i.description
            tools.append(i)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an intelligent assistant that can answer user questions by querying multiple databases using specialized tools. "
            "You have access to the following tools:\n"
            "{tools}"
            "Decide which tool is best suited for each query."
            "If the user's question is unrelated to the databases, respond directly without using a tool. "
            "Always explain your reasoning before presenting the final answer."
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    master_agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=master_agent, tools=tools, verbose=True)
    return agent_executor, tools

agent_executor, tools = load_agent()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🦜 LangChain SQL Agent Chat")
st.markdown(
    """
    This assistant can answer questions by querying multiple SQL databases. 
    Ask anything about population, New York, or San Francisco city data!
    """
)

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# User input
user_input = st.chat_input("Ask your question...")

if user_input:
    st.chat_message("user").write(user_input)
    # Call the agent
    with st.spinner("Thinking..."):
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history,
            "tools": tools
        })
    ai_response = result["output"]
    st.chat_message("assistant").write(ai_response)
    # Update session history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=ai_response))
