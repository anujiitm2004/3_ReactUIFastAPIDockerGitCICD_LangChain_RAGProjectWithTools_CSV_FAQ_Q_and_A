from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()  
# Check if loaded properly
print("SERPER_API_KEY:", os.getenv("SERPER_API_KEY"))
# Create Google Palm LLM model
# llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
llm = ChatOpenAI(model='gpt-4o-mini-2024-07-18', temperature=1.5, max_completion_tokens=50)
# # Initialize instructor embeddings using the Hugging Face model
# instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
instructor_embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=64)
vectordb_file_path = "faiss_index"


#Anuj
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
# from langchain_community.tools import DuckDuckGoSearchRun
import requests
#  DuckDuckGoSearchRun rename to ddgs
from ddgs import DDGS
from langchain_community.utilities import GoogleSerperAPIWrapper

print("--------anuj start----------------")
# ---------tool create----------------

@tool
def search_tool_google(query: str) -> str:
    """
    Google par live and latest search karta hai on internet web search using Serper API.
    """
    search = GoogleSerperAPIWrapper()
    result = search.run(query)
    return result

@tool
def search_tool_duckduck(query: str) -> str:
    """DuckDuckGo par query web search karta hai aur top 3 results ke title aur link return karta hai."""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=3)]
        if not results:
            return "Koi result nahi mila."
        return "\n".join(f"{r['title']} - {r['href']}" for r in results)

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data, temperature for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=57cb0c1f517879c324bb2ce91061071d&query={city}'

  response = requests.get(url)

  return response.json()

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

@tool
def add(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their sum"""
  return a + b

@tool
def divide(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their divide"""
  return a / b

# ----Tool Binding---------print(multiply.invoke({'a':3, 'b':4}))
llm = llm.bind_tools([multiply, add, divide, search_tool_duckduck, search_tool_google, get_weather_data])

query = HumanMessage('can you multiply 3 with 1000')
print("--------anuj end----------------")
#Anuj End

#pdf faq data cleaning
def extract_pdf_qa(pdf_docs):
    qa_pairs = []
    for doc in pdf_docs:
        lines = doc.page_content.split('\n')
        current_q = ""
        current_a = []
        for line in lines:
            line = line.strip()
            if line.startswith("●"):  # New question
                if current_q and current_a:
                    answer = " ".join(current_a).strip()
                    qa_pairs.append(Document(page_content=f"Q: {current_q}\nA: {answer}"))
                current_q = line.replace("●", "").strip()
                current_a = []
            else:
                if line != "":
                    current_a.append(line)
        # Add last pair
        if current_q and current_a:
            answer = " ".join(current_a).strip()
            qa_pairs.append(Document(page_content=f"Q: {current_q}\nA: {answer}"))
    return qa_pairs


def create_vector_db():
    # Load data from FAQ sheet
    print("CSV file exists?", os.path.exists("codebasics_faqs.csv"))
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt",encoding="ISO-8859-1")
    data = loader.load()

    # Load PDF
    pdf_loader = PyPDFLoader("DSMP 2.0 FAQs.pdf")
    pdf_raw_docs = pdf_loader.load()

    # Extract Q&A from PDF
    pdf_docs = extract_pdf_qa(pdf_raw_docs)

    # Combine all docs
    all_docs = data + pdf_docs

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=all_docs,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain(question):
    messages= []
    query = HumanMessage(question)
    messages = [query]
    response = llm.invoke(messages)
    messages.append(response)

    if hasattr(response, "tool_calls") and response.tool_calls:
        # STEP 1: Tool-based response
        # Tool name ke hisaab se sahi function call karna
        tool_mapping = {
            "multiply": multiply,
            "add": add,
            "divide": divide,
            "search_tool_duckduck": search_tool_duckduck,
            "search_tool_google" : search_tool_google,
            "get_weather_data" : get_weather_data
        }
        tool_name = response.tool_calls[0]["name"]
        if tool_name in tool_mapping:
            # tool_result = multiply.invoke(response.tool_calls[0])
            # tool_result = search_tool_duckduck.invoke(response.tool_calls[0])
            # yaha corresponding tool invoke hua, aur tool_message bana.
            tool_result = tool_mapping[tool_name].invoke(response.tool_calls[0])
            #now human_message, AI_message aur tool_message ko ek saath phir se llm ko bheja
            # and invoke kiya for final answer
            messages.append(tool_result)
            chain = llm.invoke(messages).content
            return [tool_name, chain , messages]
        else:
            return f"Tool '{tool_name}' not implemented."

        # tool_call = response.tool_calls[0]
        # tool_result = multiply.invoke(tool_call.args)
        # return f"Tool Response: {tool_result}"
    else:
        # Load the vector database from the local folder
        vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings,allow_dangerous_deserialization=True)

        # Create a retriever for querying the vector database
        retriever = vectordb.as_retriever(score_threshold=0.7)

        prompt_template = """Given the following context and a question, generate an answer based on this context only. Also you can use tools/function which are bound to you when necessary for generating answer.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context/tools, kindly state "I don't know." Don't try to make up an answer.
    
    
        CONTEXT: \n{context}
        
    
        QUESTION: {question}"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        print("-------anujPromptStartPrinting-----------")
        print("PROMPT")
        print("-------anujPromptendPrinting-----------")
        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            input_key="query",
                                            return_source_documents=True,
                                            chain_type_kwargs={"prompt": PROMPT})

        response = chain({"query": question})

        # Get the docs
        docs = response.get("source_documents", [])
        # Create context manually from docs
        context = "\n\n".join([doc.page_content for doc in docs])

        response["PROMPT"] = PROMPT
        response["source_documents"]= response.get("source_documents", [])
        response["input"]= {
            "context": context,
            "question": question
        }
        return response  # ✅ Now this is a dict with 'result' key
        # return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))