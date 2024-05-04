import openai
from openai import OpenAI
import sys
import os
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader

DATA_PATH = "documentation"

#Get the OpenAI API from system Environment Variables
openai.api_key = os.getenv('OPENAI_API_KEY')

if openai.api_key is None:
    print("API key not found. Please check your environment variables.")
    sys.exit(1)
else:
    print("API key has been found.")

client = OpenAI()

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "Supernik22"

graph = Neo4jGraph()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm_transformer = LLMGraphTransformer(llm=llm)

# Load all text documents from documentation directory
loader = DirectoryLoader(DATA_PATH, glob="**/*.txt")
docs = loader.load()
documents = [Document(page_content=docs[0].page_content)]

graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

graph.add_graph_documents(graph_documents)