from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
from openai import OpenAI
import sys
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

openai.api_key = os.getenv('OPENAI_API_KEY')

if openai.api_key is None:
    print("API key not found. Please check your environment variables.")
    sys.exit(1)
else:
    print("API key has been found.")

client = OpenAI()

DATA_PATH = "documentation"
CHROMADB_PATH = "chromadb-storage"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Check if the database directory already exists
if os.path.exists(CHROMADB_PATH) and os.listdir(CHROMADB_PATH):
    print("Loading existing database.")
    db = Chroma(persist_directory=CHROMADB_PATH, embedding_function=embeddings)
else:
    print("Database not found. Creating new database.")

    loader = DirectoryLoader(DATA_PATH, show_progress=True)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMADB_PATH)

model = ChatOpenAI(model="gpt-4-turbo")

ticket_details = """
Title: Computer keyboard is not working
Description: I cannot type on my computer, please replace keyboard. 
"""

query = f"""
Based on the ticket details, state whether the issue is high or standard priority.
Only output one word: either the word "high" or the word "standard"
{ticket_details}
"""
docs = db.similarity_search_with_score(ticket_details)

# Extract top N most relevant document chunks
top_docs = docs[:3]  # Adjust the number of documents as needed
context = "\n\n".join([doc[0].page_content for doc in top_docs])

#print(context)

messages = [
    SystemMessage(content=context),
    HumanMessage(content=query),
]

model_output = model.invoke(messages)

print(model_output.content)
