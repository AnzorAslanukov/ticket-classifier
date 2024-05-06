from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
from openai import OpenAI
import sys
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
from datetime import datetime
from Levenshtein import ratio

#Get the OpenAI API from system Environment Variables
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

# Check if the ChromaDB database directory already exists and has files in it
if os.path.exists(CHROMADB_PATH) and os.listdir(CHROMADB_PATH):
    print("Loading existing database.")
    db = Chroma(persist_directory=CHROMADB_PATH, embedding_function=embeddings)
else:
    # ChhromaDB database creation code
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

# gpt-4-turbo model performs decision making
# Price for 1M input tokens as of 5/4/2024: $10
decision_model = ChatOpenAI(model="gpt-4-turbo")

# Dictionary contains ticket title, description, location
ticket_details = {
    "title":"IntelliVue is not connected to monitoring system",
    "description":"Patient data is not going into central monitoring system.",
    "location":"Pediatric Hospital - 700 Walnut Street, Franklin City, PA"
}

# Query for gpt-4-turbo with ticket details to get decision
gpt4_query = f"""
Based on the ticket details, state whether the issue is high or standard priority and
state which IT support group queue the ticket will need to go to.
For ticket severity, only output either "high" or "standard" 
For ticket IT department, only output the IT department name. 
Ticket title: {ticket_details["title"]}
Ticket description: {ticket_details["description"]}
ticket location: {ticket_details["location"]}
Follow output format example: 
"high, telecommunications"
"""

# Function to remove duplicate sentences and paragraphs for context string
def remove_duplicates(text, separator='\n'):
    lines = text.split(separator)  
    unique_lines = list(dict.fromkeys(lines))
    return separator.join(unique_lines)

docs = db.similarity_search_with_score(f"{ticket_details["title"]} {ticket_details["description"]}")

# Extract top N most relevant document chunks
top_docs = docs[:3]  # Adjust the number of documents as needed
context = "\n\n".join([doc[0].page_content for doc in top_docs])

# Remove duplicated sentces from context variable
context = remove_duplicates(context)

'''
print("Context length:", len(context))
print("Context word count:", str(len(context.split())))
print("GPT 4 query length:", len(gpt4_query))
'''

# Use gpt-4-turbo to make decision for ticket
gpt4_messages = [
    SystemMessage(content=context),
    HumanMessage(content=gpt4_query)
]

model_decision = decision_model.invoke(gpt4_messages)
model_decision_content = model_decision.content.lower()
# print(model_decision_content)

# Parse model output into relevant dictionary keys
parts = model_decision_content.split(", ")
decision_dict = {
    "severity": parts[0],
    "department": parts[1]
}

def find_assignment_details(ticket_details, decision_dict):
    if decision_dict["severity"].lower() != "high":
        return {}  # Only process if severity is 'high'

    # Load the on-call list from CSV
    df = pd.read_csv("lhc-on-call-list.csv")

    # Ensure all data are treated as strings
    df['IT Department'] = df['IT Department'].astype(str)
    df['Building Support Group'] = df['Building Support Group'].astype(str)

    # Prepare to search by concatenating building name and address
    df['Combined Address'] = df['Building Name'].astype(str) + " " + df['Building Address'].astype(str)

    # Determine current time and decide shift
    current_time = datetime.now().time()
    shifts = {
        "7:00 AM - 3:00 PM": (datetime.strptime("7:00 AM", "%I:%M %p").time(), datetime.strptime("3:00 PM", "%I:%M %p").time()),
        "3:00 PM - 11:00 PM": (datetime.strptime("3:00 PM", "%I:%M %p").time(), datetime.strptime("11:00 PM", "%I:%M %p").time()),
        "11:00 PM - 7:00 AM": (datetime.strptime("11:00 PM", "%I:%M %p").time(), datetime.strptime("7:00 AM", "%I:%M %p").time())
    }

    # Determine current shift based on time
    current_shift = None
    for shift, times in shifts.items():
        start, end = times
        if start <= current_time <= end or (start > end and (start <= current_time or current_time <= end)):
            current_shift = shift
            break

    # Search for a matching row in the dataframe
    for _, row in df.iterrows():
        if current_shift == row['On-Call Shift'] and (
            decision_dict['department'].lower() in [row['IT Department'].lower(), row['Building Support Group'].lower()]) and (
            ratio(ticket_details['location'].lower(), row['Combined Address'].lower()) > 0.8):  # Adjust similarity threshold as needed
            # Return assignment details from the matching row
            return {
                "location": ticket_details['location'],
                "support_group": row['Building Support Group'],
                "shift": row['On-Call Shift'],
                "contact_number": row['On-Call Phone Number'],
                "name": row['On-Call Name']
            }
    
    return {}  # Return empty if no match found

assignment_details = find_assignment_details(ticket_details, decision_dict)
print(assignment_details)
