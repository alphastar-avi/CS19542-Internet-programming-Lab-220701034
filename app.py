from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os



def load_diary_entries(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Load diary 
diary_file_path = "diary.txt"
diary_content = load_diary_entries(diary_file_path)

# Split 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_texts = text_splitter.split_text(diary_content)

# embeddings and vector 
embeddings = OllamaEmbeddings(model="llama3.2:3b")
vectorstore = Chroma.from_texts(split_texts, embeddings)

# Set up the LLM
llm = OllamaLLM(model="llama3.2:3b", streaming=True)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

def analyze_diary(query):
    result = qa_chain.invoke({"query": query})  
    return result["result"]

# prompt put
print(analyze_diary("tell me some happy moments in my life"))

# update emb to vect (chromadb)
def update_knowledge_base():
    global vectorstore
    diary_content = load_diary_entries(diary_file_path)
    split_texts = text_splitter.split_text(diary_content)
    vectorstore = Chroma.from_texts(split_texts, embeddings)
    print("Knowledge base updated with the latest diary entries.")

