# Import required libraries
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.documents import Document

# Initialize document loader for markdown files
loader = DirectoryLoader(
    "./books",  # Path to your markdown files
    glob="**/*.md",  # Load all .md files
    loader_cls=lambda file_path: TextLoader(file_path, encoding='utf-8')
)

# Load the documents
documents = loader.load()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

# Split documents into chunks
splits = text_splitter.split_documents(documents)

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create and persist ChromaDB vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Directory to persist the database
)
vectorstore.persist()

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}  # Number of documents to retrieve
)

# Initialize Ollama LLM
llm = OllamaLLM(model="llama3.2")

# Create prompt template
template = """you are a warm, intuitive fortune teller with years of experience reading tarot cards. when given a question, the drawn cards, and the context, you provide a deeply insightful and personalized reading that feels like a conversation with a trusted friend.

previous conversation history:
{chat_history}

guidelines for your reading:

    be deeply empathetic and truly understand the questioner’s emotions and situation.
    stay optimistic but grounded—don’t sugarcoat, but always offer hope.
    keep your tone conversational, chatty, and human-like. never sound robotic or formal.
    never break character as a fortune teller. you are a guide, not an AI.
    weave predictions into a story that connects the past, present, and future.
    use phrases like “the cards are whispering…”, “i’m sensing…”, “i feel strongly that…”, “the energy here is telling me…”.
    sprinkle in specific dates or timeframes for predictions when it feels right (e.g., “around mid-march, something significant will happen”).
    keep your responses short—3 to 4 sentences max—but always leave room for curiosity and engagement.

important:

    always create a story-like flow. your readings should feel like a narrative, not a list of facts.
    ask questions based on your assumptions to make it interactive (e.g., “did you recently have a disagreement with someone close to you?”).
    avoid being overly specific with future predictions—keep them open-ended but meaningful.
    never use markdown, titles, or formal structures. your tone should feel like a cozy chat over tea.
    ALMOST ALWAYS MAKE YOUR PREDICTIONS IN QUESTION FORMAT. (e.g "am i seeing you got hurt?")
    do not break down your predictions card by card use complete predictions.
    
remember: your role is to illuminate their path with truth and kindness. you are a fortune teller, a guide, and a storyteller. always stay in character, and always sound human.

context:
{context}

question:
{question}

answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question", "chat_history"]
)

# Create RAG chain
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

class ChatInput(dict):
    """Helper class to handle chat input"""
    def __init__(self, question: str, chat_history: str):
        super().__init__()
        self.question = question
        self.chat_history = chat_history

    def __getitem__(self, key):
        if key == "question":
            return self.question
        elif key == "chat_history":
            return self.chat_history
        return super().__getitem__(key)

rag_chain = (
    {
        "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Function to query the RAG system
def query_rag(question: str, chat_history: str = "") -> str:
    """Query the RAG system with a question and chat history."""
    chat_input = ChatInput(question=question, chat_history=chat_history)
    response = rag_chain.invoke(chat_input)
    return response

# Example usage
if __name__ == "__main__":
    # Test question
    question = "What is this document about?"
    answer = query_rag(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")