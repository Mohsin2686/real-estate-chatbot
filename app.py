from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ---- Setup Embeddings & Pinecone ----
embeddings = download_hugging_face_embeddings()
index_name = "real-estate-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# ---- Setup Chat Model ----
chatModel = ChatOpenAI(temperature=0.7)

# ---- Add Memory ----
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ---- Contextualization Prompt for history-aware retriever ----
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Given the chat history and a user query, rewrite the query to be a standalone question. "
         "Do NOT answer it, just reformulate if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    chatModel, retriever, contextualize_prompt
)

# ---- QA Prompt with retrieved context ----
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt + "\n\nUse the following retrieved context to answer:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, qa_prompt)

# ---- Retrieval + Memory Chain ----
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print("User:", input_text)

    # Include memory into the chain call
    response = rag_chain.invoke(
        {"input": input_text, "chat_history": memory.load_memory_variables({})["chat_history"]}
    )

    # Save conversation to memory
    memory.save_context({"input": input_text}, {"output": response["answer"]})

    print("Response:", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
