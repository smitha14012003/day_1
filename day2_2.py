import os
import requests
import streamlit as st
import weaviate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatVertexAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from trulens_eval import Feedback, Huggingface, Tru, TruChain
from weaviate.embedded import EmbeddedOptions

from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

hugs = Huggingface()
tru = Tru()

chain_recorder = None
conversation = None

def handle_conversation(user_input):
    input_dict = {"question": user_input}
    try:
        with chain_recorder as recording:
            response = conversation(input_dict)
            return response.get("answer", "No response generated.")
    except Exception as e:
        return f"An error occurred: {e}"

st.sidebar.title("Configuration")
url = st.sidebar.text_input("Enter URL")
submit_button = st.sidebar.button("Submit")

if 'initiated' not in st.session_state:
    st.session_state['initiated'] = False
    st.session_state['messages'] = []

if submit_button or st.session_state['initiated']:
    st.session_state['initiated'] = True

    if url and not conversation:
        # Load and process the document
        res = requests.get(url)
        with open("state_of_the_union.txt", "w") as f:
            f.write(res.text)

        loader = TextLoader('./state_of_the_union.txt')
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        client = weaviate.Client(embedded_options=EmbeddedOptions())
        vectorstore = Weaviate.from_documents(client=client, documents=chunks, embedding=OpenAIEmbeddings(), by_text=False)
        
        retriever = vectorstore.as_retriever()

        llm = ChatVertexAI()
        template = """You are an assistant for question-answering tasks..."""
        prompt = ChatPromptTemplate.from_template(template)
        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        conversation = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

        chain_recorder = TruChain(
            conversation,
            app_id="RAG-System",
            feedbacks=[
                Feedback(hugs.language_match).on_input_output(),
                Feedback(hugs.not_toxic).on_output(),
                Feedback(hugs.pii_detection).on_input(),
                Feedback(hugs.positive_sentiment).on_output(),
            ]
        )

                