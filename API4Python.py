import gradio as gr
import os
from getpass import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
#arg to return source document = true
#langchain search gpt enabled
from langchain import OpenAI 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAIChat
from langchain.document_loaders import PagedPDFSplitter
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
#Additional
import logging
import fitz
from PIL import Image

os.environ['OPENAI_API_KEY'] = "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"
model = ChatOpenAI(model_name="gpt-3.5-turbo")
#Directory
file = 'Mobiz_TE.pdf'


chat_history = []

page_number = ""
page_citation = ""


#
#from langchain.document_loaders import TextLoader

def process_file(dir):
    # elevate an error if API key isn’t supplied
    # Load the PDF file utilizing PyPDFLoader
    loader = PyPDFLoader(dir)
    documents = loader.load()

    #   Initialize OpenAIEmbeddings for textual content embeddings
    embeddings = OpenAIEmbeddings()

    # Create a ConversationalRetrievalChain with ChatOpenAI language mannequin
    # and PDF search retriever
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    pdfsearch = DeepLake.from_documents(texts, embeddings,)

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3),
    retriever = pdfsearch.as_retriever(search_kwargs={'ok': 1}),
    return_source_documents=True,)
    return chain


#Picture rendering
def render_file(file, N):
    """
    Function to render a specific page of a PDF file as an image
    """
    logging.info("Rendering File...")
    #global N, dir
    #file = dir
    doc = fitz.open(file)
    page = doc[N]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    logging.info("Rendering File Completed...")
    return image

question1 = ["give me one General Travel Safety Tip"]
chain = process_file(file)

def generate_answer(user_input):
    # Generate response using ChatGPT model
    result = chain({"question": user_input, "chat_history": chat_history})
    source_documents = result['source_documents']

    for document in source_documents:
    # Access the page number from the metadata
        page_number = document.metadata['page']
        page_citation = document.metadata['source']

    return result




chatgpt_response = generate_answer(question1)

