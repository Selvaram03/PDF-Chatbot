import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain.document_loaders import CSVLoader, PyPDFLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader,UnstructuredXMLLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from io import BytesIO
import textract
import tempfile
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

text_splitter = CharacterTextSplitter(
              separator="\n",
              chunk_size=400,
              chunk_overlap=200,
              length_function=len)
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
DOCUMENT_MAP = {
    "txt": TextLoader,
    "pdf": PyPDFLoader,
    "docx": Docx2txtLoader,
    "doc": Docx2txtLoader,
    "xml": UnstructuredXMLLoader
}

@st.cache_data
def extract_text_from_file(file):
    file_extension = file.name.split(".")[-1].lower()
    temp_file_path = f"temp_file.{file_extension}"

    with open(temp_file_path, "wb") as f:
        f.write(file.read())

    loader_class = DOCUMENT_MAP.get(file_extension)

    if loader_class:
        loader = loader_class(temp_file_path)
        documents = loader.load()
        os.remove(temp_file_path)
        return documents
    else:
        return "File type not supported for text extraction."
# def extract_text_from_pdf(file):

#     text = ""
#     pdf_document = fitz.open(stream=BytesIO(file.read()))
    
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document[page_num]
#         text += page.get_text()
    
#     return text

@st.cache_resource
def get_conversation_chain(_doc_txt):

  chunks = text_splitter.split_text(_doc_txt)

  knowledge_base = Chroma.from_texts(chunks, embeddings)
 
  llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={
          "min_length": 200,
           "max_length": 512,
           "temperature": 0.5,
           "max_new_tokens": 200,
           "num_return_sequences": 1
       })
  template = """Use the following pieces of information to answer the user's question.
                  If you don't know the answer, just say that you don't know, don't try to make up an answer.
                  
                  Context: {context}
                  Question: {question}
                  
                  Only return the helpful answer below and nothing else.
                  Helpful answer:
                  """
 


  QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template,)
  
  memory = ConversationBufferMemory(
          memory_key='chat_history', return_messages=True)

  conversation_chain = ConversationalRetrievalChain.from_llm(
          llm=llm,
          retriever=knowledge_base.as_retriever(),
          memory=memory,
          combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT}
      )
  return conversation_chain


def generate_response(question):
    response= st.session_state.conversation({"question": question})
    return response["answer"]

def regenerate_response():
    last_question = st.session_state.last_question
    if last_question:
      # if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
      #       st.session_state.messages.pop() 
      response = generate_response(last_question)
      if response:
            st.session_state.messages.append({"role": "assistant", "content": response})



def main():
    st.header("Chat with Nova Techset QA BOT")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_question = prompt 
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt) 
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
    
    with st.sidebar:
            st.image("https://novatechset.com/wp-content/uploads/2022/08/novatechset-logo.png", width=300)
            #st.subheader("Your documents")
            uploaded_files = st.file_uploader("Upload your File here and click on 'Process'", type=["pdf", "txt", "docx", "doc", "xml"])
            if st.button("Process"):
              with st.spinner("Processing"): 
                text = extract_text_from_file(uploaded_files)
                plain_text=""
                for i in range(len(text)):
                  plain_text+=str(st.write(text[i].page_content))


                st.session_state.conversation = get_conversation_chain(plain_text)
                st.sidebar.success("Done! You Can Chat 💬")
    col1, col2 = st.columns([30, 5])  # Adjust the column ratios as needed
    with col2:
      if st.session_state.messages[-1]["role"] != "assistant":
          st.write("")  # Placeholder to align the button vertically
      if st.button("👎", key='regenerate_button'):
          if st.session_state.last_question:
              regenerate_response()
    
if __name__ == '__main__':
    main()

