import os
import random
import shutil
import streamlit as st  # used to create our UI frontend
import traceback
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI  # used for GPT3.5/4 model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.callbacks import get_openai_callback


def init_page():
    st.set_page_config(
        page_title="The Moss Masher",
        page_icon=":dog:"
    )
    st.sidebar.title("Menu")
    st.sidebar.image("./FullLogo.png", width=300, use_column_width="always")
    st.session_state.costs = []


def select_model(model, aiTemp):
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
        st.session_state.max_token = 16385
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
        st.session_state.max_token = 16385
    else:
        st.session_state.model_name = "gpt-4"
        st.session_state.max_token = 8192

    # 300: The number of tokens for instructions outside the main text
    return ChatOpenAI(temperature=aiTemp, model_name=st.session_state.model_name)


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.costs = []


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


def create_login_page():
    clear_files()
    st.title(":football: :green[The Moss Masher] :dog:")
    st.header("Mash your long documents and videos into easy answers!")
    container = st.container()
    with container:
        with st.form(key="frmOptions", clear_on_submit=True):
            st.markdown("To use this app, please provided an OpenAI key")
            st.markdown(
                "First, create an OpenAI account or sign in: "
                "https://platform.openai.com/signup and then go to the API key page, "
                "https://platform.openai.com/account/api-keys, and create new secret key.")
            userEnteredCode = st.text_input("Please enter your API Key:")
            submit_button = st.form_submit_button(label='Authenticate My Key')
            st.session_state["holdMe"] = os.environ["OPENAI_API_KEY"]
    if submit_button and userEnteredCode and userEnteredCode != "":
        # We need to set our environmental variable
        if os.environ["MOSS_MASHER"] == userEnteredCode:
            print("Login complete")
        elif "sk-" not in userEnteredCode:
            st.error("Please enter a valid Open API key")
            return
        else:
            os.environ["OPENAI_API_KEY"] = userEnteredCode
        #We also create a subdirectory to avoid multiple user issues
        haveDir = False
        subPath = ""
        while not haveDir:
            try:
                subPath = os.path.join("./", "X-" + str(random.randrange(1,999999)))
                os.makedirs(subPath, exist_ok=False)
                haveDir = True
            except:
                print("Repeat dir")
        st.session_state["subPath"] = subPath
        st.markdown("Got your code!  Now click on Upload a Document or Enter Youtube Video!")


def create_pdf_page():
    st.title(":football: :green[The Moss Masher] :dog:")
    st.header("Mash your long documents and videos into easy answers!")
    container = st.container()

    with container:
        uploaded_file = st.file_uploader('Select your file and click Add File:', type=['pdf', 'docx', 'txt'])
        add_file = st.button('Add File', on_click=clear_history)

    if uploaded_file and add_file:
        with st.spinner('Retrieving your file, this may take a while...'):
            bytes_data = uploaded_file.read()
            file_name = st.session_state["subPath"] + '/' + uploaded_file.name
            with open(file_name, 'wb') as f:
                f.write(bytes_data)

            name, extension = os.path.splitext(file_name)
            if extension == '.pdf':
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_name)
            elif extension == '.docx':
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_name)
            elif extension == '.txt':
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_name)

            st.session_state["loader"] = loader
            st.write("Document retrieved!  Click on Ask Questions!")


def create_youtube_page():
    st.title(":football: :green[The Moss Masher] :dog:")
    st.header("Mash your long documents and videos into easy answers!")
    container = st.container()
    with container:
        youtube_url = st.text_input('Or enter your Youtube URL')
    if youtube_url:
        with st.spinner('Retrieving your video...'):
            loader = YoutubeLoader.from_youtube_url(youtube_url)
            st.session_state["loader"] = loader
        st.write("Video retrieved!  Click on Ask Questions!")


def ask_questions():
    st.title(":football: :green[The Moss Masher] :dog:")
    st.header("Mash your long documents and videos into easy answers!")
    st.markdown("<h3>Question Ideas:</h3>",unsafe_allow_html=True)
    st.markdown("<ul><li>Please summarize this document</li><li>What is this video about?</li></ul>",
                unsafe_allow_html=True)
    question = st.text_input('Enter your question here!')
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    aiTemp = st.sidebar.slider("How Strict (0) to Creative(10) do you want your responses:", min_value=0.0,
                               max_value=1.0, value=0.2, step=0.01)
    if question:
        with st.spinner('Running the Moss Masher....'):
            llm = select_model(model, aiTemp)
            contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                ]
            )
            contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
            qa_system_prompt = """You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            
            {context}"""
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                ]
            )
            loader = st.session_state["loader"]
            docs = loader.load()

            def contextualized_question(input: dict):
                if input.get("chat_history"):
                    return contextualize_q_chain
                else:
                    return input["question"]

            def format_docs(self):
                return "\n\n".join(doc.page_content for doc in docs)

            with st.spinner('Mashing your data into vectors, this may take a while...'):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)

                embeddings = OpenAIEmbeddings()
                vector_store = Chroma.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever()
                rag_chain = (
                        RunnablePassthrough.assign(
                            context=contextualized_question | retriever | format_docs
                        )
                        | qa_prompt
                        | llm
                )
            if 'history' not in st.session_state:
                st.session_state['history'] = []
            with st.spinner('Looking for your answer now....'):
                with get_openai_callback() as cb:
                    ai_msg = rag_chain.invoke({"question": question, "chat_history": st.session_state['history']})
                    st.session_state['costs'].append(cb.total_cost)
            if 'costs' not in st.session_state:
                st.session_state['costs'] = []
            with st.chat_message('Question:'):
                st.markdown(question)
            with st.chat_message('Answer'):
                st.markdown(ai_msg.content)
            for message in st.session_state['history']:
                if isinstance(message, AIMessage):
                    with st.chat_message('Answer'):
                        st.markdown(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message('Question'):
                        st.markdown(message.content)
            st.session_state['history'].extend([HumanMessage(content=question), ai_msg])
            costs = st.session_state.get('costs', [])
            st.sidebar.markdown("## Costs")
            st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
            for cost in costs:
                st.sidebar.markdown(f"- ${cost:.5f}")

def clear_files():
    for subdir, dirs, files in os.walk("./"):
        if subdir.startswith("."):
            for dir in dirs:
                thisDir = os.path.join("./", dir)
                if os.path.isdir(thisDir) and "X-" in thisDir:
                    try:
                        shutil.rmtree(thisDir)
                    except:
                        print("mep")
            break
def main():
    try:
        init_page()
        selection = st.sidebar.radio("Go to",
                                     ["Enter OpenAI Key", "Upload A Document", "Enter YouTube Video", "Ask Questions"])
        if selection == "Enter OpenAI Key":
            create_login_page()
        elif selection == "Upload A Document":
            create_pdf_page()
        elif selection == "Enter YouTube Video":
            create_youtube_page()
        elif selection == "Ask Questions":
            ask_questions()
    except:
        traceback.print_exc()
        st.error("Your document was too big for your API access level.  Please choose a smaller document and try again!")
    finally:
        print("Done")


if __name__ == "__main__":
    main()
