import os
import random
import streamlit as st  # used to create our UI frontend
from langchain_community.document_loaders import YoutubeLoader
from utilities import clear_files, clear_history

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
    if "rag_chain" in st.session_state:
        del st.session_state["rag_chain"]
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
    if "rag_chain" in st.session_state:
        del st.session_state["rag_chain"]
    with container:
        youtube_url = st.text_input('Or enter your Youtube URL')
    if youtube_url:
        with st.spinner('Retrieving your video...'):
            loader = YoutubeLoader.from_youtube_url(youtube_url)
            st.session_state["loader"] = loader
        st.write("Video retrieved!  Click on Ask Questions!")