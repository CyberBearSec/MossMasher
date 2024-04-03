import os
import random
import shutil
import streamlit as st
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

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
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)