# app.py
import streamlit as st
import google.generativeai as genai
from src.chatbot import RAGChatbot

def initialize_chatbot():
    """Initialize the chatbot with API key"""
    if "chatbot" not in st.session_state:
        chatbot = RAGChatbot(st.session_state.google_api_key)
        chatbot.initialize_vector_stores()
        st.session_state.chatbot = chatbot

def main():
    st.title("Secure RAG Chatbot")
    
    # Google API Key input
    if "google_api_key" not in st.session_state:
        api_key = st.text_input("Enter your Google API Key:", type="password")
        # api_key ='AIzaSy'
        if api_key:
            st.session_state.google_api_key = api_key
            genai.configure(api_key=api_key)
            initialize_chatbot()
            st.rerun()
    
    # Login Section
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit and hasattr(st.session_state, "chatbot"):
                if st.session_state.chatbot.authenticate_user(email, password):
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    else:
        st.write(f"Logged in as: {st.session_state.user_email}")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question"):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                response = st.session_state.chatbot.get_response(
                    prompt, 
                    st.session_state.user_email
                )
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Logout button
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = None
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()