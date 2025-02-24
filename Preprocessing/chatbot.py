import streamlit as st
import RAG_With_QWEN  # Import your RAG class
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Clear session state to reset history
def clear_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# Initialize the RAG object
if "rag_obj" not in st.session_state:
    st.session_state.rag_obj = RAG_With_QWEN.RAG()

# Initialize current chat variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "new_chat" not in st.session_state:
    st.session_state.new_chat = False
if "current_chat_label" not in st.session_state:
    st.session_state.current_chat_label = None

# Create directory to store chat sessions
chat_dir = "chat_sessions"
if not os.path.exists(chat_dir):
    os.makedirs(chat_dir)

# Function to save chat session
def save_chat_session(label, messages):
    chat_path = os.path.join(chat_dir, f"{label}.json")
    with open(chat_path, "w") as f:
        json.dump(messages, f)

# Function to load chat session
def load_chat_session(label):
    chat_path = os.path.join(chat_dir, f"{label}.json")
    if os.path.exists(chat_path):
        with open(chat_path, "r") as f:
            return json.load(f)
    return []

# Function to delete a chat session
def delete_chat_session(label):
    chat_path = os.path.join(chat_dir, f"{label}.json")
    if os.path.exists(chat_path):
        os.remove(chat_path)

# Function to extract important keywords from text
def extract_keywords(text, num_keywords=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return "_".join(keywords)

# Create a two-column layout (left panel: 30%, right panel: 70%)
col1, col2 = st.columns([3, 7])

# LEFT PANEL: Chat Sessions and New Chat button
with col1:
    st.header("Chat Sessions")
    existing_chats = os.listdir(chat_dir)
    
    if existing_chats:
        # Loop over all the chat files and display them as buttons
        for chat_file in existing_chats:
            chat_label = chat_file.replace(".json", "")
            
            # Creating a clickable button for each chat session
            col1_button, col1_delete = st.columns([8, 2])
            
            with col1_button:
                if st.button(f"{chat_label}", key=f"chat_{chat_label}"):
                    st.session_state.messages = load_chat_session(chat_label)
                    st.session_state.current_chat_label = chat_label
                    st.session_state.new_chat = False
            
            with col1_delete:
                # Using the delete icon as a clickable button
                if st.button("\U0001F5D1", key=f"delete_{chat_label}", help="Delete this chat"):
                    delete_chat_session(chat_label)
                    st.session_state.messages = []
                    st.session_state.current_chat_label = None
                    st.session_state.new_chat = False
                    st.rerun()  # Re-run to refresh the UI

    # Option to start a new chat
    if st.button("Start New Chat"):
        st.session_state.new_chat = True
        st.session_state.messages = []
        st.session_state.current_chat_label = None

# RIGHT PANEL: Chat Interface
with col2:
    st.title("LLM Capstone Chatbot")
    
    if st.session_state.new_chat:
        st.session_state.messages = []
        st.session_state.current_chat_label = None
        st.session_state.new_chat = False

    # Container for chat messages (above)
    chat_container = st.container()
    # Container for input bar (at the bottom)
    input_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    with input_container:
        user_input = st.chat_input("Ask me anything about F-1 OPT/CPT...")
        if user_input:
            # Append user message if not already present
            if not any(m["role"] == "user" and m["content"] == user_input for m in st.session_state.messages):
                st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            # Generate assistant response
            response = st.session_state.rag_obj.generate_answer(user_input)
            if not any(m["role"] == "assistant" and m["content"] == response for m in st.session_state.messages):
                st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)

    # Set chat label if it's a new chat session
    if st.session_state.messages and st.session_state.current_chat_label is None:
        first_message = st.session_state.messages[0]["content"]

        # Extract keywords from the first message
        label = extract_keywords(first_message)

        # Clean up the label to ensure valid file naming
        st.session_state.current_chat_label = label
        save_chat_session(label, st.session_state.messages)
    elif st.session_state.current_chat_label:
        save_chat_session(st.session_state.current_chat_label, st.session_state.messages)
