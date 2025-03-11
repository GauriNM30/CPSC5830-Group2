import streamlit as st
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from RAG_With_QWEN import RAG

# Set up Streamlit page configuration - MUST be the first Streamlit command!
st.set_page_config(page_title="VisaWise", layout="centered", initial_sidebar_state="expanded")

# Terms and Conditions Popup
if "terms_accepted" not in st.session_state:
    st.session_state.terms_accepted = False

if not st.session_state.terms_accepted:
    st.markdown("<h2 style='text-align: center;'>Privacy Policy</h2>", unsafe_allow_html=True)
    st.write("Please read and accept our Terms and Conditions to continue using the application.")
    st.write("""
Visa Wise Last Comments 

Name: If our scope is only F-1 Visa, then the tool's name should be F Visa Wise or something that shows that it only answers F Visa questions. It might be considering misleading but it's up to you.  

Privacy: 

Data the App May Collect 

Disclaimer: the tool won't ask for your data on its own. The data collected will be shared by you, including but not limited to: 

- Name 
- Student ID 
- Passport Details 

Data Sharing of the information collected 

- International Student Service Center 
- Legal Compliance 
- Consented by you 
 
Your privacy is important to us, and we are committed to protecting your personal information. Please take a moment to review our Privacy Policy for more information on how we handle your data. 
 
have read the Privacy Policy and understand how my data will be collected, used and protected. 
 
 
 
    """)
    
    col1, col2 = st.columns(2)
    if col1.button("Accept"):
        st.session_state.terms_accepted = True
        st.rerun()  # Rerun the app after acceptance
    if col2.button("Decline"):
        st.write("You have declined the Terms and Conditions. The application will now exit.")
        st.stop()  # Halt app execution
    
    st.stop()  # Prevent the rest of the app from loading until terms are accepted

# Everything below only executes after Terms are accepted.

# Clear Cache
st.cache_data.clear()
st.cache_resource.clear()

# Directory for saving chat sessions
CHAT_DIR = Path("chat_sessions")
CHAT_DIR.mkdir(exist_ok=True)

# Initialize RAG instance and session state variables
if "rag" not in st.session_state:
    st.session_state.rag = RAG()
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_list" not in st.session_state:
    st.session_state.chat_list = [f.stem for f in CHAT_DIR.glob("*.json")]

# Helper functions for chat persistence
def save_chat(chat_id, messages):
    file_path = CHAT_DIR / f"{chat_id}.json"
    with open(file_path, "w") as f:
        json.dump({
            "id": chat_id,
            "timestamp": datetime.now().isoformat(),
            "messages": messages
        }, f, indent=2)

def load_chat(chat_id):
    file_path = CHAT_DIR / f"{chat_id}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return None

def delete_chat(chat_id):
    file_path = CHAT_DIR / f"{chat_id}.json"
    if file_path.exists():
        file_path.unlink()
    st.session_state.chat_list = [f.stem for f in CHAT_DIR.glob("*.json")]
    if st.session_state.current_chat == chat_id:
        st.session_state.current_chat = None
        st.session_state.messages = []

# Custom CSS for UI styling
st.markdown(
    """
    <style>
        /* Fixed Header */
        .fixed-title {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 15px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: black;
            z-index: 9999;
            border-bottom: 2px solid #ddd;
        }
        /* Push chat content down to avoid overlap */
        .main .block-container {
            padding-top: 80px !important;
        }
        /* Sidebar chat history container */
        .chat-container {
            max-height: 350px;
            overflow-y: auto;
        }
        .chat-entry {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #f9f9f9;
            padding: 8px 10px;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.2s;
            border: 1px solid #ddd;
            font-size: 14px;
        }
        .chat-entry:hover {
            background-color: #e6e6e6;
        }
        .delete-btn {
            background: none;
            border: none;
            cursor: pointer;
            color: red;
            font-size: 16px;
            padding: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Fixed Title
st.markdown('<div class="fixed-title">VisaWise</div>', unsafe_allow_html=True)

# Sidebar for new chat and chat history
with st.sidebar:
    st.header("New Chat")
    if st.button("+ New Chat", use_container_width=True):
        st.session_state.current_chat = None
        st.session_state.messages = []

    st.divider()
    st.header("Chat History")

    # Scrollable Chat History Container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for chat_id in st.session_state.chat_list:
        chat_data = load_chat(chat_id)
        label = "Empty Chat"
        if chat_data and chat_data["messages"]:
            for msg in chat_data["messages"]:
                if msg["role"] == "user":
                    label = msg["content"][:20] + ("..." if len(msg["content"]) > 20 else "")
                    break

        # Align chat label and delete button properly
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=chat_id, use_container_width=True):
                loaded_chat = load_chat(chat_id)
                if loaded_chat:
                    st.session_state.current_chat = chat_id
                    st.session_state.messages = loaded_chat["messages"]

        with col2:
            if st.button("🗑", key=f"del_{chat_id}", help="Delete this chat", use_container_width=True):
                delete_chat(chat_id)

    st.markdown('</div>', unsafe_allow_html=True)  # Close chat container

# Main chat area
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
st.markdown('</div>', unsafe_allow_html=True)  # Close chat container

# Handle new user input
if prompt := st.chat_input("Ask about F-1 OPT/CPT..."):
    # Append and display the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate the assistant's response
    try:
        response = st.session_state.rag.generate_answer(prompt)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        response = "Sorry, there was an error generating a response."
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # If this is a new chat, assign a UUID and update the chat list
    if not st.session_state.current_chat:
        st.session_state.current_chat = str(uuid.uuid4())
        st.session_state.chat_list.append(st.session_state.current_chat)
    # Save the updated chat session
    save_chat(st.session_state.current_chat, st.session_state.messages)

