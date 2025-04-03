import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import PyPDF2
import nltk
nltk.download('punkt')


# Load environment variables
load_dotenv()
groq_api_key = os.getenv("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")

# Initialize Groq client
client = Groq(api_key="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")

# Streamlit App
st.title("📄 Legal Assistance Chat!!")

# Default model selection (most efficient)
model = "Llama3-8b-8192"

# Initialize session state for chat history and PDF content
if "history" not in st.session_state:
    st.session_state.history = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# File uploader for PDF
uploaded_file = st.file_uploader("Upload Legal Doc", type=["pdf"])

if uploaded_file:
    # Extract text from the PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    st.session_state.pdf_text = pdf_text
    st.success("PDF content successfully uploaded and processed!")

# User input
user_input = st.text_input("Ask a question about the PDF:", "")

if st.button("Send"):
    # Call the Groq API with the PDF text as context
    try:
        if st.session_state.pdf_text:
            context = f"PDF Content: {st.session_state.pdf_text}\nUser Query: {user_input}"
        else:
            context = user_input

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": context}],
            model=model,
        )
        # Parse the response
        response = chat_completion.choices[0].message.content
        # Append to chat history
        st.session_state.history.append({"query": user_input, "response": response})
    except Exception as e:
        response = f"Error: {e}"
        st.session_state.history.append({"query": user_input, "response": response})

    # Display the response
    st.markdown(f"Bot: {response}")

# Display chat history
st.sidebar.title("Chat History")
for i, entry in enumerate(st.session_state.history):
    if st.sidebar.button(f"Query {i + 1}"):
        st.write(f"You: {entry['query']}")
        st.write(f"Chat: {entry['response']}")
