import streamlit as st
from chains import Chain
import tempfile

# Initialize the Chain instance
chain_instance = Chain()

# Streamlit app layout
st.title("PDF Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.write("PDF uploaded successfully!")

    # Parse the PDF and create embeddings
    chunks = chain_instance.pdf_parser(temp_file_path)
    db = chain_instance.create_embeddings(chunks)
    qa_chain = chain_instance.create_conversational_retrieval_chain(db)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Ask questions
    st.subheader("Ask questions about the uploaded PDF")
    user_input = st.text_input("Enter your question")

    if st.button("Submit"):
        if user_input:
            # Query the chain
            answer = chain_instance.query_chain(
                qa_chain, user_input, st.session_state.chat_history
            )

            # Append to chat history
            st.session_state.chat_history.append((user_input, answer))

            # Display the answer
            st.write("Answer:", answer)

        # Display chat history
       # st.write("Chat History:")
        #for q, a in st.session_state.chat_history:
         #   st.write(f"**Q:** {q}")
          #  st.write(f"**A:** {a}")
