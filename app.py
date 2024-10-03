import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Bedrock
import boto3
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain_community.llms.bedrock import Bedrock


# Load the existing Pinecone index
faiss_store_path = "faiss_store"

# Load Hugging Face model for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Generate embeddings using Hugging Face model
embeddings = HuggingFaceEmbeddings(model_name=model_name)

vectorstoredb = FAISS.load_local(faiss_store_path, embeddings, allow_dangerous_deserialization=True)

retriever = vectorstoredb.as_retriever()


boto_session = boto3.session.Session(
    aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name = st.secrets["REGION_NAME"]
)

# Load the Bedrock client using Boto3.
bedrock = boto_session.client(service_name='bedrock-runtime')


llm = Bedrock(
    model_id="meta.llama3-8b-instruct-v1:0", 
    client=bedrock, 
    model_kwargs={
        "max_gen_len": 512,  # Maximum generation length
        "temperature": 0.5,  # Temperature for controlling randomness
        "top_p": 0.9         # Nucleus sampling (top-p) parameter
    }
)

# system_prompt = (
#     "You are a knowledgeable and friendly assistant specialized in answering questions about MIDA Malaysia. "
#     "You provide natural, engaging, and insightful responses to users, covering all aspects of MIDA's role in "
#     "industrial development, investment opportunities, business growth, and economic policies in Malaysia. "
#     "Your goal is to help users understand how MIDA can support their investment and business efforts, "
#     "including providing detailed information about services, incentives, success stories, and market insights. "
#     "Be conversational and approachable, offering useful recommendations and insights whenever possible. "
#     "Encourage users to ask about various topics, including but not limited to MIDA's functions, investment processes, "
#     "industry sectors, and success stories. If you are unsure about something, use your external knowledge, "
#     "but ensure the information is reliable and closely related to MIDA or investments in Malaysia. "
#     "Only answer queries related to MIDA Malaysia or anything relevant to investment and business growth in Malaysia."
#     "Do NOT rephrase the question"
#     "Only provide the answer"
#     "Answer only the relevant portion of the context, don't include the template"
#     "\n\n"
#     "{context}"
# )

system_prompt = (
    "You are a knowledgeable and friendly assistant specialized in answering questions about MIDA Malaysia. "
    "Your goal is to provide clear, direct answers to the questions the user asks, without generating additional questions or follow-up responses. "
    "Only answer the specific question that the user provides. "
    "Do not introduce new questions, continue the conversation, or provide extra context unless asked. "
    "Do not include any prefixes such as 'Answer:' or 'Human:' in your response."
    "\n\n"
    "{context}"
)



# Define the prompt template for OpenAI models using ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def clean_response(response: str) -> str:
    # Stop at the first occurrence of a potential new question
    cleaned = response.split("Human:")[0].strip()  # Stop at 'Human:' if present
    # Remove trailing '**' or any unnecessary characters
    cleaned = cleaned.replace("**", "").strip()
    return cleaned




# Streamlit interface
st.title("MIDA Malaysia Conversational Chatbot")

# Initialize session state to store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial greeting from the assistant
    initial_message = "Hello! I'm here to help you with any questions you have about MIDA Malaysia and investment opportunities. Feel free to ask me anything!"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

# Display chat messages in collapsible sections (accordion-style)
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.expander(f"Question {idx // 2 + 1}: {message['content'][:50]}...", expanded=False):
            # Display the user's question
            st.write(f"**User:** {message['content']}")
            # Display the assistant's response (next message in the list)
            if idx + 1 < len(st.session_state.messages):
                st.write(f"**Assistant:** {st.session_state.messages[idx + 1]['content']}")

# Accept user input
if prompt := st.chat_input("Ask about MIDA Malaysia..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat
    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        # Retrieve relevant context and generate response
        result = rag_chain.invoke({
            "input": prompt
        })

        response = result['answer'] if 'answer' in result else "I'm not sure."

        # Made a change here and to the st markdown response
        cleaned_response = clean_response(response)

        st.markdown(cleaned_response)
    
    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})

# # Initialize session state to store conversation history
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#     # Initial greeting from the assistant
#     initial_message = "Hello! I'm here to help you with any questions you have about MIDA Malaysia and investment opportunities. Feel free to ask me anything!"
#     st.session_state.messages.append({"role": "assistant", "content": initial_message})

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if prompt := st.chat_input("Ask about MIDA Malaysia..."):

#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # Display user message in chat
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):

#         # Retrieve relevant context and generate response
#         result = rag_chain.invoke({
#             "input": prompt
#         })

#         response = result['answer'] if 'answer' in result else "I'm not sure."

#         st.markdown(response)
    
#     st.session_state.messages.append({"role": "assistant", "content": response})




















# from langchain.chains import create_history_aware_retriever
# from langchain_core.prompts import MessagesPlaceholder

# contextualize_q_system_prompt = (
#     "Given a chat history and the latest user question "
#     "which might reference context in the chat history, "
#     "formulate a standalone question which can be understood "
#     "without the chat history. Do NOT answer the question, "
#     "just reformulate it if needed and otherwise return it as is."
# )

# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}")
#     ]
# )

# history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}")
#     ]
# )


# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# chat_history = []
# question = "What is MIDA?"
# response1 = rag_chain.invoke({
#     "input": question,
#     "chat_history": chat_history
# })

# chat_history.extend(
#     [
#         HumanMessage(content=question),
#         AIMessage(content=response1["answer"])
#     ]
# )

# question2 = "Tell me more about it?"
# response2 = rag_chain.invoke({
#     "input": question,
#     "chat_history": chat_history
# })



