import os
import gradio as gr
from llama_index.core import Document, VectorStoreIndex
import openai
from dotenv import load_dotenv 

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Function to read content from a .txt file
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Read the content from vizdale.txt
doc_content = read_txt('vizdale.txt')

# Split the content into logical documents (for simplicity, here we assume each paragraph is a document)
documents = [Document(text=para.strip()) for para in doc_content.split('\n') if para.strip()]

# Load documents into LlamaIndex
index = VectorStoreIndex.from_documents(documents)

# Predefined follow-up questions based on keywords
follow_up_questions_map = {
    "services": "Can you tell me more about which specific services you are interested in?",
    "contact": "Would you like to know the different ways you can reach out to us? You can schedule a meeting directly through our Calendly link: https://www.vizdale.com/contact-us/",
    "digital transformation": "Are you looking for specific solutions in digital transformation?",
    "branding": "What aspects of branding are you most interested in?",
    "web technology": "Do you have a particular web technology requirement?",
    "website development": "What kind of website are you looking to develop?",
    "business grow": "Can you share more about your business goals?",
    "pricing": "Would you like to know about our pricing models in detail?",
    "case studies": "Are you interested in a particular industry for our case studies?",
    "industries": "Which industry are you inquiring about?"
}

# Function to generate follow-up question based on user query
def generate_follow_up_question(query):
    for key, question in follow_up_questions_map.items():
        if key in query.lower():
            return question
    return "Is there anything else you would like to know?"

# Function to interact with OpenAI's GPT-3.5-turbo using LlamaIndex
def get_response(query, chat_history):
    # Retrieve relevant document using LlamaIndex
    query_engine = index.as_query_engine()
    results = query_engine.query(query)

    # If results are empty, provide a fallback response
    response_text = results.response if results and results.response else "I'm not sure about that. Please contact our support team for more information. Here are some topics you can ask about: branding, web technology, digital transformation."

    # Generate a follow-up question
    follow_up_question = generate_follow_up_question(query)
    response_text += " " + follow_up_question
    
    return response_text

def chatbot_interface(user_input, chat_history):
    if chat_history is None:
        chat_history = []
    response = get_response(user_input, chat_history)
    chat_history.append((user_input, response))
    return response, chat_history

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=["text", "state"],
    outputs=["text", "state"],
    title="Vizdale Support Chatbot",
    description="I would like to support you. Ask me anything regarding your queries.",
    examples=[
        ["What services do you provide?"],
        ["How can we contact Vizdale?"],
        ["Can you explain your digital transformation services?"],
        ["Tell me more about your branding services."],
        ["What is Vizdale's approach to web technology?"],
        ["Do you offer support for website development?"],
        ["How can Vizdale help my business grow?"],
        ["What is your pricing model?"]
        # ["Do you have case studies or success stories?"],
        # ["What industries do you specialize in?"]
    ]
)

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860)
