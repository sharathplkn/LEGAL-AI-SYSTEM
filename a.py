from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from transformers import pipeline, AutoTokenizer
import gradio as gr
import re
import warnings
warnings.filterwarnings('ignore')

class EnhancedLegalAssistant:
    def __init__(self):
        print("Initializing Legal Assistant...")
        
        # Initialize embeddings
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector database
        print("Loading vector database...")
        try:
            self.db = Chroma(
                persist_directory="vector_db",
                embedding_function=self.embeddings
            )
            print("✓ Vector database loaded successfully")
        except Exception as e:
            print(f"✗ Error loading vector database: {e}")
            print("Make sure you have created the vector database first with your IPC documents")
            self.db = None
        
        # Setup retriever
        if self.db:
            self.retriever = self.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        else:
            self.retriever = None
        
        # Initialize language model
        print("Loading language model...")
        try:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Fix the generation config warning
            self.pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )
            
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            print("✓ Language model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading language model: {e}")
            self.llm = None
        
        # Initialize LangChain memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Keywords to identify legal queries
        self.legal_keywords = [
            'ipc', 'section', 'law', 'legal', 'court', 'crime', 'punishment',
            'offence', 'penalty', 'fine', 'imprisonment', 'arrest', 'bail',
            'murder', 'theft', 'robbery', 'assault', 'cheating', 'rape',
            'harassment', 'dowry', 'divorce', 'marriage', 'property',
            'constitution', 'supreme court', 'high court', 'judgment',
            'advocate', 'lawyer', 'petition', 'fir', 'complaint'
        ]
        
        print("✓ Legal Assistant initialized successfully!")
    
    def is_legal_query(self, message):
        """Check if the query is IPC/law related"""
        if not message:
            return False
            
        message_lower = message.lower()
        
        # Check for IPC section pattern
        section_pattern = r'(section|ipc|sec)[.\s]*(\d+[A-Z]?)'
        if re.search(section_pattern, message_lower):
            return True
        
        # Check for keywords
        return any(keyword in message_lower for keyword in self.legal_keywords)
    
    def extract_section_numbers(self, message):
        """Extract IPC section numbers from query"""
        if not message:
            return []
            
        pattern = r'(?:section|ipc|sec)[.\s]*(\d+[A-Z]?)'
        matches = re.findall(pattern, message.lower())
        return matches
    
    def get_chat_history(self):
        """Get formatted chat history from memory"""
        if not self.memory.chat_memory.messages:
            return ""
        
        history_str = ""
        for msg in self.memory.chat_memory.messages[-4:]:  # Last 4 messages
            if isinstance(msg, HumanMessage):
                history_str += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_str += f"Assistant: {msg.content}\n"
        
        return history_str
    
    def generate_response(self, message, history):
        """Main response generation function"""
        try:
            if not self.llm:
                return "I'm having trouble connecting to my language model. Please check the installation."
            
            # Check if it's a legal query
            is_legal = self.is_legal_query(message)
            
            # Simple greeting detection
            greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
            is_greeting = any(greet in message.lower() for greet in greetings)
            
            if is_greeting:
                response = "Hello! I'm your IPC Legal Assistant. How can I help you with Indian Penal Code questions today? You can ask me about specific sections, legal concepts, or general IPC information."
                # Save to memory
                self.memory.chat_memory.add_user_message(message)
                self.memory.chat_memory.add_ai_message(response)
                return response
            
            if is_legal and self.retriever:
                # Extract section numbers if present
                section_numbers = self.extract_section_numbers(message)
                
                # Try to retrieve relevant documents
                try:
                    context_docs = self.retriever.invoke(message)
                    
                    # Get chat history from memory
                    chat_history = self.get_chat_history()
                    
                    # Format context from retrieved documents
                    if context_docs:
                        context_parts = []
                        for i, doc in enumerate(context_docs[:3], 1):
                            content = doc.page_content
                            if len(content) > 500:
                                content = content[:500] + "..."
                            
                            source = doc.metadata.get('source', 'IPC Database')
                            section = doc.metadata.get('section', 'Unknown')
                            
                            context_parts.append(f"[Reference {i} - Section {section}]:\n{content}")
                        
                        context = "\n\n".join(context_parts)
                        
                        # Create prompt with context and history
                        prompt = f"""You are an AI Legal Assistant specializing in Indian Penal Code (IPC). 
Use the following legal context to answer the user's question accurately.

Chat History:
{chat_history}

Legal Context:
{context}

User Question: {message}

Instructions:
Answer strictly using the legal context provided.

1. Identify the IPC section mentioned in the context.
2. Explain the section in simple language.
3. Mention the punishment exactly as stated.
4. Do not mix information from different sections.
5. If the context does not contain the answer, say so.
Answer:"""
                    else:
                        prompt = f"""You are an AI Legal Assistant specializing in Indian Penal Code (IPC).

Chat History:
{chat_history}

User Question: {message}

I don't have specific information about this in my database. Here's what I can tell you:
1. This appears to be a legal question about Indian law
2. To get accurate information, please consult:
   - Official IPC documents
   - A qualified lawyer
   - Reputable legal sources

Could you please:
- Rephrase your question
- Specify which IPC section you're asking about
- Ask about a different IPC topic

General Response:"""
                    
                except Exception as e:
                    print(f"Retrieval error: {e}")
                    prompt = f"""You are an AI Legal Assistant. The user asks: {message}

I'm having trouble accessing my legal database. Please try:
1. Asking about specific IPC sections
2. Rephrasing your question
3. Checking your connection

Response:"""
            
            else:
                # General conversation - get history from memory
                chat_history = self.get_chat_history()
                
                prompt = f"""You are a helpful AI Legal Assistant specializing in Indian law. 

Chat History:
{chat_history}

User: {message}

Keep your response friendly but brief. If the conversation goes to non-legal topics, 
politely steer it back to IPC-related matters.

Assistant:"""
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            # Clean up response
            if isinstance(response, str):
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
                elif "Response:" in response:
                    response = response.split("Response:")[-1].strip()
                elif "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                
                # Remove any remaining prompt artifacts
                response = re.sub(r'User:.*?$', '', response, flags=re.MULTILINE)
                response = re.sub(r'Instructions:.*?$', '', response, flags=re.MULTILINE | re.DOTALL)
                
                # Clean up whitespace
                response = ' '.join(response.split())
                
                # Truncate if too long
                if len(response) > 1000:
                    response = response[:1000] + "..."
            
            # Add disclaimer for legal queries
            if is_legal and "disclaimer" not in response.lower():
                response += "\n\n---\n*⚠️ This information is for educational purposes only. Please consult a qualified lawyer for legal advice.*"
            
            # Save to memory
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(response)
            
            return response if response else "I'm not sure how to respond to that. Could you please rephrase your question?"
            
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return f"I encountered an error: {str(e)}. Please try again with a different question."
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        return "Conversation memory cleared!"

# Initialize the assistant
print("="*50)
print("Starting IPC Legal Assistant with Memory")
print("="*50)
assistant = EnhancedLegalAssistant()

def chat_interface(message, history):
    """Gradio interface function"""
    if not message or message.strip() == "":
        return "Please enter a question."
    return assistant.generate_response(message, history)

# Create Gradio interface - FOR GRADIO 3.x
with gr.Blocks(theme=gr.themes.Soft(), title="IPC Legal Assistant with Memory") as demo:
    gr.Markdown("""
    # ⚖️ IPC Legal Assistant with Memory
    
    Welcome! I'm your AI assistant specialized in **Indian Penal Code (IPC)**. 
    I remember our conversation so you can ask follow-up questions!
    
    ### 📋 Try asking:
    - "What is Section 302?"
    - "Explain punishment for theft"
    - "What is culpable homicide?"
    - "Tell me about Section 376"
    """)
    
    # For Gradio 3.x, remove the 'type' parameter
    chatbot = gr.Chatbot(height=400)
    
    with gr.Row():
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Ask about IPC sections, legal concepts...",
            scale=4,
            lines=2
        )
        submit = gr.Button("Submit", variant="primary", scale=1)
    
    with gr.Row():
        clear_chat = gr.Button("Clear Chat")
        clear_memory = gr.Button("Clear Memory", variant="secondary")
    
    with gr.Row():
        examples = gr.Examples(
            examples=[
                "What is IPC Section 302?",
                "Explain punishment for theft",
                "Tell me about Section 376",
                "What is the difference between murder and culpable homicide?",
                "What was the last section I asked about?"
            ],
            inputs=msg
        )
    
    gr.Markdown("""
    ---
    *⚠️ Disclaimer: This AI assistant provides general information for educational purposes only. 
    Not a substitute for professional legal advice.*
    """)
    
    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        
        response = assistant.generate_response(message, chat_history)
        
        # For Gradio 3.x, chat_history is a list of tuples
        if chat_history is None:
            chat_history = []
        
        # Append as tuple (user, assistant) for Gradio 3.x
        chat_history.append((message, response))
        
        return "", chat_history
    
    def clear_all():
        """Clear both chat display and memory"""
        assistant.clear_memory()
        return []
    
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear_chat.click(lambda: [], None, chatbot, queue=False)
    clear_memory.click(clear_all, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        debug=True
    )