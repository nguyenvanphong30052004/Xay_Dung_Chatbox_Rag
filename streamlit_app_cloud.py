import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot - AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

class RAGChatbot:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    def create_sample_data(self):
        """Create sample data for demo"""
        return """
Artificial Intelligence (AI) là một lĩnh vực khoa học máy tính tập trung vào việc tạo ra các hệ thống có thể thực hiện các nhiệm vụ thường đòi hỏi trí thông minh của con người. AI bao gồm nhiều nhánh con như machine learning, deep learning, natural language processing, computer vision, và robotics.

Machine Learning (ML) là một nhánh của AI cho phép máy tính học hỏi và cải thiện từ dữ liệu mà không cần được lập trình rõ ràng. Có ba loại machine learning chính: supervised learning, unsupervised learning, và reinforcement learning.

Deep Learning là một tập con của machine learning sử dụng các mạng neural với nhiều lớp (deep neural networks) để học các mẫu phức tạp từ dữ liệu. Deep learning đã đạt được những thành tựu đáng kể trong các lĩnh vực như nhận dạng hình ảnh, xử lý ngôn ngữ tự nhiên, và chơi game.

Natural Language Processing (NLP) là lĩnh vực AI tập trung vào việc giúp máy tính hiểu, diễn giải và tạo ra ngôn ngữ tự nhiên của con người. NLP được sử dụng trong các ứng dụng như chatbot, dịch máy, phân tích cảm xúc, và tóm tắt văn bản.

Computer Vision là lĩnh vực AI nghiên cứu cách máy tính có thể hiểu và xử lý thông tin từ hình ảnh và video. Các ứng dụng bao gồm nhận dạng khuôn mặt, phát hiện đối tượng, phân đoạn hình ảnh, và xe tự lái.
        """
    
    def load_documents(self, text_content):
        """Load documents from text content"""
        paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
        documents = [Document(page_content=p, metadata={'source': 'sample_data'}) for p in paragraphs]
        return documents
    
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, chunks):
        """Create vector store from document chunks"""
        try:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OpenAI API key not found.")
                return False
            
            embeddings = OpenAIEmbeddings()
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False
    
    def setup_qa_chain(self):
        """Setup the question-answering chain"""
        try:
            if not self.vectorstore:
                st.error("Vector store not initialized.")
                return False
            
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
            return True
        except Exception as e:
            st.error(f"Error setting up QA chain: {e}")
            return False
    
    def ask_question(self, question):
        """Ask a question and get response"""
        try:
            if not self.qa_chain:
                return "Chatbot chưa được khởi tạo."
            
            result = self.qa_chain({"question": question})
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            response = f"**🤖 Trả lời:**\n{answer}\n\n"
            if source_docs:
                response += "**📚 Nguồn tham khảo:**\n"
                for i, doc in enumerate(source_docs[:2], 1):
                    response += f"{i}. {doc.page_content[:200]}...\n\n"
            
            return response
        except Exception as e:
            return f"Lỗi khi xử lý câu hỏi: {e}"
    
    def initialize_chatbot(self):
        """Initialize the chatbot"""
        with st.spinner("Đang khởi tạo chatbot..."):
            text_content = self.create_sample_data()
            documents = self.load_documents(text_content)
            chunks = self.split_documents(documents)
            st.success(f"Đã chia thành {len(chunks)} chunks")
            
            if not self.create_vectorstore(chunks):
                return False
            
            if not self.setup_qa_chain():
                return False
            
            st.success("Chatbot đã được khởi tạo thành công!")
            return True

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### Trợ lý AI thông minh với khả năng tìm kiếm và trả lời dựa trên tài liệu")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Nhập OpenAI API key của bạn"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize button
        if st.button("🚀 Khởi tạo Chatbot", type="primary"):
            if not api_key:
                st.error("Vui lòng nhập OpenAI API key!")
                return
            
            if 'chatbot' not in st.session_state:
                st.session_state.chatbot = RAGChatbot()
            
            if st.session_state.chatbot.initialize_chatbot():
                st.session_state.initialized = True
                st.success("Chatbot đã sẵn sàng!")
            else:
                st.error("Không thể khởi tạo chatbot!")
        
        # Clear chat button
        if st.button("🗑️ Xóa lịch sử chat"):
            if 'chat_history' in st.session_state:
                st.session_state.chat_history = []
            st.rerun()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    # Main chat area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("💬 Cuộc trò chuyện")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>Bạn:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Bot:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        if st.session_state.initialized:
            user_question = st.text_input(
                "Nhập câu hỏi của bạn...",
                key="user_input",
                placeholder="Ví dụ: AI là gì?"
            )
            
            if st.button("Gửi", key="send_button"):
                if user_question.strip():
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_question
                    })
                    
                    with st.spinner("🤖 Đang suy nghĩ..."):
                        response = st.session_state.chatbot.ask_question(user_question)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()
        else:
            st.info("👆 Vui lòng khởi tạo chatbot trong sidebar để bắt đầu chat!")
            
            st.markdown("### 📝 Câu hỏi mẫu")
            sample_questions = [
                "AI là gì và có những ứng dụng nào?",
                "Machine Learning có những loại nào?",
                "Deep Learning được sử dụng trong những lĩnh vực nào?",
                "NLP là gì và được ứng dụng như thế nào?",
                "Computer Vision có những ứng dụng gì?"
            ]
            
            for question in sample_questions:
                st.markdown(f"- {question}")

if __name__ == "__main__":
    main() 