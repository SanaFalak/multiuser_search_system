import json
from pathlib import Path
from typing import Dict, List
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.document_processor import DocumentProcessor

class RAGChatbot:
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0
        )
        self.processor = DocumentProcessor()
        self.vector_stores = {}  # Store by category instead of document ID
        self.user_stores = {}    # Store combined vectors per user
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Load configurations
        self.config_path = Path(__file__).parent.parent / "config"
        self.load_configurations()
        
    def load_configurations(self):
        """Load user access and document metadata configurations"""
        with open(self.config_path / "user_access.json", "r") as f:
            self.user_access = json.load(f)
        with open(self.config_path / "document_metadata.json", "r") as f:
            self.document_metadata = json.load(f)
    
    def initialize_vector_stores(self):
        """Initialize vector stores by category"""
        for category, metadata in self.document_metadata.items():
            print(f"Processing category: {category}")
            category_chunks = []
            category_metadatas = []
            
            for doc in metadata["documents"]:
                try:
                    doc_path = Path(__file__).parent.parent / doc["path"]
                    print(f"Processing document: {doc['title']} at {doc_path}")
                    chunks = self.processor.process_pdf(str(doc_path))
                    
                    if chunks:
                        category_chunks.extend(chunks)
                        # Add metadata for each chunk
                        category_metadatas.extend([{
                            "source": doc["id"],
                            "title": doc["title"],
                            "category": category,
                            "access_level": metadata["access_level"]
                        } for _ in range(len(chunks))])
                        print(f"Successfully processed: {doc['title']}")
                    else:
                        print(f"Warning: No content extracted from {doc['title']}")
                        
                except FileNotFoundError:
                    print(f"Warning: Document not found - {doc['title']} at {doc_path}")
                except Exception as e:
                    print(f"Error processing document {doc['title']}: {str(e)}")
            
            if category_chunks:
                # Create one vector store per category
                self.vector_stores[category] = FAISS.from_texts(
                    texts=category_chunks,
                    embedding=self.embeddings,
                    metadatas=category_metadatas
                )
                print(f"Created vector store for category: {category}")

    def get_user_accessible_categories(self, user_email: str) -> List[str]:
        """Get list of categories accessible to user"""
        if user_email not in self.user_access:
            return []
        return self.user_access[user_email]["accessible_docs"]

    def authenticate_user(self, email: str, password: str) -> bool:
        """Authenticate user credentials"""
        return (email in self.user_access and 
                self.user_access[email]["password"] == password)

    # def combine_user_stores(self, user_email: str) -> FAISS:
    #     """Combine vector stores for user's accessible categories"""
    #     accessible_categories = self.get_user_accessible_categories(user_email)
        
    #     if not accessible_categories or not self.vector_stores:
    #         return None

    #     # Get available stores for accessible categories
    #     available_stores = []
    #     for category in accessible_categories:
    #         if category in self.vector_stores:
    #             available_stores.append(self.vector_stores[category])

    #     if not available_stores:
    #         return None

    #     # Use the first store as base
    #     combined_store = available_stores[0]
        
    #     # Merge remaining stores if any
    #     for store in available_stores[1:]:
    #         combined_store.merge_from(store)
            
    #     return combined_store

    def combine_user_stores(self, user_email: str) -> FAISS:
        """Combine vector stores for user's accessible categories."""
        accessible_categories = self.get_user_accessible_categories(user_email)
        
        if not accessible_categories or not self.vector_stores:
            return None

        available_stores = [
            self.vector_stores[category]
            for category in accessible_categories
            if category in self.vector_stores
        ]

        if not available_stores:
            return None

        # Merge stores
        combined_store = available_stores[0]
        for store in available_stores[1:]:
            combined_store.merge_from(store)

        return combined_store


    def get_response(self, query: str, user_email: str) -> str:
        """Get response based on user's access level."""
        if user_email not in self.user_access:
            return "Invalid user email."
        
        try:
            # Handle generic messages like "Hi" or "Hello"
            if query.strip().lower() in ["hi", "hello", "hey"]:
                return "Hello! How can I assist you today?"

            # Get or create combined store for user
            if user_email not in self.user_stores:
                combined_store = self.combine_user_stores(user_email)
                if combined_store is None:
                    return "You do not have permission to access any documents."
                self.user_stores[user_email] = combined_store

            # Create QA chain with user's combined store
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.user_stores[user_email].as_retriever(),
                memory=self.memory,
                return_source_documents=True
            )
            
            # Get response
            response = qa_chain({"question": query})
            
            # Format source information
            sources_info = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    source_info = (
                        f"{doc.metadata['title']} "
                        f"({doc.metadata['category']})"
                    )
                    if source_info not in sources_info:
                        sources_info.append(source_info)
            
            # Construct final response
            answer = response["answer"]
            if sources_info:
                sources_text = "\n\nInformation sourced from:\n" + "\n".join(f"- {s}" for s in sources_info)
                return f"{answer}{sources_text}"
            
            return answer
        
        except Exception as e:
            return f"Error processing your query: {str(e)}"

    def reset_user_session(self, user_email: str):
        """Reset the conversation memory and user's combined store"""
        if user_email in self.user_stores:
            del self.user_stores[user_email]
        self.memory.clear()