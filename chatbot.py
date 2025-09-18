import os
import json
from typing import List, Dict
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

class BilingualChatbot:
    def __init__(self, file_paths: List[str], api_key: str):
        # Load the knowledge base
        self.knowledge_base = self.load_knowledge_base(file_paths)
        
        # Process and index all QA pairs for better retrieval
        self.process_qa_pairs()
        
        # Initialize the OpenAI client with OpenRouter configuration
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize embeddings and vector store for RAG
        self.embeddings = FakeEmbeddings(size=384)
        self.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
        
    def process_qa_pairs(self):
        self.qa_pairs = []
        
        for qa_pair in self.knowledge_base:
            # Process English QA pairs
            if 'question_en' in qa_pair and 'answer_en' in qa_pair:
                self.qa_pairs.append({
                    'keywords': self._extract_keywords(qa_pair['question_en']),
                    'language': 'en',
                    'qa_pair': qa_pair
                })
            
            # Process Tamil QA pairs
            if 'question_ta' in qa_pair and 'answer_ta' in qa_pair:
                self.qa_pairs.append({
                    'keywords': self._extract_keywords(qa_pair['question_ta']),
                    'language': 'ta',
                    'qa_pair': qa_pair
                })
    
    def _extract_keywords(self, text: str) -> Counter:
        # Simple keyword extraction: lowercase, split, and count words
        words = text.lower().split()
        return Counter(words)
        
        documents = []
        metadatas = []
        ids = []
        doc_id = 0
        
        for qa_pair in self.knowledge_base:
            # Handle English QA pairs
            if 'question_en' in qa_pair and 'answer_en' in qa_pair:
                documents.append(f"Question: {qa_pair['question_en']}\nAnswer: {qa_pair['answer_en']}")
                metadatas.append({
                    "language": "en",
                    "question": qa_pair['question_en'],
                    "answer": qa_pair['answer_en']
                })
                ids.append(f"doc_{doc_id}")
                doc_id += 1
            
            # Handle Tamil QA pairs
            if 'question_ta' in qa_pair and 'answer_ta' in qa_pair:
                documents.append(f"Question: {qa_pair['question_ta']}\nAnswer: {qa_pair['answer_ta']}")
                metadatas.append({
                    "language": "ta",
                    "question": qa_pair['question_ta'],
                    "answer": qa_pair['answer_ta']
                })
                ids.append(f"doc_{doc_id}")
                doc_id += 1
        
        # Add documents to the collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def load_knowledge_base(self, file_paths: List[str]) -> List[Dict]:
        knowledge_base = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list):
                    knowledge_base.extend(data)
                else:
                    knowledge_base.append(data)
        return knowledge_base

    def detect_language(self, text: str) -> str:
        # Simple language detection based on Unicode ranges
        for char in text:
            if '\u0B80' <= char <= '\u0BFF':  # Tamil Unicode range
                return 'tamil'
        return 'english'
    
    def get_context(self, query: str) -> str:
        language = self.detect_language(query)
        query_keywords = self._extract_keywords(query)
        
        # Calculate relevance scores for each QA pair
        scored_pairs = []
        for qa in self.qa_pairs:
            # Calculate keyword overlap score
            overlap = sum((query_keywords & qa['keywords']).values())
            
            # Apply language preference boost
            lang_boost = 1.2 if qa['language'] == ('ta' if language == 'tamil' else 'en') else 1.0
            
            score = overlap * lang_boost
            scored_pairs.append((score, qa['qa_pair'], qa['language']))
        
        # Sort by score and get top 3
        scored_pairs.sort(reverse=True)
        
        relevant_pairs = []
        for _, qa_pair, lang in scored_pairs[:3]:
            if lang == 'en':
                relevant_pairs.append(f"English Q: {qa_pair['question_en']}\nEnglish A: {qa_pair['answer_en']}")
            else:
                relevant_pairs.append(f"Tamil Q: {qa_pair['question_ta']}\nTamil A: {qa_pair['answer_ta']}")
        
        qa_context = "\n\n".join(relevant_pairs)
        
        # Get context from RAG
        docs = self.vectorstore.similarity_search(query, k=3)
        rag_context = "\n\n".join([doc.page_content for doc in docs])
        
        # Combine contexts
        full_context = f"QA Context:\n{qa_context}\n\nRAG Context:\n{rag_context}"
        
        return full_context
    
    def get_response(self, query: str) -> str:
        language = self.detect_language(query)
        context = self.get_context(query)
        
        system_prompt = f"""You are a bilingual government services assistant that specializes in Tamil Nadu government schemes and services.

        IMPORTANT RULES:
        1. If the user asks in Tamil, you MUST respond ONLY in Tamil
        2. If the user asks in English, you MUST respond ONLY in English
        3. Never mix languages in your response
        4. Give accurate and complete information based on the context
        5. If the exact information is not in the context, clearly state that but provide the most relevant available information
        6. For Tamil responses, use clear and simple Tamil that is easily understood
        
        Current language: {language}
        You MUST respond in: {"tamil" if language == "tamil" else "english"}
        
        Use this context to answer:
        {context}
        
        Previous conversation:
        {self.memory.buffer}
        """
        
        response = self.client.chat.completions.create(
            model="google/gemma-3-27b-it:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=1000,
            extra_headers={
                "HTTP-Referer": "https://github.com/chatbot",
                "X-Title": "Bilingual Government Services Chatbot"
            },
            extra_body={}
        )
        
        # Update memory
        self.memory.save_context({"input": query}, {"output": response.choices[0].message.content})
        
        return response.choices[0].message.content
        
        return response.content

def main():
    load_dotenv()
    # List of dataset files
    files = [
        "finetune_QA.json",
        "processed_rag_dept.json",
        "processed_rag_services.json",
        "rag_new_scheme.json",
        "tamil_scheme_data.json"
    ]
    
    # Initialize the chatbot
    chatbot = BilingualChatbot(
        file_paths=files,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print("Bilingual Chatbot initialized! (Type 'quit' to exit)")
    print("You can ask questions in English or Tamil about government services and schemes.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        try:
            response = chatbot.get_response(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
