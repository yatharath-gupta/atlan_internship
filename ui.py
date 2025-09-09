import time
from prompts import get_classification_prompt, get_customer_response_prompt
import streamlit as st
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai
import re
from urllib.parse import urlparse
import pandas as pd
from sample import SAMPLE

# Fix SQLite issue before importing ChromaDB
try:
    # Fix for SQLite version issue
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError as e:
    st.error(f"ChromaDB import failed: {e}")
    CHROMADB_AVAILABLE = False
    chromadb = None
except RuntimeError as e:
    st.error(f"ChromaDB runtime error: {e}")
    CHROMADB_AVAILABLE = False
    chromadb = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with metadata"""
    content: str
    url: str
    title: str
    chunk_id: str
    chunk_index: int
    similarity_score: float
    token_count: int
    metadata: Dict

@dataclass
class RAGResponse:
    """Complete RAG response with answer and citations"""
    query: str
    answer: str
    sources: List[RetrievedChunk]
    confidence_score: float
    response_metadata: Dict

@dataclass
class TicketClassification:
    """Ticket classification result"""
    topic: str
    sentiment: str
    priority: str
    confidence_scores: Dict[str, float]

class AtlanRAGSystem:
    def __init__(self, 
                 gemini_api_keys: List[str],
                 chromadb_config: Dict[str, str] = None,
                 generation_model: str = "gemini-2.0-flash-exp",
                 embedding_model: str = "models/text-embedding-004",
                 collection_name: str = "atlan_docs"):
        
        self.gemini_api_keys = gemini_api_keys
        self.current_key_index = 0
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.chromadb_available = CHROMADB_AVAILABLE
        
        # Initialize ChromaDB only if available
        if self.chromadb_available and chromadb_config and chromadb:
            try:
                self.chroma_client = chromadb.CloudClient(**chromadb_config)
                self.collection = self.chroma_client.get_collection(collection_name)
                
                # Check collection status
                collection_count = self.collection.count()
                logger.info(f"AtlanRAGSystem initialized successfully. Collection has {collection_count} documents.")
                
                if collection_count == 0:
                    logger.warning("⚠️ ChromaDB collection is empty! Please run the embedding generation script first.")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self.chromadb_available = False
                self.chroma_client = None
                self.collection = None
        else:
            logger.warning("ChromaDB not available - running in classification-only mode")
            self.chroma_client = None
            self.collection = None
        
        # Configure initial API key
        self._configure_gemini()
    
    def _configure_gemini(self):
        """Configure Gemini API with current key"""
        genai.configure(api_key=self.gemini_api_keys[self.current_key_index])
    
    def _rotate_api_key(self):
        """Rotate to next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
        self._configure_gemini()
        logger.info(f"Rotated to API key {self.current_key_index + 1}")

    def load_sample_tickets():
        """Load sample tickets for demonstration"""
        sample_tickets = SAMPLE
        logger.info(f"Loading {len(sample_tickets)} sample tickets")
        
        # Transform the data structure to match expected format
        transformed_tickets = []
        for i, ticket in enumerate(sample_tickets):
            transformed_ticket = {
                "id": ticket["id"],
                "text": ticket["body"],  # Map 'body' to 'text'
                "subject": ticket["subject"]
            }
            logger.info(f"Ticket {i+1}: ID={transformed_ticket['id']}, text_length={len(transformed_ticket['text'])}")
            transformed_tickets.append(transformed_ticket)
        
        return transformed_tickets
    
    def classify_ticket(self, ticket_text: str, max_retries: int = 3) -> TicketClassification:
        """Classify a ticket for topic, sentiment, and priority"""
        
        # Debug: Check if ticket_text is empty
        if not ticket_text or not ticket_text.strip():
            logger.error(f"Empty ticket text provided: '{ticket_text}'")
            return TicketClassification(
                topic="Product",
                sentiment="Neutral",
                priority="P2",
                confidence_scores={"topic": 0, "sentiment": 0, "priority": 0}
            )
        
        logger.info(f"Classifying ticket with text length: {len(ticket_text)}")
        
        try:
            classification_prompt = get_classification_prompt(ticket_text=ticket_text)
            
            # Debug: Check if prompt is empty
            if not classification_prompt or not classification_prompt.strip():
                logger.error(f"Empty classification prompt generated for ticket: '{ticket_text[:50]}...'")
                return TicketClassification(
                    topic="Product",
                    sentiment="Neutral",
                    priority="P2",
                    confidence_scores={"topic": 0, "sentiment": 0, "priority": 0}
                )
            
            logger.info(f"Generated prompt length: {len(classification_prompt)}")
            
        except Exception as e:
            logger.error(f"Error generating classification prompt: {e}")
            return TicketClassification(
                topic="Product",
                sentiment="Neutral",
                priority="P2",
                confidence_scores={"topic": 0, "sentiment": 0, "priority": 0}
            )

        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(self.generation_model)
                response = model.generate_content(
                    classification_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,
                        top_p=0.8,
                        max_output_tokens=1024,
                    )
                )
                
                # Parse JSON response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()
                
                result = json.loads(response_text)
                
                return TicketClassification(
                    topic=result['topic'],
                    sentiment=result['sentiment'],
                    priority=result['priority'],
                    confidence_scores=result['confidence_scores']
                )
                
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "rate limit" in error_msg:
                    logger.warning(f"Rate limit hit during classification, rotating API key...")
                    self._rotate_api_key()
                    continue
                else:
                    logger.error(f"Error classifying ticket (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        # Return default classification
                        return TicketClassification(
                            topic="Product",
                            sentiment="Neutral",
                            priority="P2",
                            confidence_scores={"topic": 0, "sentiment": 0, "priority": 0}
                        )
        
        # Fallback classification
        return TicketClassification(
            topic="Product",
            sentiment="Neutral", 
            priority="P2",
            confidence_scores={"topic": 0, "sentiment": 0, "priority": 0}
        )
    
    def generate_query_embedding(self, query: str, max_retries: int = 3) -> List[float]:
        """Generate embedding for the query with retry logic"""
        if not self.chromadb_available:
            raise Exception("ChromaDB not available - cannot generate embeddings")
            
        for attempt in range(max_retries):
            try:
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=query,
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768
                )
                return response['embedding']
            
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "rate limit" in error_msg:
                    logger.warning(f"Rate limit hit, rotating API key...")
                    self._rotate_api_key()
                    continue
                else:
                    logger.error(f"Error generating query embedding (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        raise
        
        raise Exception("Failed to generate query embedding after all retries")
    
    def retrieve_relevant_chunks(self, 
                                query: str, 
                                top_k: int = 15,
                                similarity_threshold: float = 0.1) -> List[RetrievedChunk]:
        """Retrieve relevant chunks from ChromaDB"""
        
        if not self.chromadb_available or not self.collection:
            logger.warning("ChromaDB not available - returning empty chunks")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert to RetrievedChunk objects
            retrieved_chunks = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    similarity_score = 1 - distance
                    
                    if similarity_score >= similarity_threshold:
                        chunk = RetrievedChunk(
                            content=doc,
                            url=metadata.get('url', ''),
                            title=metadata.get('title', ''),
                            chunk_id=metadata.get('chunk_id', ''),
                            chunk_index=metadata.get('chunk_index', 0),
                            similarity_score=similarity_score,
                            token_count=metadata.get('token_count', 0),
                            metadata=metadata
                        )
                        retrieved_chunks.append(chunk)
                
                retrieved_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return retrieved_chunks
        
        except Exception as e:
            logger.error(f"Error in retrieve_relevant_chunks: {e}")
            return []
        

    def _is_response_helpful(self, response: str, query: str) -> bool:
        """Check if the response is helpful and comprehensive"""
        if not response or len(response.strip()) < 100:
            return False
        
        # Check for unhelpful patterns
        unhelpful_patterns = [
            "i don't have enough information",
            "please contact support",
            "i cannot help",
            "not available in the documentation",
            "unable to provide",
            "contact your administrator"
        ]
        
        response_lower = response.lower()
        for pattern in unhelpful_patterns:
            if pattern in response_lower:
                return False
        
        # Check for helpful indicators
        helpful_patterns = [
            "you can",
            "try",
            "follow these steps",
            "here's how",
            "to do this",
            "configure",
            "set up",
            "navigate to",
            "click on"
        ]
        
        helpful_count = sum(1 for pattern in helpful_patterns if pattern in response_lower)
        return helpful_count >= 2  # At least 2 helpful indicators
    
    def _generate_fallback_response(self, query: str, classification: TicketClassification) -> str:
        """Generate a fallback response when primary generation fails"""
        topic = classification.topic
        
        # Topic-specific fallback responses
        fallback_responses = {
            "How-to": f"""I understand you're looking for guidance on how to accomplish something in Atlan. While I don't have specific documentation for your exact question, I can suggest some general approaches:

1. **Check the Atlan Help Center**: Visit the official documentation for step-by-step guides
2. **Use the Search Function**: Try searching within your Atlan instance for relevant features
3. **Explore the Interface**: Many features have tooltips and help text when you hover over them
4. **Community Resources**: Check if there are community forums or user groups for additional insights

Based on your question about "{query[:100]}...", you might want to look into the relevant section of the platform that deals with this functionality.""",

            "Product": f"""I see you have a question about Atlan's product features. While I don't have specific information about your particular scenario, here are some general suggestions:

1. **Feature Documentation**: Check the official product documentation for detailed feature explanations
2. **Product Updates**: Look for recent release notes that might address your question
3. **Best Practices**: Consider if there are established best practices for your use case
4. **Alternative Approaches**: There might be multiple ways to achieve what you're looking for

For your specific question about "{query[:100]}...", I'd recommend exploring the relevant product area in more detail.""",

            "Connector": f"""Regarding your connector question, here are some general troubleshooting and guidance steps:

1. **Connector Documentation**: Check the specific connector's documentation for setup and configuration details
2. **Connection Settings**: Verify all connection parameters are correct
3. **Permissions**: Ensure proper permissions are set on both Atlan and the source system
4. **Network Connectivity**: Confirm network access between systems
5. **Logs and Diagnostics**: Check system logs for any error messages

For your question about "{query[:100]}...", these steps might help identify the issue or provide the guidance you need.""",

            "API/SDK": f"""For API and SDK related questions, here are some general approaches:

1. **API Documentation**: Refer to the official API documentation for endpoints and parameters
2. **Authentication**: Ensure you have proper API keys and authentication setup
3. **Rate Limits**: Check if you're hitting any API rate limits
4. **SDK Examples**: Look for code examples in the SDK documentation
5. **Error Handling**: Implement proper error handling in your code

Regarding your question about "{query[:100]}...", these resources should help you find the specific implementation details you need."""
        }
        
        # Get topic-specific response or default
        fallback = fallback_responses.get(topic, f"""I understand you have a question about {topic}. While I don't have specific documentation for your exact scenario, I recommend:

1. **Official Documentation**: Check Atlan's help center and documentation
2. **Support Resources**: Look for relevant guides and tutorials
3. **Community**: Consider reaching out to user communities for insights
4. **Experimentation**: Try exploring the relevant features in a test environment

For your specific question about "{query[:100]}...", these approaches should help you find the information or solution you're looking for.""")
        
        return fallback

    
    def generate_rag_response(self, query: str, classification: TicketClassification, max_retries: int = 3) -> RAGResponse:
        """Generate RAG response with improved fallback handling"""
        
        # Retrieve relevant chunks (always attempt, regardless of topic)
        chunks = []
        if self.chromadb_available:
            chunks = self.retrieve_relevant_chunks(query, top_k=15, similarity_threshold=0.05)  # Lower threshold for more results
            logger.info(f"Retrieved {len(chunks)} chunks for query")
        else:
            logger.info("ChromaDB not available, using expert knowledge mode")
        
        # Determine response strategy based on available context
        if chunks and len(chunks) >= 3:
            # High confidence - good context available
            context = "\n".join([f"Source {i+1}: {chunk.content}" for i, chunk in enumerate(chunks[:8])])  # More context
            confidence_boost = 0.2
            response_type = "full_context"
            logger.info("Using full context mode with 8 chunks")
        elif chunks and len(chunks) >= 1:
            # Medium confidence - some context available
            context = "\n".join([f"Source {i+1}: {chunk.content}" for i, chunk in enumerate(chunks[:5])])
            confidence_boost = 0.1
            response_type = "partial_context"
            logger.info(f"Using partial context mode with {len(chunks)} chunks")
        else:
            # Low confidence - no specific context, but still attempt to help
            context = f"""No specific documentation found for this query. However, as an Atlan expert, I can provide guidance based on general platform knowledge and common patterns for {classification.topic} questions.
            
    Based on the topic '{classification.topic}' and your question, I will draw from my knowledge of data catalog and governance platforms to provide helpful guidance."""
            confidence_boost = 0.0
            response_type = "expert_knowledge"
            logger.info("Using expert knowledge mode - no specific documentation found")
        
        # Initialize answer
        answer = ""
        
        # Generate response with multiple attempts
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating response attempt {attempt + 1}/{max_retries}")
                
                # Modify context for retry attempts to encourage more detailed responses
                enhanced_context = context
                if attempt > 0:
                    enhanced_context += f"\n\nIMPORTANT: Previous attempts were not comprehensive enough. Provide MORE detailed, actionable guidance with specific steps, examples, and alternatives. Do NOT simply refer to support - be helpful and thorough."
                
                response_prompt = get_customer_response_prompt(
                    query=query,
                    context=enhanced_context,
                    classification=classification
                )

                model = genai.GenerativeModel(self.generation_model)
                response = model.generate_content(
                    response_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.4 if response_type == "expert_knowledge" else 0.2,  # Higher creativity for expert knowledge
                        top_p=0.9,
                        max_output_tokens=2000,  # Increased token limit for more comprehensive responses
                        candidate_count=1
                    )
                )
                
                if response and response.text:
                    answer = response.text.strip()
                    logger.info(f"Generated response of length: {len(answer)}")
                    
                    # Validate response quality
                    if self._is_response_helpful(answer, query):
                        logger.info("Response validated as helpful")
                        break
                    elif attempt < max_retries - 1:
                        logger.warning(f"Response not helpful enough, retrying... (attempt {attempt + 1})")
                        continue
                    else:
                        logger.warning("All attempts exhausted, using current response")
                else:
                    logger.error("Empty response received from model")
                    if attempt == max_retries - 1:
                        answer = self._generate_fallback_response(query, classification)
                    continue
                    
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Error generating response (attempt {attempt + 1}): {e}")
                
                if "quota" in error_msg or "rate limit" in error_msg or "resource exhausted" in error_msg:
                    logger.warning(f"Rate limit hit during response generation, rotating API key...")
                    self._rotate_api_key()
                    time.sleep(1)  # Brief pause after key rotation
                    continue
                elif "safety" in error_msg or "blocked" in error_msg:
                    logger.warning("Response blocked by safety filters, trying with modified prompt")
                    # For safety blocks, try with a more neutral context
                    context = f"Please provide helpful technical guidance for this {classification.topic} question in a professional manner."
                    continue
                else:
                    if attempt == max_retries - 1:
                        logger.error("All attempts failed, using fallback response")
                        answer = self._generate_fallback_response(query, classification)
                    continue
        
        # If we still don't have an answer, use fallback
        if not answer or len(answer.strip()) < 50:
            logger.warning("No valid response generated, using fallback")
            answer = self._generate_fallback_response(query, classification)
        
        # Calculate confidence based on context and response quality
        base_confidence = 0.4  # Increased minimum confidence
        if chunks:
            avg_similarity = sum(chunk.similarity_score for chunk in chunks) / len(chunks)
            context_confidence = min(avg_similarity * 1.2, 0.6)  # Cap at 0.6
            logger.info(f"Average chunk similarity: {avg_similarity:.3f}")
        else:
            context_confidence = 0.3 if response_type == "expert_knowledge" else 0.1  # Higher confidence in expert knowledge
        
        # Response quality bonus
        response_quality_bonus = 0.1 if self._is_response_helpful(answer, query) else 0
        
        final_confidence = min(base_confidence + context_confidence + confidence_boost + response_quality_bonus, 1.0)
        
        logger.info(f"Final response confidence: {final_confidence:.3f}")
        
        return RAGResponse(
            query=query,
            answer=answer,
            sources=chunks[:5] if chunks else [],  # Return top 5 sources for display
            confidence_score=final_confidence,
            response_metadata={
                "chunks_retrieved": len(chunks),
                "response_type": response_type,
                "avg_similarity": sum(chunk.similarity_score for chunk in chunks) / len(chunks) if chunks else 0,
                "chromadb_available": self.chromadb_available,
                "attempts_made": max_retries,
                "final_answer_length": len(answer)
            }
        )

# Streamlit App Configuration
st.set_page_config(
    page_title="Atlan AI Customer Support Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #0052CC;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .ticket-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .topic-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        color: white;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    .sentiment-label {
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }
    
    .priority-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        color: white;
        font-size: 0.75rem;
        font-weight: bold;
    }
    
    .p0-badge { background-color: #FF5630; }
    .p1-badge { background-color: #FF8B00; }
    .p2-badge { background-color: #FFC400; color: black; }
    
    .backend-view {
        background-color: #F5F5F5;
        border-left: 4px solid #0052CC;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .frontend-view {
        background-color: #FFFFFF;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .source-list {
        background-color: #F8F9FA;
        padding: 0.5rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def get_topic_color(topic: str) -> str:
    """Get color for topic badge"""
    colors = {
        "How-to": "#36B37E",
        "Product": "#0052CC", 
        "Connector": "#8777D9",
        "Lineage": "#FF8B00",
        "API/SDK": "#403294",
        "SSO": "#FF5630",
        "Glossary": "#00875A",
        "Best Practices": "#0747A6",
        "Sensitive Data": "#DE350B"
    }
    return colors.get(topic, "#6B778C")

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    emojis = {
        "Frustrated": "😤",
        "Curious": "🤔", 
        "Angry": "😠",
        "Neutral": "😐"
    }
    return emojis.get(sentiment, "😐")

def load_sample_tickets():
    """Load sample tickets for demonstration"""
    sample_tickets = SAMPLE
    # Transform the data structure to match expected format
    transformed_tickets = []
    for ticket in sample_tickets:
        transformed_tickets.append({
            "id": ticket["id"],
            "text": ticket["body"],  # Map 'body' to 'text'
            "subject": ticket["subject"]
        })
    return transformed_tickets

def initialize_rag_system():
    """Initialize the RAG system"""
    if "rag_system" not in st.session_state:
        # Configuration - Add your actual API keys here
        GEMINI_API_KEYS = [
            "AIzaSyAFHriOAJQFwaVcSgAXpdyUW_DvIPdWQd4",
            "AIzaSyA2eGfn-HYFgVVU3146LQMqD_QVIf_7snY", 
            "AIzaSyAwjBzdYJVQUehCCLigvjNKOEb3Szo6HkY",
            "AIzaSyCWvK_GYiy2ZpITZxpWb7453zFzoN_VqmM"
        ]
        
        # Only use ChromaDB config if available
        CHROMADB_CONFIG = None
        if CHROMADB_AVAILABLE:
            CHROMADB_CONFIG = {
                'api_key': 'ck-GgDLCLEeXKAEhpWCWMDwcFP1hEVH4gpqhii25vw98XSC',
                'tenant': '94df3293-175e-443f-994a-22655697ffc9',
                'database': 'atlan'
            }
        
        try:
            if not GEMINI_API_KEYS or GEMINI_API_KEYS == ['']:
                st.error("⚠️ Please add your Gemini API keys to initialize the system")
                st.stop()
                
            st.session_state.rag_system = AtlanRAGSystem(
                gemini_api_keys=GEMINI_API_KEYS,
                chromadb_config=CHROMADB_CONFIG
            )
            
            if CHROMADB_AVAILABLE:
                st.success("✅ RAG System initialized successfully!")
            else:
                st.warning("⚠️ RAG System initialized in classification-only mode (ChromaDB unavailable)")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            st.stop()

def bulk_ticket_dashboard():
    """Render the bulk ticket classification dashboard"""
    st.markdown('<h1 class="main-header">Bulk Ticket Classification</h1>', unsafe_allow_html=True)
    
    # Load and classify tickets if not already done
    if "classified_tickets" not in st.session_state:
        # Create two columns for side-by-side display
        col_progress, col_results = st.columns([1, 1])
        
        with col_progress:
            st.markdown("### Processing Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            current_ticket = st.empty()
            
        with col_results:
            st.markdown("### Live Results")
            results_container = st.empty()
        
        # Load tickets
        tickets = load_sample_tickets()
        classified_tickets = []
        
        for i, ticket in enumerate(tickets):
            # Update progress
            progress = (i + 1) / len(tickets)
            progress_bar.progress(progress)
            status_text.text(f"Processing ticket {i + 1} of {len(tickets)}")
            current_ticket.text(f"Current: Ticket #{ticket['id']}")
            
            # Classify ticket
            classification = st.session_state.rag_system.classify_ticket(ticket["text"])
            classified_ticket = {
                **ticket,
                "classification": classification
            }
            classified_tickets.append(classified_ticket)
            
            # Update results display in real-time
            with results_container.container():
                st.write(f"**Completed: {len(classified_tickets)} tickets**")
                
                # Show last few processed tickets
                recent_tickets = classified_tickets[-3:]  # Show last 3
                for ct in recent_tickets:
                    with st.expander(f"Ticket #{ct['id']} - {ct['classification'].topic}", expanded=False):
                        topic_color = get_topic_color(ct["classification"].topic)
                        sentiment_emoji = get_sentiment_emoji(ct["classification"].sentiment)
                        
                        st.markdown(f'<span class="topic-badge" style="background-color: {topic_color};">{ct["classification"].topic}</span> '
                                  f'<span class="sentiment-label">{sentiment_emoji} {ct["classification"].sentiment}</span> '
                                  f'<span class="priority-badge {ct["classification"].priority.lower()}-badge">{ct["classification"].priority}</span>', 
                                  unsafe_allow_html=True)
                        
                        st.write(ct["text"][:200] + "..." if len(ct["text"]) > 200 else ct["text"])
        
        # Final update
        status_text.text("✅ All tickets processed!")
        progress_bar.progress(1.0)
        
        st.session_state.classified_tickets = classified_tickets
        st.success(f"Successfully processed {len(classified_tickets)} tickets!")
        
        # Clear the progress display after completion
        time.sleep(1)
        col_progress.empty()

def interactive_ai_agent():
    """Render the interactive AI agent interface"""
    st.markdown('<h1 class="main-header">Interactive AI Agent</h1>', unsafe_allow_html=True)
    
    # Ticket input
    st.write("**Enter your ticket or question:**")
    ticket_text = st.text_area(
        "Ticket Content",
        placeholder="Enter ticket or question here...",
        height=100,
        label_visibility="collapsed"
    )
    
    if st.button("Submit Ticket", type="primary"):
        if not ticket_text.strip():
            st.warning("Please enter a ticket or question.")
            return
        
        with st.spinner("Analyzing ticket..."):
            # Classify the ticket
            classification = st.session_state.rag_system.classify_ticket(ticket_text)
            
            # Generate RAG response
            rag_response = st.session_state.rag_system.generate_rag_response(ticket_text, classification)
        
        # Backend Analysis View
        st.markdown("## Internal Analysis")
        st.markdown('<div class="backend-view">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Topic", classification.topic)
            st.metric("Topic Confidence", f"{classification.confidence_scores.get('topic', 0)}%")
        
        with col2:
            st.metric("Sentiment", f"{get_sentiment_emoji(classification.sentiment)} {classification.sentiment}")
            st.metric("Sentiment Confidence", f"{classification.confidence_scores.get('sentiment', 0)}%")
        
        with col3:
            st.metric("Priority", classification.priority)
            st.metric("Priority Confidence", f"{classification.confidence_scores.get('priority', 0)}%")
        
        if rag_response.sources:
            st.write("**RAG Sources Considered:**")
            for i, source in enumerate(rag_response.sources[:3], 1):
                st.write(f"- Source {i}: {source.title} (Score: {source.similarity_score:.3f})")
                st.write(f"  URL: {source.url}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Frontend Response View
        st.markdown("## Customer Response")
        st.markdown('<div class="frontend-view">', unsafe_allow_html=True)
        
        # Display response
        st.write("**Answer:**")
        st.write(rag_response.answer)
        
        # Display sources if available
        if rag_response.sources:
            st.write("**Sources:**")
            st.markdown('<div class="source-list">', unsafe_allow_html=True)
            
            for source in rag_response.sources:
                if source.url:
                    st.write(f"- [{source.title}]({source.url})")
                else:
                    st.write(f"- {source.title}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Copy button for response
            copy_text = f"Answer: {rag_response.answer}\n\nSources:\n" + \
                       "\n".join([f"- {s.url}" for s in rag_response.sources if s.url])
            
            if st.button("📋 Copy Response"):
                st.success("Response copied to clipboard!")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Sidebar
    with st.sidebar:
        # Logo placeholder
        st.markdown("## 🏢 Atlan AI Support")
        st.markdown("---")
        
        # Navigation - Changed default to Interactive AI Agent
        page = st.radio(
            "Navigation",
            ["Interactive AI Agent", "Bulk Ticket Dashboard"],  # Swapped order
            key="navigation"
        )
        
        st.markdown("---")
        
        # Optional controls
        st.markdown("### Settings")
        
        # Theme switcher placeholder
        theme = st.selectbox("Theme", ["Light", "Dark"], disabled=True)
        st.caption("Theme switcher coming soon")
        
        # System status
        st.markdown("### System Status")
        if "rag_system" in st.session_state:
            if CHROMADB_AVAILABLE:
                st.success("🟢 RAG System Online")
                # Show collection count
                try:
                    count = st.session_state.rag_system.collection.count()
                    st.write(f"📚 Documents: {count}")
                except:
                    st.write("📚 Documents: Unknown")
            else:
                st.warning("🟡 Classification Only Mode")
                st.write("📚 Documents: ChromaDB unavailable")
        else:
            st.error("🔴 RAG System Offline")
    
    # Initialize RAG system
    initialize_rag_system()
    
    # Main content area - Updated condition order
    if page == "Interactive AI Agent":
        interactive_ai_agent()
    elif page == "Bulk Ticket Dashboard":
        bulk_ticket_dashboard()

if __name__ == "__main__":
    main()
