import streamlit as st
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Represents a single message in the conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    classification: Optional[object] = None
    sources: List[object] = None
    confidence: float = 0.0

class ConversationManager:
    """Manages conversation state and context"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        if "conversation_context" not in st.session_state:
            st.session_state.conversation_context = {}
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to the conversation history"""
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=time.strftime("%H:%M:%S"),
            **kwargs
        )
        st.session_state.conversation_history.append(message)
        return message
    
    def get_conversation_context(self, max_messages: int = 5) -> str:
        """Get recent conversation context for better responses"""
        if not st.session_state.conversation_history:
            return ""
        
        recent_messages = st.session_state.conversation_history[-max_messages:]
        context_parts = []
        
        for msg in recent_messages:
            if msg.role == "user":
                context_parts.append(f"User: {msg.content}")
            else:
                context_parts.append(f"Assistant: {msg.content[:200]}...")
        
        return "\n".join(context_parts)
    
    def clear_conversation(self):
        """Clear the conversation history"""
        st.session_state.conversation_history = []
        st.session_state.conversation_context = {}

def render_message(message: ConversationMessage):
    """Render a single conversation message"""
    if message.role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
            st.caption(f"⏰ {message.timestamp}")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
            
            # Show classification info in expandable section
            if message.classification:
                with st.expander("📊 Analysis Details", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Topic", message.classification.topic)
                    with col2:
                        st.metric("Sentiment", f"{get_sentiment_emoji(message.classification.sentiment)} {message.classification.sentiment}")
                    with col3:
                        st.metric("Priority", message.classification.priority)
                    
                    if message.confidence:
                        st.metric("Response Confidence", f"{message.confidence:.1%}")
            
            # Show sources if available
            if message.sources:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message.sources, 1):
                        if source.url:
                            st.markdown(f"{i}. [{source.title}]({source.url}) (Score: {source.similarity_score:.3f})")
                        else:
                            st.markdown(f"{i}. {source.title} (Score: {source.similarity_score:.3f})")
            
            st.caption(f"⏰ {message.timestamp}")

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    emojis = {
        "Frustrated": "😤",
        "Curious": "🤔", 
        "Angry": "😠",
        "Neutral": "😐"
    }
    return emojis.get(sentiment, "😐")

def get_enhanced_response_prompt(query: str, context: str, classification: object, conversation_context: str = "") -> str:
    """Get enhanced prompt that includes conversation context"""
    base_prompt = f"""You are an expert Atlan customer support specialist with deep knowledge of the platform. This is part of an ongoing conversation with a customer.

CONVERSATION CONTEXT:
{conversation_context}

CUSTOMER CONTEXT:
- Topic: {classification.topic}
- Sentiment: {classification.sentiment}  
- Priority: {classification.priority}

DOCUMENTATION CONTEXT:
{context}

CURRENT QUESTION: {query}

RESPONSE GUIDELINES:

**Conversational Principles:**
- Reference previous parts of the conversation when relevant
- Build upon previously provided information rather than repeating it
- Acknowledge if this relates to earlier questions in the conversation
- Maintain conversational flow and context awareness

**Core Principles:**
- ALWAYS attempt to provide actionable guidance, even with limited context
- Use your knowledge of data platforms and common patterns to infer solutions
- Provide helpful alternatives when exact answers aren't available
- Only escalate to human support as a LAST resort after exhausting all options

**Response Strategy:**
1. **Acknowledge Context**: Reference relevant parts of the ongoing conversation
2. **Primary Answer**: Use documentation context to address the core question
3. **Informed Inference**: If context is partial, use platform knowledge to provide educated guidance
4. **Alternative Solutions**: Suggest related approaches or workarounds
5. **Step-by-Step Breakdown**: Break complex questions into manageable steps
6. **Proactive Guidance**: Anticipate follow-up questions and provide comprehensive coverage

**Quality Standards:**
- Be specific with actionable steps
- Include fallback options and alternatives
- Provide context for why certain approaches are recommended
- End with clear next steps, even if they involve experimentation
- Maintain conversational continuity

**Tone Matching:**
- High priority/frustrated: Be direct, acknowledge urgency, provide immediate actionable steps
- Curious/learning: Be educational, provide context and background
- Technical questions: Be precise, include examples and edge cases

IMPORTANT: This is an ongoing conversation. Build upon the context and avoid repeating information you've already provided. Focus on moving the conversation forward constructively.

Provide a comprehensive, conversational response:"""
    
    return base_prompt

def conversational_ai_mode():
    """Render the conversational AI interface"""
    
    # Initialize conversation manager
    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager(st.session_state.rag_system)
    
    conv_mgr = st.session_state.conversation_manager
    
    # Header
    st.markdown('<h1 class="main-header">💬 Conversational AI Agent</h1>', unsafe_allow_html=True)
    st.markdown("Have a natural conversation with the AI about your Atlan questions and issues.")
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            conv_mgr.clear_conversation()
            st.rerun()
    
    with col2:
        message_count = len(st.session_state.conversation_history)
        st.metric("Messages", message_count)
    
    # Chat container
    chat_container = st.container()
    
    # Display conversation history
    with chat_container:
        if st.session_state.conversation_history:
            for message in st.session_state.conversation_history:
                render_message(message)
        else:
            # Welcome message
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
                👋 **Welcome to the Atlan AI Assistant!**
                
                I'm here to help you with all your Atlan questions. You can ask me about:
                - How to set up connectors and data sources
                - Product features and capabilities  
                - Data lineage and governance
                - API and SDK usage
                - Best practices and troubleshooting
                
                Just start typing your question below and we'll have a conversation!
                """)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about Atlan..."):
        # Add user message
        conv_mgr.add_message("user", prompt)
        
        # Show user message immediately
        with chat_container:
            render_message(st.session_state.conversation_history[-1])
        
        # Generate response
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    # Get conversation context
                    conversation_context = conv_mgr.get_conversation_context()
                    
                    # Classify the current message
                    classification = st.session_state.rag_system.classify_ticket(prompt)
                    
                    # Generate RAG response with conversation context
                    rag_response = st.session_state.rag_system.generate_rag_response(
                        prompt, classification
                    )
                    
                    # If we have conversation context, enhance the prompt
                    if conversation_context:
                        try:
                            # Get enhanced response that considers conversation history
                            enhanced_prompt = get_enhanced_response_prompt(
                                query=prompt,
                                context="\n".join([f"Source {i+1}: {chunk.content}" for i, chunk in enumerate(rag_response.sources[:5])]),
                                classification=classification,
                                conversation_context=conversation_context
                            )
                            
                            # Generate enhanced response
                            import google.generativeai as genai
                            model = genai.GenerativeModel(st.session_state.rag_system.generation_model)
                            enhanced_response = model.generate_content(
                                enhanced_prompt,
                                generation_config=genai.GenerationConfig(
                                    temperature=0.3,
                                    top_p=0.9,
                                    max_output_tokens=2000,
                                )
                            )
                            
                            if enhanced_response and enhanced_response.text:
                                rag_response.answer = enhanced_response.text.strip()
                            
                        except Exception as e:
                            logger.warning(f"Failed to generate enhanced response, using standard RAG: {e}")
                
                # Display the response
                st.markdown(rag_response.answer)
                
                # Add assistant message to history
                conv_mgr.add_message(
                    "assistant", 
                    rag_response.answer,
                    classification=classification,
                    sources=rag_response.sources,
                    confidence=rag_response.confidence_score
                )
                
                # Show analysis details in expandable section
                if classification:
                    with st.expander("📊 Analysis Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Topic", classification.topic)
                        with col2:
                            st.metric("Sentiment", f"{get_sentiment_emoji(classification.sentiment)} {classification.sentiment}")
                        with col3:
                            st.metric("Priority", classification.priority)
                        
                        st.metric("Response Confidence", f"{rag_response.confidence_score:.1%}")
                
                # Show sources if available
                if rag_response.sources:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(rag_response.sources, 1):
                            if source.url:
                                st.markdown(f"{i}. [{source.title}]({source.url}) (Score: {source.similarity_score:.3f})")
                            else:
                                st.markdown(f"{i}. {source.title} (Score: {source.similarity_score:.3f})")
                
                st.caption(f"⏰ {time.strftime('%H:%M:%S')}")
        
        # Rerun to update the display
        st.rerun()
    
    # Conversation statistics in sidebar
    if st.session_state.conversation_history:
        with st.sidebar:
            st.markdown("### 💬 Conversation Stats")
            user_messages = len([m for m in st.session_state.conversation_history if m.role == "user"])
            assistant_messages = len([m for m in st.session_state.conversation_history if m.role == "assistant"])
            
            st.metric("User Messages", user_messages)
            st.metric("Assistant Responses", assistant_messages)
            
            # Recent topics
            if assistant_messages > 0:
                recent_topics = []
                for msg in reversed(st.session_state.conversation_history):
                    if msg.role == "assistant" and msg.classification:
                        recent_topics.append(msg.classification.topic)
                        if len(recent_topics) >= 3:
                            break
                
                if recent_topics:
                    st.markdown("**Recent Topics:**")
                    for topic in recent_topics:
                        st.write(f"• {topic}")