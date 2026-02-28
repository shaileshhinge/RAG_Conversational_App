"""
Task 2: Multi-Agent System

Create two agents that work together:
1. SearchAgent - finds relevant information from documents
2. ResponseAgent - formats and improves responses

Requirements:
- SearchAgent uses the document loader to find relevant content
- ResponseAgent formats responses and handles different request types
- Simple coordination between agents
"""

from typing import Dict, List, Any

class SearchAgent:
    """Agent responsible for finding relevant information"""
    
    def __init__(self, document_loader):
        """
        Initialize the search agent
        
        Task: Complete initialization
        - Store reference to document_loader
        - Set agent name as "SearchAgent"
        """
        self.document_loader = document_loader
        self.name = "SearchAgent"
    
    def search_documents(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Search for relevant document content
        
        Task: Complete document search:
        1. Use document_loader.search_similar() to find relevant chunks
        2. If no results found, return appropriate response
        3. Return structured result with:
           - 'found': boolean indicating if results were found
           - 'content': list of found chunks with similarity scores
           - 'query': original query
           - 'intent': detected intent category
        
        Args:
            query: User's search query
            top_k: Number of results to return
            
        Returns:
            Dictionary with search results and metadata
        """
        results = self.document_loader.search_similar(query=query, top_k=top_k)
        if not results:
            return {
                "found": False,
                "content": [],
                "query": query,
                "intent": self.analyze_intent(query=query)
            }

        return {
            "found": True,
            "content": results,
            "query": query,
            "intent": self.analyze_intent(query=query)
        }

    def analyze_intent(self, query: str) -> str:
        """
        Analyze user query intent
        
        Task: Complete intent analysis using keyword matching:
        1. Convert query to lowercase
        2. Check for intent keywords:
           - "risk", "danger", "liability", "threat" -> "risk"  
           - "cost", "price", "payment", "financial" -> "financial"
           - "comply", "regulation", "requirement", "legal" -> "compliance"
           - everything else -> "general"
        3. Return intent category as string
        
        Args:
            query: User's query
            
        Returns:
            Intent category string
        """
        query_lower = query.lower()
        if any(word in query_lower for word in ["risk", "danger", "liability", "threat"]):
            return "risk"
        elif any(word in query_lower for word in ["cost", "price", "payment", "financial"]):
            return "financial"
        elif any(word in query_lower for word in ["comply", "regulation", "requirement", "legal"]):
            return "compliance"
        else:
            return "general"

class ResponseAgent:
    """Agent responsible for formatting responses"""
    
    def __init__(self):
        """
        Initialize the response agent
        
        Task: Complete initialization
        - Set agent name as "ResponseAgent"
        """
        self.name = "ResponseAgent"
    
    def format_response(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the response based on search results
        
        Task: Complete response formatting:
        1. Check if search found any results
        2. If no results, return "no information found" message
        3. If results found:
           - Get the most relevant chunk (first result)
           - Format based on intent:
             * "risk" -> "Risk Analysis: [content]"
             * "financial" -> "Financial Information: [content]" 
             * "compliance" -> "Compliance Information: [content]"
             * "general" -> "Document Information: [content]"
        4. Return dict with 'response' and 'sources' keys
        
        Args:
            search_results: Results from SearchAgent
            
        Returns:
            Dictionary with formatted response and sources
        """
        if not search_results.get("found"):
            return { 
                "response" : "I could not find any information for your query",
                "sources": []
            }
        
        content_list = search_results.get('content')
        intent = search_results.get("intent", "general")

        top_chunk = content_list[0] if content_list else {}
        text = top_chunk[0] if isinstance(top_chunk, tuple) else (top_chunk.get('text', '') if isinstance(top_chunk, dict) else str(top_chunk))

        intent_map = {
            "risk": f"Risk Analysis: {text}",
            "financial": f"Financial information: {text}",
            "compliance": f"Compliance information: {text}",
            "general": f"Document information: {text}"
        }
        response_text = intent_map.get(intent, f"Document information: {text}")
        sources = [
            chunk.get("source", "document") if isinstance(chunk, dict) else "document"
            for chunk in content_list
        ]
        return {
            "response": response_text,
            "sources": sources
        }

    def generate_follow_ups(self, intent: str) -> List[str]:
        """
        Generate follow-up questions based on intent
        
        Task: Complete follow-up generation:
        1. Based on intent category, return 2-3 relevant follow-up questions:
           - "risk": questions about mitigation, financial impact
           - "financial": questions about terms, costs, budget
           - "compliance": questions about requirements, penalties
           - "general": general questions about other aspects
        
        Args:
            intent: Detected intent category
            
        Returns:
            List of follow-up question strings
        """
        follow_ups = {
            "risk": [
                "what are the ways of mitigating this risk?",
                "what is the financial impact of this risk?",
                "How likely will this risk occur?"
            ],
            "financial": [
                "What are the terms of the payment?",
                "Are there any additional costs involved?"
            ],
            "compliance": [
                "What is the deadline for the compliance?",
                "Is there any penalty involved for not complying?"
            ],
            "general": [
                "Can you provide more details on this topic?",
                "Are there related documents available?"
            ]
        }
        return follow_ups.get(intent, follow_ups["general"])

    def make_concise(self, text: str) -> str:
        """
        Make response more concise
        
        Task: Complete response shortening:
        1. Split text into sentences
        2. Take first 2 sentences
        3. Add "..." if text was longer
        4. Return shortened version
        
        Args:
            text: Original text
            
        Returns:
            Shortened text
        """
        sentences = text.split(".")
        if len(sentences) <= 2:
            return text
        else:
            short = ".".join(sentences[:2])
            return short + "..."

class AgentOrchestrator:
    """Coordinates between the two agents"""
    
    def __init__(self, search_agent: SearchAgent, response_agent: ResponseAgent):
        """
        Initialize the orchestrator
        
        Task: Complete initialization
        - Store references to both agents
        - Initialize conversation_history as empty list
        """
        self.search_agent = search_agent
        self.response_agent = response_agent
        self.conversation_history = []
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query using both agents
        
        Task: Complete query processing:
        1. Check if this is a "make it concise" request:
           - If yes and conversation_history exists, make last response concise
        2. Otherwise, normal processing:
           - Use SearchAgent to find relevant information  
           - Use ResponseAgent to format the response
           - Generate follow-up questions
        3. Store result in conversation_history
        4. Return complete response with all fields
        
        Args:
            query: User's question
            
        Returns:
            Dict with keys: 'response', 'sources', 'follow_ups', 'intent'
        """
        # Using this if there is "make it concise" request
        if "make it concise" in query.lower() and self.conversation_history:
            last_response = self.conversation_history[-1]
            concise_text = self.response_agent.make_concise(last_response["response"])
            return {
                "response": concise_text,
                "sources": last_response.get("sources"),
                "follow_ups": last_response.get("follow_ups"),
                "intent": last_response.get("intent")
            }

        # Regular processing
        search_results = self.search_agent.search_documents(query=query)
        formatted = self.response_agent.format_response(search_results=search_results)
        follow_ups = self.response_agent.generate_follow_ups(search_results["intent"])

        result = {
            "response": formatted["response"],
            "sources": formatted["sources"],
            "intent": search_results["intent"],
            "follow_ups": follow_ups
        }

        self.conversation_history.append(result)
        return result