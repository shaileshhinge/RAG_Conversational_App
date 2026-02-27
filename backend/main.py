"""
Task 3: Main Application

Create a simple command-line interface to demonstrate the multi-agent system:
1. Initialize document loader and agents
2. Process the document
3. Interactive query loop for testing

Requirements:
- Load and process doc.txt automatically
- Allow users to input queries
- Show agent responses with proper formatting
- Handle errors gracefully
"""

from document_loader import DocumentLoader
from agents import SearchAgent, ResponseAgent, AgentOrchestrator

class DocumentChatSystem:
    def __init__(self):
        """
        Initialize the complete system
        
        Task: Complete initialization:
        1. Create DocumentLoader instance
        2. Call load_and_process() to prepare the document
        3. If successful, initialize SearchAgent and ResponseAgent
        4. Create AgentOrchestrator with both agents
        5. Set self.ready based on initialization success
        """
        document_loader = DocumentLoader()
        success = document_loader.load_and_process()

        if success:
            search_agent = SearchAgent(document_loader)
            response_agent = ResponseAgent()
            self.orchestrator = AgentOrchestrator(search_agent=search_agent, response_agent=response_agent)
            self.ready = True
        else:
            self.orchestrator = None
            self.ready = False
    
    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Process a single query through the system
        
        Task: Complete query processing:
        1. Check if system is ready
        2. If not ready, return error message
        3. If ready, use orchestrator to process query
        4. Return the result
        
        Args:
            user_input: User's question
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.ready:
            return {
                "response": "System is not ready. Failed to load the document.",
                "sources": [],
                "follow_ups": [],
                "intent": "general"
            }
        return self.orchestrator.process_query(user_input)
    
    def interactive_mode(self):
        """
        Run interactive command-line interface
        
        Task: Complete interactive mode:
        1. Print welcome message and instructions
        2. Loop continuously asking for user input
        3. Handle 'quit' or 'exit' commands
        4. Process queries and display results nicely
        5. Handle keyboard interrupts gracefully
        """
        print("=" * 50)
        print("Document Chat System - Multi Agent RAG")
        print("=" * 50)
        print("Type your query, Type 'quit' or 'exit' to stop. \n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['quit', 'exit']:
                    print("Bye.. have a good day!")
                    break
                
                result = self.query(user_input=user_input)
                print("\nAgent: {result['response']}")
                if result.get('sources'):
                    print("Sources: {','.join(result['sources'])}")
                if result.get("follow_ups"):
                    print("Follow-ups.")
                    for q in result["follow_ups"]:
                        print(f" -{q}")
                print()
            


    def demo_queries(self):
        """
        Run demonstration with sample queries
        
        Task: Complete demo mode:
        1. Define 4 sample queries covering different intents
        2. Process each query and display results
        3. Show the multi-agent workflow in action
        """
        queries = [
            "what are the main risks mentioned?",
            "what re the financial costs?",
            "Give me a general summary."
        ]

def main():
    """
    Main entry point
    
    Task: Complete main function:
    1. Create DocumentChatSystem instance
    2. Check if system initialized successfully
    3. If successful, run interactive_mode()
    4. If failed, show error message
    """
    system = DocumentChatSystem()
    if sustem.ready:
        system.interactive_mode()
    else:
        print("Error. Failed to initialize. Please chcek if the doc exists").

if __name__ == "__main__":
    main()