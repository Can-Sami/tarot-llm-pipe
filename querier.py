from main import query_rag

def main():
    # Initialize chat history
    chat_history = ""
    
    print("Welcome! Ask me questions about tarot readings. Type 'quit' to exit.")
    
    while True:
        # Ask a question
        question = input("\nYour question: ").strip()
        
        # Check for exit condition
        if question.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
            
        # Get the answer
        response = query_rag(question, chat_history)
        print("\nFortune Teller:", response)
        
        # Update chat history
        chat_history += f"\nHuman: {question}\nFortune Teller: {response}"

if __name__ == "__main__":
    main()
