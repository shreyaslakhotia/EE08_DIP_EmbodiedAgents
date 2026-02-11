# ONLY HAS CHATBOT FUNCTION

import ollama

def start_agent():
    # 1. Initialize Memory with a 'System' instruction
    history = [
        {'role': 'system', 'content': 'You are a helpful EEE study assistant. Give concise, engineering-focused answers.'}
    ]
    
    print("--- Qwen 3 Agent Initialized (Type 'exit' to quit) ---")
    
    while True:
        # 2. Get User Input
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        # 3. Save User Message to Memory
        history.append({'role': 'user', 'content': user_input})
        
        # 4. Stream Response from Qwen 3
        print("Agent: ", end="", flush=True)
        full_response = ""
        
        # We use 'stream=True' so the text appears word-by-word
        for chunk in ollama.chat(model='qwen3', messages=history, stream=True):
            content = chunk['message']['content']
            print(content, end='', flush=True)
            full_response += content
            
        # 5. Save Assistant Answer to Memory for context in the next turn
        history.append({'role': 'assistant', 'content': full_response})
        print()

if __name__ == "__main__":
    start_agent()