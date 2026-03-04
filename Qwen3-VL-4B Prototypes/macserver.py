import time
from ollama import Client

# --- THE MAGIC ROUTING LINE ---
# Replace this IP with your actual MacBook IP address
MAC_IP = "10.91.51.112" 
remote_client = Client(host=f'http://{MAC_IP}:11434')

print("="*50)
print("📡 DISTRIBUTED AI BENCHMARK (Pi -> Mac)")
print("="*50)

while True:
    prompt = input("\nEnter prompt (or 'quit' to exit): ").strip()
    if prompt.lower() == 'quit':
        break
    if not prompt:
        continue

    print("\n[Transmitting to MacBook...]")
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    try:
        # Call the remote client instead of the local one
        stream = remote_client.chat(
            model='qwen3-vl:2b-instruct-q4_k_m', 
            messages=[{'role': 'user', 'content': prompt}], 
            stream=True
        )
        
        print("Response: ", end="", flush=True)
        
        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time()
                time_to_first = first_token_time - start_time
                
            token = chunk['message']['content']
            print(token, end="", flush=True)
            token_count += 1
            
        end_time = time.time()
        
        # --- METRICS ---
        gen_time = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_time if gen_time > 0 else 0
        
        print(f"\n\n[📊 METRICS]")
        print(f"- Network + Time to First Word: {time_to_first:.2f} seconds")
        print(f"- MacBook Generation Speed:     {tps:.2f} tokens/second")
        
    except Exception as e:
        print(f"\n[❌ NETWORK ERROR]: {e}")
        print("Make sure OLLAMA_HOST=0.0.0.0 is running on the Mac and IPs match.")