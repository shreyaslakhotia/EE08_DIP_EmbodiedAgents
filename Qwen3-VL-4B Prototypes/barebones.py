import time
from ollama import Client

# Bypass IPv6 DNS timeout by explicitly pointing to the local IPv4 socket
client = Client(host='http://127.0.0.1:11434')

print("="*50)
print("🧠 RAW OLLAMA API BENCHMARK (Text Only)")
print("="*50)

while True:
    prompt = input("\nEnter prompt (or 'quit' to exit): ").strip()
    if prompt.lower() == 'quit':
        break
    if not prompt:
        continue

    print("\n[Thinking...]")
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    try:
        # Synchronous streaming with optimized hardware options
        stream = client.chat(
            model='qwen3-vl:4b', 
            messages=[{'role': 'user', 'content': prompt}], 
            stream=True,
            keep_alive=-1,
            options={
                "num_thread": 4,  # Force all 4 Pi 5 cores
                "num_ctx": 1024   # Clamp the context window
            }
        )
        
        print("Response: ", end="", flush=True)
        
        for chunk in stream:
            # Capture the exact moment the first word arrives
            if first_token_time is None:
                first_token_time = time.time()
                time_to_first = first_token_time - start_time
                
            token = chunk['message']['content']
            print(token, end="", flush=True)
            token_count += 1
            
        end_time = time.time()
        
        # --- METRICS CALCULATION ---
        total_time = end_time - start_time
        gen_time = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_time if gen_time > 0 else 0
        
        print(f"\n\n[📊 METRICS]")
        print(f"- Time to First Word: {time_to_first:.2f} seconds")
        print(f"- Generation Speed:   {tps:.2f} tokens/second")
        print(f"- Total Tokens:       {token_count}")
        
    except Exception as e:
        print(f"\n[❌ API ERROR]: {e}")