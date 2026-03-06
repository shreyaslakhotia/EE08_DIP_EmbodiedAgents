import ollama 

messages = [
    {
        'role': 'user',
        'content': 'Explain what Python is in one sentence.',
    },
]

response = ollama.chat(model='tinyllama:1.1b', messages=messages)
print(response['message']['content'])

#run in venv for this and download ollama
