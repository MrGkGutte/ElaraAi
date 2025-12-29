from flask import Flask, request
import os
from groq import Groq

app = Flask(__name__)

# API Key Setup (Render se key lega)
groq_api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

@app.route('/')
def home():
    return "Mobile API is Running!"

# Yeh wo rasta hai jahan App Inventor connect karega
@app.route('/api/ask', methods=['GET'])
def ask_ai():
    # App se aaya hua sawal yahan receive hoga
    user_question = request.args.get('text')
    
    if not user_question:
        return "Error: No question provided."

    try:
        # Groq AI se sawal puchna
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": "You are Elara AI. Answer briefly and clearly in plain text."
                },
                {
                    "role": "user", 
                    "content": user_question,
                }
            ],
            model="llama-3.1-8b-instant",
        )
        
        # Sirf jawab return karna (Plain Text mein)
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
