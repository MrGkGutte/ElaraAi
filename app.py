from flask import Flask, request, render_template
import os
import json
from groq import Groq
from tavily import TavilyClient
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- API Keys ---
groq_api_key = os.environ.get("GROQ_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

client = Groq(api_key=groq_api_key)
tavily = TavilyClient(api_key=tavily_api_key)

# --- MODELS PRIORITY LIST ---
# 1. llama-3.3-70b-versatile (Newest, but sometimes errors)
# 2. llama-3.1-8b-instant    (Fastest, very stable)
# 3. mixtral-8x7b-32768      (Old reliable backup)
MODELS_TO_TRY = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except:
        return "Smart Mobile API is Running!"

# --- Helper: Web Search ---
def get_web_search(query):
    print(f"LOG: Searching for: {query}")
    try:
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        context = "\n".join([f"- {obj['content']}" for obj in response['results']])
        return context
    except Exception as e:
        return f"Search Error: {str(e)}"

@app.route('/api/ask', methods=['GET'])
def ask_ai():
    user_question = request.args.get('text')
    if not user_question:
        return "Error: No question provided."

    # 1. Simplified Tool Definition (Optimized to prevent Error 400)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Get live information about news, weather, or facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The topic to search",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # 2. System Prompt
    messages = [
        {
            "role": "system", 
            "content": (
                "You are Elara, created by Gk Gutte. "
                "If the user asks for live info (news, weather, crypto), use the 'web_search' tool. "
                "Otherwise answer directly."
            )
        },
        {
            "role": "user", 
            "content": user_question
        }
    ]

    # 3. ROBUST LOOP (Catches Error 400 and Switches Models)
    last_error = ""
    
    for model in MODELS_TO_TRY:
        try:
            print(f"LOG: Attempting with model: {model}")
            
            # Call Groq
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                tools=tools,
                tool_choice="auto",
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Handle Search
            if tool_calls:
                print(f"LOG: {model} requested search.")
                available_functions = {"web_search": get_web_search}
                messages.append(response_message) 

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    search_results = function_to_call(query=function_args.get("query"))
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": search_results,
                    })

                # Final Answer (Must use same model)
                final_response = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                return final_response.choices[0].message.content

            else:
                # No search needed
                return response_message.content

        except Exception as e:
            # THIS IS THE KEY FIX:
            # Instead of crashing, we print the error and TRY THE NEXT MODEL
            print(f"LOG: Model {model} failed with error: {e}")
            last_error = str(e)
            continue 

    # If ALL 3 models fail
    return f"System currently busy. Please try again. (Debug: {last_error})"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
        
