from flask import Flask, request, jsonify, render_template
import os
import json
from groq import Groq
from tavily import TavilyClient
from flask_cors import CORS  # <--- NEW IMPORT

app = Flask(__name__)
CORS(app)  # <--- THIS FIXES THE CONNECTION ERROR

# --- API Keys Setup ---
groq_api_key = os.environ.get("GROQ_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

client = Groq(api_key=groq_api_key)
tavily = TavilyClient(api_key=tavily_api_key)

# --- ROUTE 1: The Website (HTML) ---
# If you uploaded index.html to a 'templates' folder, this works.
# If not, it just returns a text message.
@app.route('/')
def home():
    try:
        return render_template('index.html')
    except:
        return "Smart Mobile API (Created by Gk Gutte) is Running! (HTML not found)"

# --- Helper Function: Web Search ---
def get_web_search(query):
    try:
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        context = "\n".join([f"- {obj['content']}" for obj in response['results']])
        return context
    except Exception as e:
        return f"Error performing search: {str(e)}"

# --- ROUTE 2: The API (For App Inventor & Website) ---
@app.route('/api/ask', methods=['GET'])
def ask_ai():
    user_question = request.args.get('text')
    
    if not user_question:
        return "Error: No question provided."

    # Define the Tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Get current news, weather, live scores, or real-time information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find information on the internet",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    try:
        # 1. First Call: Ask Groq
        messages = [
            {
                "role": "system", 
                "content": "You are Elara, an AI assistant created by Gk Gutte. If the user asks for current/live info, use the web_search tool. Otherwise, answer directly."
            },
            {
                "role": "user", 
                "content": user_question
            }
        ]

        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # 2. Check decision: Search needed?
        if tool_calls:
            print("LOG: Search needed. Searching...")
            available_functions = {"web_search": get_web_search}
            messages.append(response_message) 

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                search_results = function_to_call(query=function_args.get("query"))
                
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": search_results,
                    }
                )

            final_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages
            )
            return final_response.choices[0].message.content

        else:
            return response_message.content

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
        
