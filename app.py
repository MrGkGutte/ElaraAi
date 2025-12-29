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

    # 1. Simplified Tool Definition
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Find current information about news, weather, or real-time facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The topic to search for",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    try:
        # 2. System Prompt
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are Elara, created by Gk Gutte. "
                    "If the user asks about current events (news, weather, crypto prices), you MUST use the 'web_search' tool. "
                    "Do not answer from memory for live data."
                )
            },
            {
                "role": "user", 
                "content": user_question
            }
        ]

        # 3. Call Groq using the NEW SUPPORTED MODEL
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile", # <--- UPDATED MODEL
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # 4. Handle Search if requested
        if tool_calls:
            print("LOG: AI is searching...")
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

            # Final response after search
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", # <--- UPDATED MODEL
                messages=messages
            )
            return final_response.choices[0].message.content

        else:
            return response_message.content

    except Exception as e:
        print(f"API Error: {e}")
        return f"I encountered an error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
    
