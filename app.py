from flask import Flask, request, jsonify, render_template
import os
import json
from groq import Groq
from tavily import TavilyClient
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- API Keys Setup ---
groq_api_key = os.environ.get("GROQ_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

# Initialize Clients
client = Groq(api_key=groq_api_key)
tavily = TavilyClient(api_key=tavily_api_key)

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except:
        return "Smart Mobile API is Running!"

# --- Helper Function: Web Search ---
def get_web_search(query):
    print(f"LOG: Running web search for: {query}")  # Log for debugging
    try:
        # Search specifically for accurate news/info
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        
        # Format the results cleanly
        context = "\n".join([f"- {obj['content']}" for obj in response['results']])
        return context
    except Exception as e:
        print(f"LOG: Search Error: {e}")
        return f"Error: {str(e)}"

@app.route('/api/ask', methods=['GET'])
def ask_ai():
    user_question = request.args.get('text')
    
    if not user_question:
        return "Error: No question provided."

    # 1. Define the Tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Find live information about news, weather, sports, stocks, or recent events.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search keywords (e.g. 'weather in Delhi', 'Bitcoin price')",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    try:
        # 2. Stronger System Prompt
        system_prompt = (
            "You are Elara, created by Gk Gutte. "
            "CRITICAL INSTRUCTION: If the user asks about ANY current event, weather, news, sports, "
            "or dynamic info (like stock prices), you MUST use the 'web_search' tool. "
            "Do not guess. Do not use your internal memory for facts that change."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]

        # 3. Call Groq (Using the SMARTER Model: llama-3.3-70b-versatile)
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile", # <--- CHANGED to 70B for better intelligence
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # 4. Check if AI wants to search
        if tool_calls:
            print("LOG: AI decided to search.")
            available_functions = {"web_search": get_web_search}
            messages.append(response_message) 

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                # Perform the search
                search_results = function_to_call(query=function_args.get("query"))
                
                # Give results back to AI
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": search_results,
                    }
                )

            # 5. Final Answer with search data
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", # <--- CHANGED to 70B
                messages=messages
            )
            return final_response.choices[0].message.content

        else:
            print("LOG: AI decided NOT to search.")
            return response_message.content

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
        
