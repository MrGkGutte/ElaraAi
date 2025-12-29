from flask import Flask, request, jsonify
import os
import json
from groq import Groq
from tavily import TavilyClient

app = Flask(__name__)

# --- API Keys Setup ---
groq_api_key = os.environ.get("GROQ_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

# Initialize Clients
client = Groq(api_key=groq_api_key)
tavily = TavilyClient(api_key=tavily_api_key)

@app.route('/')
def home():
    return "Smart Mobile API is Running!"

# --- Helper Function: Perform Web Search ---
def get_web_search(query):
    """
    Searches the web using Tavily and returns a summary.
    """
    try:
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        # We join the results into a single string for the AI to read
        context = "\n".join([f"- {obj['content']}" for obj in response['results']])
        return context
    except Exception as e:
        return f"Error performing search: {str(e)}"

@app.route('/api/ask', methods=['GET'])
def ask_ai():
    user_question = request.args.get('text')
    
    if not user_question:
        return "Error: No question provided."

    # 1. Define the Tool (Start telling Groq it HAS the ability to search)
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
        # 2. First Call: Ask Groq (Let it decide if it needs the tool)
        messages = [
            {"role": "system", "content": "You are Elara. If the user asks for current/live info, use the web_search tool. Otherwise, answer directly."},
            {"role": "user", "content": user_question}
        ]

        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            tools=tools,
            tool_choice="auto", # This lets the AI decide automatically
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # 3. Check decision: Did Groq ask for a search?
        if tool_calls:
            # YES, it wants to search
            print("LOG: Search needed. Searching...")
            
            # Extract the arguments (what Groq wants to search for)
            available_functions = {"web_search": get_web_search}
            
            # Add Groq's request to the conversation history
            messages.append(response_message) 

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                # Run the Tavily Search
                search_results = function_to_call(query=function_args.get("query"))
                
                # Add the search results to the conversation
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": search_results,
                    }
                )

            # 4. Final Call: Ask Groq again, now with the search results
            final_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages
            )
            return final_response.choices[0].message.content

        else:
            # NO, it answered directly (Memory only)
            print("LOG: No search needed.")
            return response_message.content

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
    
