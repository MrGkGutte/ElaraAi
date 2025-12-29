import os
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from groq import Groq
from tavily import TavilyClient

app = Flask(__name__)
CORS(app) 

# Secure secret key
app.secret_key = os.urandom(24)

# --- HELPER: SMART SEARCH ROUTER ---
def check_if_search_needed(prompt, client):
    try:
        system_instruction = """
        You are a search router. Analyze the user's query.
        Return 'YES' if the query requires external real-time information.
        Return 'NO' if the query is conversational or generic.
        Reply ONLY with 'YES' or 'NO'.
        """
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5
        )
        return "YES" in response.choices[0].message.content.strip().upper()
    except:
        return False

@app.route('/')
def home():
    if 'messages' not in session:
        session['messages'] = []
    return render_template('index.html', messages=session['messages'])

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session['messages'] = []
    return jsonify({"status": "success"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    
    # 1. Try to get keys from Frontend, otherwise check Server Environment Variables
    groq_api_key = data.get('groq_api_key') or os.environ.get('GROQ_API_KEY')
    tavily_api_key = data.get('tavily_api_key') or os.environ.get('TAVILY_API_KEY')
    
    model_option = data.get('model_option', 'llama-3.3-70b-versatile')
    use_search = data.get('use_search', False)

    if not groq_api_key:
        return jsonify({"error": "Groq API Key is missing. Please set GROQ_API_KEY in Render Environment Variables."}), 400

    session['messages'].append({"role": "user", "content": user_input})
    session.modified = True

    try:
        client = Groq(api_key=groq_api_key)
        
        search_context = ""
        should_search = False
        search_results_display = None

        if use_search and tavily_api_key:
            should_search = check_if_search_needed(user_input, client)
            if should_search:
                try:
                    tavily = TavilyClient(api_key=tavily_api_key)
                    search_result = tavily.search(query=user_input, search_depth="basic", max_results=3)
                    snippets = [f"Source: {res['url']}\nContent: {res['content']}" for res in search_result['results']]
                    search_context = "\n\n".join(snippets)
                    search_results_display = search_result['results']
                except Exception as e:
                    print(f"Search failed: {e}")

        system_content = "You are Elara, a helpful and smart AI assistant created by Gk Gutte. Always introduce yourself as Elara when asked."
        if search_context:
            system_content += f"\n\nLIVE WEB CONTEXT:\n{search_context}\n\nINSTRUCTION: Answer using the context above."
        else:
            system_content += "\n\nINSTRUCTION: Answer based on your training data."

        messages_for_api = [{"role": "system", "content": system_content}]
        messages_for_api.extend(session['messages'])

        completion = client.chat.completions.create(
            model=model_option,
            messages=messages_for_api,
            stream=False 
        )
        
        ai_response = completion.choices[0].message.content
        session['messages'].append({"role": "assistant", "content": ai_response})
        session.modified = True

        return jsonify({
            "response": ai_response, 
            "search_results": search_results_display,
            "searched": should_search
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- CRITICAL FIX FOR RENDER DEPLOYMENT ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Get port from Render, default to 10000
    app.run(host='0.0.0.0', port=port)         # Listen on 0.0.0.0 (Public)
    
