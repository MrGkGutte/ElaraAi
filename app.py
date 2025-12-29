import os
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS  # <--- IMPORT THIS
from groq import Groq
from tavily import TavilyClient

app = Flask(__name__)
CORS(app)  # <--- ENABLE CORS FOR ALL ROUTES

# A secret key is required for sessions.
app.secret_key = os.urandom(24)

# --- HELPER: SMART SEARCH ROUTER ---
def check_if_search_needed(prompt, client):
    """
    Uses a fast/small model to decide if the user prompt requires a web search.
    """
    try:
        system_instruction = """
        You are a search router. Analyze the user's query.
        Return 'YES' if the query requires external real-time information (news, weather, sports, specific facts, current events).
        Return 'NO' if the query is conversational, generic, creative writing, code generation, or about your identity ("Who are you").
        Reply ONLY with 'YES' or 'NO'.
        """
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", # Use fast model for decision
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5
        )
        decision = response.choices[0].message.content.strip().upper()
        return "YES" in decision
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
    groq_api_key = data.get('groq_api_key')
    tavily_api_key = data.get('tavily_api_key')
    model_option = data.get('model_option', 'llama-3.3-70b-versatile')
    use_search = data.get('use_search', False)

    if not groq_api_key:
        return jsonify({"error": "Groq API Key is missing."}), 400

    # Add User Message to Session
    session['messages'].append({"role": "user", "content": user_input})
    session.modified = True

    try:
        client = Groq(api_key=groq_api_key)
        
        # --- SMART SEARCH LOGIC ---
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

        # --- PREPARE PROMPT ---
        system_content = "You are Elara, a helpful and smart AI assistant created by Gk Gutte. Always introduce yourself as Elara when asked."
        
        if search_context:
            system_content += f"""
            \n\nLIVE WEB CONTEXT:\n{search_context}
            \n\nINSTRUCTION: Answer the user's question using the context above.
            """
        else:
            system_content += "\n\nINSTRUCTION: Answer the user's question based on your training data."

        messages_for_api = [{"role": "system", "content": system_content}]
        messages_for_api.extend(session['messages'])

        # --- CALL GROQ API ---
        completion = client.chat.completions.create(
            model=model_option,
            messages=messages_for_api,
            stream=False 
        )
        
        ai_response = completion.choices[0].message.content
        
        # Add AI Message to Session
        session['messages'].append({"role": "assistant", "content": ai_response})
        session.modified = True

        return jsonify({
            "response": ai_response, 
            "search_results": search_results_display,
            "searched": should_search
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    
