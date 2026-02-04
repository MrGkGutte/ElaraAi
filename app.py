import os
import json
import pytz
import io
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from groq import Groq
from tavily import TavilyClient
from fpdf import FPDF

# --- CONFIGURATION ---
app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

chat_memory = {}
MODEL_FAST = "llama-3.1-8b-instant" 
MODEL_SMART = "llama-3.3-70b-versatile"

# --- 1. LEGAL SEARCH ROUTER ---
def check_if_search_needed(user_query):
    try:
        system_instruction = """
        You are a legal search router for 'Pocket Lawyer'. 
        Return 'YES' if the query asks for:
        - Latest news on Indian laws, recent judgments (SC/HC), or new amendments like BNS/BNSS.
        - Current contact info of consumer courts or local legal authorities.
        Return 'NO' if it's general legal advice, drafting help, or simple conversation.
        Reply ONLY with 'YES' or 'NO'.
        """
        response = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_query}],
            temperature=0, max_tokens=5
        )
        return "YES" in response.choices[0].message.content.strip().upper()
    except: return False

# --- 2. HELPERS ---
def get_india_context():
    india_tz = pytz.timezone('Asia/Kolkata')
    return datetime.now(india_tz).strftime("%I:%M %p, %d %b %Y")

def web_search(query):
    try:
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        return "\n\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in response.get('results', [])])
    except: return ""

# --- 3. MAIN AI LOGIC ---
def get_ai_response(user_input, session_id):
    india_time = get_india_context()
    if session_id not in chat_memory: chat_memory[session_id] = []
    
    search_context = web_search(user_input) if check_if_search_needed(user_input) else ""

    system_prompt = f"""
    You are **Pocket Lawyer**, an expert AI assistant for Indian Law.
    Tone: Professional yet simple 'Hinglish'.
    Instructions:
    1. Help common people understand their rights (Consumer, Police/FIR, Rent, Labor, etc.).
    2. Provide a 'Step-by-Step Action Plan' for every legal problem.
    3. If a draft is needed, provide it clearly.
    4. Identity: Created by Gahininath Gutte. 
    5. Context: India. Time: {india_time}.
    6. Safety: Mention you are an AI, not a human lawyer.
    """
    
    if search_context:
        system_prompt += f"\n\nLIVE WEB INFO:\n{search_context}"

    chat_memory[session_id].append({"role": "user", "content": user_input})
    messages_payload = [{"role": "system", "content": system_prompt}] + chat_memory[session_id][-6:]

    try:
        response = client.chat.completions.create(model=MODEL_SMART, messages=messages_payload)
        ai_reply = response.choices[0].message.content.strip()
        chat_memory[session_id].append({"role": "assistant", "content": ai_reply})
        return ai_reply
    except Exception as e: return f"Error: {str(e)}"

# --- 4. ROUTES ---
@app.route('/')
def home(): return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    reply = get_ai_response(data.get('message'), data.get('session_id', 'guest'))
    return jsonify({"reply": reply})

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    content = request.json.get('content', '')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=content.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf_output = io.BytesIO()
    pdf_str = pdf.output(dest='S').encode('latin-1')
    pdf_output.write(pdf_str)
    pdf_output.seek(0)
    return send_file(pdf_output, mimetype='application/pdf', as_attachment=True, download_name='Legal_Draft.pdf')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
