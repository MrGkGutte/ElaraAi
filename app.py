import os
import io
import pytz
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from groq import Groq
from tavily import TavilyClient
from fpdf import FPDF

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

chat_memory = {}
MODEL_SMART = "llama-3.3-70b-versatile"

def get_india_time():
    return datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%I:%M %p, %d %b %Y")

@app.route('/')
def home(): return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    session_id = data.get('session_id', 'guest')
    
    if session_id not in chat_memory: chat_memory[session_id] = []
    
    # System Prompt with Drafting Instruction
    system_prompt = f"""You are **Pocket Lawyer**. Created by Gahininath Gutte. 
    Expert in Indian Law. Use 'Hinglish'. 
    IMPORTANT: When writing a formal notice or email, always start with 'Subject:' or 'To:' so the user can download it."""
    
    messages = [{"role": "system", "content": system_prompt}] + chat_memory[session_id][-6:] + [{"role": "user", "content": user_input}]
    
    try:
        response = client.chat.completions.create(model=MODEL_SMART, messages=messages)
        ai_reply = response.choices[0].message.content.strip()
        chat_memory[session_id].append({"role": "user", "content": user_input})
        chat_memory[session_id].append({"role": "assistant", "content": ai_reply})
        return jsonify({"reply": ai_reply})
    except Exception as e: return jsonify({"reply": f"Error: {str(e)}"})

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    content = request.json.get('content', '')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    safe_text = content.replace('**', '').replace('###', '').encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    
    pdf_output = io.BytesIO()
    pdf_str = pdf.output(dest='S')
    if isinstance(pdf_str, str): pdf_str = pdf_str.encode('latin-1')
    pdf_output.write(pdf_str)
    pdf_output.seek(0)
    return send_file(pdf_output, mimetype='application/pdf', as_attachment=True, download_name='Pocket_Lawyer_Draft.pdf')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
