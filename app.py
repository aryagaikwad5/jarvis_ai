from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import re
import os

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are JARVIS — an advanced AI assistant.
You give short, smart and confident answers.
You refer to yourself as 'Jarvis', never as 'AI' or 'assistant'.
STRICTLY follow the language instruction given in [LANG:...] tag at the start of each message."""

conversation_history = []


def detect_lang(text):
    has_hindi   = bool(re.search(r'[\u0900-\u097F]', text))
    has_english = bool(re.search(r'[a-zA-Z]{2,}', text))
    if has_hindi and has_english:
        return "HINGLISH"
    elif has_hindi:
        return "HINDI"
    else:
        return "ENGLISH"


@app.route("/ask", methods=["POST"])
def ask():
    global conversation_history
    try:
        data         = request.get_json()
        user_message = data.get("message", "").strip()
        lang         = data.get("lang", "ENGLISH")  # frontend sends detected lang
        print(f"User [{lang}]: {user_message}")

        if not user_message:
            return jsonify({"error": "Message is empty!"}), 400

        # Use frontend lang if provided, else detect server-side
        if lang not in ["ENGLISH", "HINDI", "HINGLISH"]:
            lang = detect_lang(user_message)

        # Language instruction map
        lang_map = {
            "ENGLISH":  "Reply in ENGLISH only. Do not use Hindi words.",
            "HINDI":    "Reply in HINDI only. Use Devanagari script. Do not use English.",
            "HINGLISH": "Reply in HINGLISH (Roman Hindi + English mix) only.",
        }

        # Prepend language tag to message
        tagged_message = f"[LANG:{lang} - {lang_map[lang]}]\n{user_message}"

        conversation_history.append({
            "role": "user",
            "content": tagged_message
        })

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
            max_tokens=512,
            temperature=0.7,
        )

        reply = response.choices[0].message.content
        print(f"Jarvis: {reply[:100]}...")

        conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        return jsonify({"reply": reply, "status": "ok", "lang": lang})

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history
    conversation_history = []
    print("Chat history reset.")
    return jsonify({"status": "Chat has been reset!"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Jarvis is online!", "model": MODEL_NAME})


if __name__ == "__main__":
    print("=" * 45)
    print("JARVIS Backend Starting...")
    print(f"Model : {MODEL_NAME}")
    print("Server: http://localhost:5000")
    print("=" * 45)
    app.run(debug=True, port=5000)