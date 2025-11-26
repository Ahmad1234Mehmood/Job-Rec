from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

SYSTEM_PROMPT = """
You are an overly honest job recommendation assistant.
Be funny, light-hearted, and helpful.
"""

@app.route("/recommend-job", methods=["POST"])
def recommend_job():
    data = request.get_json()
    required_fields = ["interests", "strengths", "weaknesses", "description"]
    missing = [f for f in required_fields if f not in data or not str(data[f]).strip()]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    user_msg = f"""
    Name: {data.get('name','User')}
    Interests: {data['interests']}
    Strengths: {data['strengths']}
    Weaknesses: {data['weaknesses']}
    Description: {data['description']}
    Give a funny, overly honest job recommendation.
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_completion_tokens=1024,
        )

        if not hasattr(response, "choices") or len(response.choices) == 0:
            return jsonify({"error": "Groq returned no choices", "response": str(response)}), 500

        text = response.choices[0].message.content
        return jsonify({"ok": True, "recommendation": text})

    except Exception as e:
        return jsonify({"error": "Groq API error", "details": str(e)}), 500

@app.route("/")
def home():
    return {"message": "API running!"}

if __name__ == "__main__":
    app.run(debug=True)
