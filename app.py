from flask import Flask, request, jsonify
from dotenv import load_dotenv
import re
import os
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL")

client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

SYSTEM_PROMPT = """
You are an overly honest job recommendation assistant.
Be funny, light-hearted, and helpful.
Keep the answer clean: no markdown, no emojis, no bullet points.
Return simple plain text only.
"""

def clean_output(text: str) -> str:
    # Remove markdown bold/italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)

    # Remove markdown headers (# Title)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Remove emojis
    text = re.sub(r"[^\w\s.,!?'-]", "", text)

    # Remove list symbols (-, •, ●)
    text = re.sub(r"^[\-\•\●]\s*", "", text, flags=re.MULTILINE)

    # Replace multiple spaces with one
    text = re.sub(r"\s{2,}", " ", text)

    # Clean extra blank lines
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()

@app.route("/recommend-job", methods=["POST"])
def recommend_job():
    data = request.get_json(force=True)

    required = ["interests", "strengths", "weaknesses", "description"]
    missing = [field for field in required if not data.get(field)]

    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    user_msg = (
        f"Name: {data.get('name', 'User')}\n"
        f"Interests: {data['interests']}\n"
        f"Strengths: {data['strengths']}\n"
        f"Weaknesses: {data['weaknesses']}\n"
        f"Description: {data['description']}\n"
        "Give a funny, overly honest job recommendation."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]
        )

        choice = response.choices[0].message.content if response.choices else None
        if not choice:
            return jsonify({"error": "No response from Groq"}), 500

        cleaned = clean_output(choice)

        return jsonify({"ok": True, "recommendation": cleaned})

    except Exception as e:
        return jsonify({"error": "Groq API error", "details": str(e)}), 500


@app.route("/")
def home():
    return {"message": "API running!"}


if __name__ == "__main__":
    app.run(debug=True)