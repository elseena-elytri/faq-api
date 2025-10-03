from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import json
import re

app = Flask(__name__)
CORS(app)

# Load the local HuggingFace model (LaMini)
faq_generator = pipeline("text2text-generation", model="MBZUAI/LaMini-Flan-T5-783M")



@app.route("/generate-faq", methods=["POST"])
def generate_faq():
    data = request.get_json()
    context = data.get("context")

    if not context:
        return jsonify({"error": "Missing context"}), 400

    # 💡 Use a smarter prompt to force diversity
    prompt = f"""
Based on the content below, generate 3 **different** frequently asked questions and their answers.
Each FAQ should be unique and not repeat the same question. Do NOT repeat the same question or answer.


Output ONLY in valid JSON like this:
[
  {{
    "question": "Question 1?",
    "answer": "Answer 1"
  }},
  {{
    "question": "Question 2?",
    "answer": "Answer 2"
  }},
  {{
    "question": "Question 3?",
    "answer": "Answer 3"
  }}
]

Content:
\"\"\"{context}\"\"\"
"""

    try:
        result = faq_generator(prompt, max_new_tokens=512)[0]["generated_text"].strip()

        # Try to parse proper JSON
        try:
            faqs = json.loads(result)
            return jsonify({"faqs": faqs})
        except json.JSONDecodeError:
            # 🧠 Backup regex parse for repeated fallback format
            qa_pairs = re.findall(r'(?:"question"\s*:\s*")(.+?)"(?:,\s*"answer"\s*:\s*")(.+?)"', result)
            if qa_pairs:
                faqs = [{"question": q.strip(), "answer": a.strip()} for q, a in qa_pairs]
                return jsonify({"faqs": faqs})
            else:
                return jsonify({"fallback_output": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
