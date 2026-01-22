import os
import openai
from dotenv import load_dotenv, find_dotenv

# Load environment variables (for local dev support)
# We use find_dotenv() to locate .env in parent directories if running from subdirectory
load_dotenv(find_dotenv(), override=True)

def generate_explanation(label, confidence, exif_data, ela_score):
    """
    Generates a natural language explanation for the image authenticity prediction.

    Args:
        label (str): "REAL" or "FAKE (AI-GENERATED)".
        confidence (float): Confidence score (0.0 to 1.0) of the 'Real' class presumably, 
                            but passed in logic handles it. 
                            (Logic in app.py passes the dominant confidence).
        exif_data (dict): Dictionary of EXIF metadata.
        ela_score (float): Average pixel intensity of the ELA image (0-255 scale).
                           Higher values generally mean more noise/compression artifacts 
                           or potential manipulation in specific regions.

    Returns:
        str: A generated text explanation or an error message.
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ **OpenAI API Key not found.** Cannot generate AI-assisted explanation. Please set the `OPENAI_API_KEY` environment variable."

    try:
        client = openai.OpenAI(api_key=api_key)

        # Simplify EXIF for prompt context to avoid token limits or noise
        exif_summary = ", ".join([f"{k}: {v}" for k, v in exif_data.items()])
        if not exif_summary:
            exif_summary = "No EXIF metadata found."

        # Interpret ELA Score roughly for the prompt context
        # This is a heuristic communication to the LLM, not a strict rule.
        ela_context = f"Average Noise Level: {ela_score:.2f} (Scale 0-255)."
        if ela_score > 10: # Arbitrary threshold for "noisy" - usually ELA is very dark (close to 0) for pristine images
             ela_context += " Note: High noise levels detected, indicating potential resaving or manipulation."
        else:
             ela_context += " Note: Low noise levels detected, consistent with original/high-quality compression."

        prompt_system = (
            "You are a Digital Forensics Expert AI. Your task is to analyze technical image analysis data "
            "and provide a professional, neutral, and factual summary report.\n"
            "Do NOT invent facts. Do NOT say you looked at the image pixels (you only see metadata).\n"
            "If the confidence is low (below 70%), express uncertainty.\n"
            "Structure your response:\n"
            "1. **Analysis Conclusion**: One sentence summary.\n"
            "2. **Key Findings**: Bullet points on Model Prediction, EXIF consistency, and ELA indications.\n"
            "3. **Verdict**: Final assessment based on provided data."
        )

        prompt_user = (
            f"Analyze the following data for an image suspected of being Deepfake/AI-generated:\n\n"
            f"**Model Prediction**: {label} (Confidence: {confidence:.2%})\n"
            f"**EXIF Metadata**: {exif_summary}\n"
            f"**Error Level Analysis (ELA)**: {ela_context}\n\n"
            "Provide a short forensic report explaining these results to a non-expert user."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini", # Cost effective and capable enough
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.3, # Low temperature for factual consistency
            max_tokens=300
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ **Error generating explanation**: {str(e)}"
