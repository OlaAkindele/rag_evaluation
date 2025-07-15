# file: quick_key_test.py
import os
import rag_evaluation as rag

# 0️⃣  make sure NO env vars will be consulted
os.environ.pop("OPENAI_API_KEY",  None)
os.environ.pop("GEMINI_API_KEY",  None)

# 1️⃣  inject one-time keys
rag.set_api_key("openai",  "REMOVED")
rag.set_api_key("gemini",  "AIzaSyAYhhtxaW3YV2Eo-bctJGTkLrpqlC-8RQc")

# 2️⃣  call the evaluator without providing a client
query     = "Explain gravitational waves in 2 lines."
document  = "Gravitational waves are ripples in spacetime predicted by Einstein..."
response  = "Gravitational waves are ripples..."

print("⇢ GPT-test")
print(rag.evaluate_response(query, response, document,
                            model_type="openai",
                            model_name="gpt-4o-mini"))      # uses the cached OpenAI key

print("\n⇢ Gemini-test")
print(rag.evaluate_response(query, response, document,
                            model_type="gemini",
                            model_name="gemini-2.5-flash"))   # uses the cached Gemini key
