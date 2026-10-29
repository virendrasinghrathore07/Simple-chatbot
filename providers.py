# providers.py alag-alag AI services se connect hone wala code organize karti hai, taaki main.py simple aur readable rahe.  
import requests    # python package 
from google import genai       
from openai import OpenAI

import config


def ask_openai(message):
    """Send one message to OpenAI and return the answer as text."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.responses.create(                                # client.response.create() is a openai 's method
        model=config.OPENAI_MODEL,
        instructions=config.SYSTEM_PROMPT,
        input=message,
    )
    return response.output_text                                 # response.output.txt is also a method to get output as text kyuki ye nhi lagayenge to faltu ki information bhi aayegi jese kitne time m question llm tak pahucha , kitne token lage , etc.


def ask_gemini(message):
    """Send one message to Google Gemini and return the answer as text."""
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=message,
        config={"system_instruction": config.SYSTEM_PROMPT},
    )
    return response.text


def ask_ollama(message):
    """Send one message to a model running locally in Ollama."""
    # Ollama has no API key: it is a plain HTTP server running on your own machine.
    try:
        response = requests.post(                                         # post() is a method
            config.OLLAMA_BASE_URL + "/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": message},                              # system ka role hoga to system prompt ka message jayega
                    {"role": "user", "content": message},                              # user ka role hoga to user ka message jayega
                ],
                "stream": False,         # false m content likhta hua nhi dikhega and true kare to dikhega
            },
            timeout=120,                # 120 sec m code ne output nhi diya to timeout ho jayega
        )
    except requests.exceptions.ConnectionError:                       # agar try fail hua to except chalega and requests module se exception aayega connection error ka 
        raise RuntimeError(                                           # or Runtime error generate hoga   
            "Could not reach Ollama at " + config.OLLAMA_BASE_URL + ". Is Ollama running?"
        )

    response.raise_for_status()                          # raise_for_status() is a method jo response deta h 
    return response.json()["message"]["content"]         # return kar do response ko as json jisme message hoga jo aage se aaya h and uska content hoga jo aage se anwer aayega 
