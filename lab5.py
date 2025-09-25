# lab5.py — Weather bot helper

import os
import requests
import streamlit as st

# ==================== Weather Function ====================
def get_current_weather(location: str, api_key: str = None) -> dict:
    """
    Get current weather data for a given location using OpenWeatherMap API.
    Converts Kelvin to Celsius and returns key weather info.
    """

    # Use API key from secrets.toml if not passed in
    if api_key is None:
        api_key = st.secrets.get("Weather_API_KEY") or os.getenv("Weather_API_KEY")
    if not api_key:
        st.error("Missing OpenWeatherMap API key! Add 'Weather_API_KEY' to .streamlit/secrets.toml")
        st.stop()

    # Normalize location (if input like "London, England", just take first part)
    if "," in location:
        location = location.split(",")[0].strip()

    # Build API request
    urlbase = "https://api.openweathermap.org/data/2.5/"
    urlweather = f"weather?q={location}&appid={api_key}"
    url = urlbase + urlweather

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return {"ok": False, "error": str(e), "location": location}

    try:
        # Extract temperatures (Kelvin → Celsius)
        temp = data['main']['temp'] - 273.15
        feels_like = data['main']['feels_like'] - 273.15
        temp_min = data['main']['temp_min'] - 273.15
        temp_max = data['main']['temp_max'] - 273.15
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']

        return {
            "ok": True,
            "location": location,
            "temperature": round(temp, 2),
            "feels_like": round(feels_like, 2),
            "temp_min": round(temp_min, 2),
            "temp_max": round(temp_max, 2),
            "humidity": humidity,
            "description": description
        }
    except KeyError:
        return {"ok": False, "error": "Unexpected API response format", "location": location}

# ==================== a6 Test ====================
"""
if __name__ == "__main__":
    result1 = get_current_weather("Syracuse, NY")
    result2 = get_current_weather("London, England")
    print(result1)
    print(result2)
"""


# ===================== Lab 5b: Construct the Travel weather and suggestion bot =====================

import streamlit as st
from openai import OpenAI


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("Lab 5 — Weather Suggestion Bot")
st.caption("Enter a city name to get the weather and clothing suggestions.")


city = st.text_input("Enter a city name:", "")


if not city.strip():
    city = "Syracuse, NY"


if st.button("Get Suggestion"):
    
    msgs = [
        {"role": "system", "content": "You are a helpful assistant that provides weather-based clothing advice."},
        {"role": "user", "content": f"What's the weather and what should I wear in {city}?"}
    ]

    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name, e.g., 'London, England'"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        tools=tools,
        tool_choice="auto",   
    )

    message = response.choices[0].message

    
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        if tool_call.function.name == "get_current_weather":
            args = eval(tool_call.function.arguments)  
            location = args.get("location", city)

            
            weather_info = get_current_weather(location, st.secrets["Weather_API_KEY"])

            
            msgs.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            msgs.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(weather_info),
            })

            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msgs,
            )

            suggestion = final_response.choices[0].message.content
            st.success(f"Weather in {location}: {weather_info}")
            st.info(f"Clothing suggestion: {suggestion}")

    else:
        st.warning("The model did not call the weather tool.")

# ===================== Lab 5c: Multi-provider =====================
import anthropic
import google.generativeai as genai

st.sidebar.header("Model Provider")
provider = st.sidebar.selectbox("Choose provider", ["OpenAI", "Claude", "Gemini"])


openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

claude_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-pro")


def get_suggestion_with_provider(weather_info, provider_choice):
    

    if provider_choice == "OpenAI":
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a weather and clothing assistant."},
                {"role": "user", "content": f"Weather info: {weather_info}. Suggest clothing."}
            ],
        )
        return resp.choices[0].message.content

    elif provider_choice == "Claude":
        msg = claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=400,
            messages=[
                {"role": "user", "content": f"Weather info: {weather_info}. Suggest clothing."}
            ]
        )
        return msg.content[0].text

    else:  # Gemini
        resp = gemini_model.generate_content(
            f"Weather info: {weather_info}. Suggest clothing."
        )
        return resp.text
