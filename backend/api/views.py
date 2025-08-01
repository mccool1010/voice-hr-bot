import google.generativeai as genai
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
import os
from dotenv import load_dotenv
from django.conf import settings
import google.api_core.exceptions

load_dotenv()
genai.configure(api_key=settings.GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

session_memory = []

@api_view(["POST"])
def hr_bot(request):
    global session_memory

    user_input = request.data.get("message", "")
    role = request.data.get("role", "Software Engineer")

    system_prompt = f"You are a helpful and professional HR interviewer for the role of {role}. Ask smart questions one at a time."

    # Reset session if new interview starts
    if "start interview" in user_input.lower():
        session_memory = []

    chat = model.start_chat(history=session_memory)

    # First message should include the prompt
    if not session_memory:
        chat.send_message(system_prompt)

    try:
        reply = chat.send_message(user_input)
        answer = reply.text

        session_memory.append({"role": "user", "parts": [user_input]})
        session_memory.append({"role": "model", "parts": [answer]})

        return Response({"reply": answer})
    except google.api_core.exceptions.ResourceExhausted:
        return Response({"error": "Gemini API quota exceeded. Please try again later."}, status=429)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

class ChatView(APIView):
    def post(self, request):
        api_key = getattr(settings, "GEMINI_API_KEY", None)
        if not api_key:
            return Response({"error": "GEMINI_API_KEY not set"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        try:
            user_input = request.data.get("message", "")
            role = request.data.get("role", "Software Engineer")

            system_prompt = f"You are a helpful and professional HR interviewer for the role of {role}. Ask smart questions one at a time."

            # Reset session if new interview starts
            if "start interview" in user_input.lower():
                session_memory = []

            chat = model.start_chat(history=session_memory)

            # First message should include the prompt
            if not session_memory:
                chat.send_message(system_prompt)

            reply = chat.send_message(user_input)
            answer = reply.text

            session_memory.append({"role": "user", "parts": [user_input]})
            session_memory.append({"role": "model", "parts": [answer]})

            return Response({"reply": answer})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Add this to a script or your view temporarily to print available models
print([m.name for m in genai.list_models()])
