from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from gradio_client import Client

# Ganti dengan nama Space HF kamu
HF_SPACE = "BeDream/Dream"

class ChatAPIView(APIView):
    def post(self, request):
        user_message = request.data.get("message")
        if not user_message:
            return Response({"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            client = Client(HF_SPACE)
            # panggil API /chat di HF Space
            result = client.predict(message=user_message, api_name="/chat")
        except Exception as e:
            return Response({"error": "Failed to call HF Space", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"response": result})
