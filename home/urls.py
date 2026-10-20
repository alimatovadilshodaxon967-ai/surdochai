from django.urls import path
from home import views


urlpatterns = [
    path('', views.CameraPageView.as_view(), name='home'),
    path('voice-surdo/', views.CameraPageView.as_view(), name='camera'),
    path('surdo-voice/', views.VoicePageView.as_view(), name='voice'),
    path('video/', views.VideoPageView.as_view(), name='video'),
    path('api/tts/', views.TTSView.as_view(), name='tts'),
    path('api/gestures/', views.GestureListView.as_view(), name='gesture_list'),
    path('api/gestures/match/', views.GestureMatchView.as_view(), name='gesture_match'),
    path('api/ai-grammar/', views.AIGrammarView.as_view(), name='ai_grammar'),
    path('api/ai-assistant/', views.AIAssistantView.as_view(), name='ai_assistant'),
]

