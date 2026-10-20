from django.shortcuts import render
from django.views import View
import cv2
import numpy as np
import os
import base64
import mediapipe as mp
try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.callbacks import TensorBoard
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
from django.http import HttpResponse
from django.http import StreamingHttpResponse


# Create your views here.

class HomaPageView(View):
    def get(self, request):
        context = {}
        return render(request, "index.html", context)

    
def webcam():
    mp_holistic = mp.solutions.holistic 
    mp_drawing = mp.solutions.drawing_utils 

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False                  
        results = model.process(image)                 
        image.flags.writeable = True                  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 

    def draw_styled_landmarks(image, results):
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                ) 
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

    actions = np.array(['hello', 'thanks', 'iloveyou'])
    model = None

    if HAS_KERAS and os.path.exists('action.h5'):
        try:
            model = Sequential()
            model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
            model.add(LSTM(128, return_sequences=True, activation='relu'))
            model.add(LSTM(64, return_sequences=False, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(actions.shape[0], activation='softmax'))
            model.load_weights('action.h5')
        except Exception as e:
            print("Model load error:", e)
            model = None

    colors = [(245,117,16), (117,245,16), (16,117,245)]
    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.8

    cap = cv2.VideoCapture(0)
    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Kamera ulanmadi yoki band", (100, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    _, buffer = cv2.imencode('.jpg', blank_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # 2. Prediction logic
                if model is not None:
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

                        # 3. Viz logic
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                        image = prob_viz(res, actions, image, colors)

                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, "Surdo AI - Imo-ishoralar aniqlanmoqda", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                _, buffer = cv2.imencode('.jpg', image)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    finally:
        cap.release()

import asyncio
from asgiref.sync import sync_to_async

async def async_webcam():
    gen = webcam()
    get_next = sync_to_async(lambda: next(gen, None), thread_sensitive=False)
    while True:
        try:
            frame = await get_next()
            if frame is None:
                break
            yield frame
            await asyncio.sleep(0.01)
        except Exception as e:
            print("Streaming error:", e)
            break

class VideoPageView(View):
    async def get(self, request):
        return StreamingHttpResponse(async_webcam(), content_type='multipart/x-mixed-replace; boundary=frame')
          
class CameraPageView(View):
    def get(self, request):
        return render(request, "camera.html", {})
    
class VoicePageView(View):
    def get(self, request):
        context = {}
        return render(request, "voice.html", context)

VOICE_AUDIO_CACHE = {}

VOICELAB_API_KEYS = [
    "vlk_e7R8sSc93z-qo11q2uauHiEhNeyzx4jeCbnJab4QfXM",  # Asosiy kalit
    "vlk_srm9l7U767RcNOLB8EehYML8HUsnt5KqLsJmia4-I1U",  # 1-Zaxira kalit
    "vlk_V7vy1tQvhAiGuU5y4vsTsJqcRRfVIEFnzpdJdIF5ods",  # 2-Zaxira kalit
]
CURRENT_KEY_INDEX = 0

VOICELAB_VOICES = {
    "madina": "voice_4ef06e6b-2229-4023-ad66-c2a7a2b736ff",   # Madina — Honey (Shirali ayol)
    "amir": "voice_8bae02c6-d078-4757-9d62-0fce6da23cf7",     # Amir — Storyteller (Shirali erkak)
    "sardor": "voice_8bae02c6-d078-4757-9d62-0fce6da23cf7",   # Amir
    "javohir": "voice_fa56cecf-9b0b-45ba-9ec3-494efb5b301f",  # Javohir — Presenter (Ta'limiy diktor)
    "kamola": "voice_47260d60-5635-4fb0-904d-0f3c19c04ce6",   # Kamola — Silk (Mayin qiz)
    "mohira": "voice_47260d60-5635-4fb0-904d-0f3c19c04ce6",   # Kamola
    "abdulhay": "voice_M-p5iz9z_kPWNm_C0cFqbC6X",             # Abdulhay Otaxo'jayev (Dublyaj)
}

class TTSView(View):
    async def get(self, request):
        import hashlib
        import os
        import json
        import urllib.request
        import urllib.error
        import asyncio
        from django.conf import settings

        text = request.GET.get('text', '').strip()
        voice = request.GET.get('voice', 'madina').lower()
        if not text:
            return HttpResponse(status=400)

        is_gemini_voice = ('gemini' in voice or 'google' in voice)

        # VoiceLab & Gemini milliy o'zbek ovozlari
        if is_gemini_voice:
            voicelab_id = "gemini_flash_tts"
        elif 'amir' in voice or 'sardor' in voice or 'george' in voice or 'erkak' in voice:
            voicelab_id = VOICELAB_VOICES["amir"]
        elif 'javohir' in voice or 'jasur' in voice:
            voicelab_id = VOICELAB_VOICES["javohir"]
        elif 'kamola' in voice or 'mohira' in voice or 'mayin' in voice:
            voicelab_id = VOICELAB_VOICES["kamola"]
        elif 'abdulhay' in voice:
            voicelab_id = VOICELAB_VOICES["abdulhay"]
        else: # Madina (standart)
            voicelab_id = VOICELAB_VOICES["madina"]

        # Kesh tekshirish (Disk va Xotira - 0ms javob)
        cache_key = hashlib.md5(f"{text}_{voice}_{voicelab_id}".encode('utf-8')).hexdigest()
        cache_dir = os.path.join(settings.BASE_DIR, 'static', 'audio_cache')
        os.makedirs(cache_dir, exist_ok=True)
        disk_file = os.path.join(cache_dir, f"{cache_key}.wav")

        if cache_key in VOICE_AUDIO_CACHE:
            return HttpResponse(VOICE_AUDIO_CACHE[cache_key], content_type="audio/wav")

        if os.path.exists(disk_file):
            try:
                with open(disk_file, 'rb') as f:
                    data = f.read()
                VOICE_AUDIO_CACHE[cache_key] = data
                return HttpResponse(data, content_type="audio/wav")
            except Exception:
                pass

        # Uzluksiz Ovoz Sintezi: Google Gemini TTS yoki VoiceLab.uz API
        loop = asyncio.get_event_loop()
        def synthesize_speech():
            global CURRENT_KEY_INDEX, CURRENT_GEMINI_KEY_INDEX

            # 1. Agar foydalanuvchi Gemini AI ovozini tanlagan bo'lsa
            if is_gemini_voice:
                total_g_keys = len(GEMINI_API_KEYS)
                for attempt in range(total_g_keys):
                    idx = (CURRENT_GEMINI_KEY_INDEX + attempt) % total_g_keys
                    api_key = GEMINI_API_KEYS[idx]
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash-lite-tts:generateContent?key={api_key}"
                    payload = json.dumps({"contents": [{"parts": [{"text": text}]}]}).encode("utf-8")
                    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(req, timeout=12) as resp:
                            data = json.loads(resp.read().decode("utf-8"))
                            b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
                            CURRENT_GEMINI_KEY_INDEX = idx
                            return base64.b64decode(b64)
                    except Exception as ge:
                        print(f"Gemini TTS kalit [{idx + 1}/{total_g_keys}] xatolik: {ge}")
                        continue

            # 2. VoiceLab.uz milliy ovozlari (yoki Gemini zaxirasi)
            url = "https://api.voicelab.uz/v1/tts"
            vl_id = voicelab_id if not is_gemini_voice else VOICELAB_VOICES["madina"]
            payload = json.dumps({
                "text": text,
                "voice_id": vl_id,
                "language": "uz"
            }).encode("utf-8")

            last_error = None
            total_keys = len(VOICELAB_API_KEYS)

            for attempt in range(total_keys):
                idx = (CURRENT_KEY_INDEX + attempt) % total_keys
                api_key = VOICELAB_API_KEYS[idx]
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                )
                try:
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        if resp.status == 200:
                            CURRENT_KEY_INDEX = idx  # Muvaffaqiyatli kalitni joriy qilib saqlash
                            return resp.read()
                except urllib.error.HTTPError as he:
                    last_error = he
                    print(f"VoiceLab API kalit [{idx + 1}/{total_keys}] limiti yoki xatosi ({he.code}), keyingi zaxira kalitga o'tilmoqda...")
                    continue
                except Exception as ex:
                    last_error = ex
                    print(f"VoiceLab API kalit [{idx + 1}/{total_keys}] ulanish xatosi: {ex}")
                    continue

            if last_error:
                raise last_error
            return None

        try:
            audio_stream = await loop.run_in_executor(None, synthesize_speech)
            if audio_stream:
                VOICE_AUDIO_CACHE[cache_key] = audio_stream
                try:
                    with open(disk_file, 'wb') as df:
                        df.write(audio_stream)
                except Exception:
                    pass
                return HttpResponse(audio_stream, content_type="audio/wav")
            else:
                return HttpResponse("Ovoz sintezi audio qaytarmadi", status=500)
        except Exception as vl_err:
            print("Ovoz sintezi xatosi:", vl_err)
            return HttpResponse(f"Ovoz sintezi xatosi: {vl_err}", status=503)


class GestureListView(View):
    def get(self, request):
        from .models import SignGesture
        include_daktil = request.GET.get('include_daktil', 'false').lower() == 'true'
        qs = SignGesture.objects.filter(is_active=True)
        if not include_daktil:
            qs = qs.exclude(category='Daktil')
        gestures = list(qs.values(
            'id', 'slug', 'name', 'category', 'hand_type', 'icon', 
            'spoken_text', 'description', 'audio_key', 'finger_pattern'
        ))
        from django.http import JsonResponse
        return JsonResponse({
            'status': 'success',
            'count': len(gestures),
            'gestures': gestures
        }, json_dumps_params={'ensure_ascii': False})


from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

@method_decorator(csrf_exempt, name='dispatch')
class GestureMatchView(View):
    def post(self, request):
        from .models import SignGesture
        from django.http import JsonResponse
        import json
        try:
            data = json.loads(request.body.decode('utf-8'))
        except Exception:
            data = request.POST

        pattern = data.get('finger_pattern', '').strip()
        slug = data.get('slug', '').strip()

        gesture = None
        if slug:
            gesture = SignGesture.objects.filter(slug=slug, is_active=True).exclude(category='Daktil').first()
        elif pattern:
            gesture = SignGesture.objects.filter(finger_pattern=pattern, is_active=True).exclude(category='Daktil').first()

        if gesture:
            return JsonResponse({
                'match': True,
                'gesture': {
                    'id': gesture.id,
                    'slug': gesture.slug,
                    'name': gesture.name,
                    'category': gesture.category,
                    'hand_type': gesture.hand_type,
                    'icon': gesture.icon,
                    'spoken_text': gesture.spoken_text,
                    'description': gesture.description,
                    'audio_key': gesture.audio_key
                }
            }, json_dumps_params={'ensure_ascii': False})

        return JsonResponse({'match': False, 'message': 'Imo-ishora topilmadi'}, status=404)


GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY", base64.b64decode("QVEuQWI4Uk42S0xvOGQ0Qmw3MlpRZl9TNkFya3VDTHdMaUc4RVZEVkxwVnVXdUU1MEh3bGc=").decode("utf-8")),
    base64.b64decode("QVEuQWI4Uk42SWZIRWx5eld0aHRfRGp6aW0xanNNRzdRbGtDQWFzbVVPRTNONFdrZE9FOFE=").decode("utf-8"),
    base64.b64decode("QVEuQWI4Uk42SkUzN3R4czZVY0VRLWtaZ0FiejBYb0p5dnNMV1d5aTRwd2YwWVgwUV9jVXc=").decode("utf-8")
]
CURRENT_GEMINI_KEY_INDEX = 0


@method_decorator(csrf_exempt, name='dispatch')
class AIAssistantView(View):
    async def post(self, request):
        import json
        import urllib.request
        import urllib.error
        from django.http import JsonResponse
        import asyncio

        try:
            body = json.loads(request.body.decode('utf-8'))
        except Exception:
            body = {}

        question = body.get('question', '').strip()
        lang = body.get('lang', 'uz').lower()

        if not question:
            return JsonResponse({'status': 'error', 'message': 'Savol kiritilmadi'}, status=400)

        lang_name = "O'zbek tili" if lang == 'uz' else ("English" if lang == 'en' else "Русский язык")
        system_instruction = (
            "Siz Surdo AI loyihasining o'zbek tilidagi ovozli aqlli yordamchisisiz. "
            f"Foydalanuvchi mikrofonda savol berdi. {lang_name}da juda samimiy, madaniy, "
            "odobli va qisqa (1-2 jumlada) aniq javob bering. "
            "QAT'IY QOIDA: Javob VoiceLab orqali ovozda o'qiladi, shuning uchun "
            "yulduzchalar (*), panjara (#), markdown, emoji yoki murakkab belgilar mutlaqo ishlatmang!"
        )

        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": question}]}],
            "generationConfig": {"temperature": 0.3}
        }

        loop = asyncio.get_event_loop()
        def ask_gemini():
            global CURRENT_GEMINI_KEY_INDEX
            total_keys = len(GEMINI_API_KEYS)
            last_error = None

            for attempt in range(total_keys):
                idx = (CURRENT_GEMINI_KEY_INDEX + attempt) % total_keys
                api_key = GEMINI_API_KEYS[idx]
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        data = json.loads(resp.read().decode('utf-8'))
                        text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                            text = text[1:-1].strip()
                        CURRENT_GEMINI_KEY_INDEX = idx
                        return text
                except urllib.error.HTTPError as he:
                    last_error = he
                    print(f"Gemini Assistant kalit [{idx + 1}/{total_keys}] xatolik ({he.code}): keyingi zaxira kalitga o'tilmoqda...")
                    continue
                except Exception as ex:
                    last_error = ex
                    print(f"Gemini Assistant kalit [{idx + 1}/{total_keys}] ulanish xatosi: {ex}")
                    continue

            if last_error:
                raise last_error
            return "Kechirasiz, hozir savolingizga javob bera olmadim."

        try:
            answer = await loop.run_in_executor(None, ask_gemini)
            return JsonResponse({'status': 'success', 'question': question, 'answer': answer, 'model': 'Google Gemini'})
        except Exception as e:
            print("Gemini Assistant error:", e)
            return JsonResponse({
                'status': 'success',
                'question': question,
                'answer': "Kechirasiz, savolingizga javob berishda ulanish xatosi yuz berdi.",
                'model': 'Fallback'
            })


@method_decorator(csrf_exempt, name='dispatch')
class AIGrammarView(View):
    async def post(self, request):
        import json
        import urllib.request
        import urllib.error
        from django.http import JsonResponse
        import asyncio

        try:
            body = json.loads(request.body.decode('utf-8'))
        except Exception:
            body = {}

        raw_words = body.get('words', [])
        if isinstance(raw_words, str):
            words = [w.strip() for w in raw_words.split(',') if w.strip()]
        else:
            words = [str(w).strip() for w in raw_words if str(w).strip()]

        lang = body.get('lang', 'uz').lower()
        if not words:
            default_empty = {
                'en': "Hello! Please perform some gestures first.",
                'ru': "Здравствуйте! Пожалуйста, покажите жесты сначала.",
                'uz': "Assalomu alaykum! Iltimos, avval imo-ishoralarni ko'rsating."
            }
            return JsonResponse({'status': 'success', 'sentence': default_empty.get(lang, default_empty['uz'])})

        words_str = ", ".join(words)

        lang_name = "O'zbek tili" if lang == 'uz' else ("English" if lang == 'en' else "Русский язык")
        system_instruction = (
            "Siz imo-ishoralarni nutqqa aylantiruvchi Surdo AI yordamchisisiz. "
            f"Foydalanuvchi ko'rsatgan uzuq-yuluq imo-ishora so'zlaridan {lang_name}da "
            "odobli, to'liq, ravon va madaniy bitta adabiy jumla tuzib bering. "
            "QAT'IY QOIDA: Faqat va faqat yakuniy gapni yozing! Kirish so'zlari ('Mana...', 'Albatta...'), "
            "tushuntirishlar, qo'shtirnoq yoki '**' belgilarini aslo qo'shmang!"
        )
        prompt = f"Imo-ishoralar so'zlari: {words_str}. Til: {lang_name}"

        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.25}
        }

        loop = asyncio.get_event_loop()
        def call_gemini():
            global CURRENT_GEMINI_KEY_INDEX
            total_keys = len(GEMINI_API_KEYS)
            last_error = None

            for attempt in range(total_keys):
                idx = (CURRENT_GEMINI_KEY_INDEX + attempt) % total_keys
                api_key = GEMINI_API_KEYS[idx]
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                try:
                    with urllib.request.urlopen(req, timeout=9) as resp:
                        data = json.loads(resp.read().decode('utf-8'))
                        text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                            text = text[1:-1].strip()
                        CURRENT_GEMINI_KEY_INDEX = idx  # Muvaffaqiyatli kalitni saqlash
                        return text
                except urllib.error.HTTPError as he:
                    last_error = he
                    print(f"Gemini API kalit [{idx + 1}/{total_keys}] xatolik ({he.code}): keyingi zaxira kalitga o'tilmoqda...")
                    continue
                except Exception as ex:
                    last_error = ex
                    print(f"Gemini API kalit [{idx + 1}/{total_keys}] ulanish xatosi: {ex}")
                    continue

            if last_error:
                raise last_error
            return None

        try:
            sentence = await loop.run_in_executor(None, call_gemini)
            return JsonResponse({'status': 'success', 'sentence': sentence, 'model': 'Google Gemini Flash'})
        except Exception as e:
            print("Gemini API error, falling back to local NLP:", e)
            lower_words = [w.lower() for w in words]
            has = lambda kw: any(kw in w for w in lower_words)
            if has('kasal') or has('yordam') or has('betob'):
                fb = {
                    'en': "Hello! I am feeling unwell, could you please help me urgently?",
                    'ru': "Здравствуйте! Я плохо себя чувствую, пожалуйста, окажите мне срочную помощь.",
                    'uz': "Assalomu alaykum! Men o'zimni yomon his qilyapman, iltimos, menga zudlik bilan yordam bera olasizmi?"
                }
            elif has('suv') or has('chanqa'):
                fb = {
                    'en': "Excuse me! I am really thirsty, could you please give me a glass of fresh water?",
                    'ru': "Простите! Я очень хочу пить, не могли бы вы дать мне стакан чистой воды?",
                    'uz': "Kechirasiz! Juda chanqadim, iltimos menga bir piyola toza suv bera olasizmi?"
                }
            elif has('rahmat') and has('ajoyib'):
                fb = {
                    'en': "Thank you so much from the bottom of my heart, everything was absolutely wonderful!",
                    'ru': "Огромное вам спасибо от всей души, всё прошло просто великолепно!",
                    'uz': "Sizga chin qalbimdan kattakon rahmat aytaman, barchasi nihoyatda ajoyib va go'zal bo'ldi!"
                }
            else:
                fb = {
                    'en': f"Respectfully: {words_str}. I wish you a wonderful day!",
                    'ru': f"С уважением: {words_str}. Желаю вам всего наилучшего!",
                    'uz': f"Hurmat bilan ma'lum qilaman: {words_str}. E'tiboringiz va tushunganingiz uchun tashakkur!"
                }
            return JsonResponse({'status': 'success', 'sentence': fb.get(lang, fb['uz']), 'model': 'Local NLP Fallback'})

    async def get(self, request):
        words = request.GET.get('words', '').split(',')
        lang = request.GET.get('lang', 'uz')
        # forward to post logic
        class FakeRequest:
            body = json.dumps({'words': words, 'lang': lang}).encode('utf-8')
        return await self.post(FakeRequest())

