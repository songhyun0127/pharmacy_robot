import os
import time
import warnings
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from gtts import gTTS
from playsound import playsound
import numpy as np
from scipy.signal import resample
import cv2
from pyzbar.pyzbar import decode

from STT import STT
import MicController

# --- 환경 설정 ---
load_dotenv(dotenv_path=".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PATH = "/home/kimhm/ros2_ws/Tutorial/VoiceProcessing/hello_rokey_8332_32.tflite"
MODEL_NAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]

# --- 음성 파일 관리 ---
SOUND_FILES = {
    "recommendation": "recommendation.mp3",
    "prescription_query": "prescription_query.mp3",
    "prescription_instructions": "prescription_instructions.mp3",
    "symptom_request": "symptom_request.mp3",
    "valid_prescription": "valid_prescription.mp3",
    "invalid_prescription": "invalid_prescription.mp3",
    "recognition_cancelled": "recognition_cancelled.mp3",
    "ask_if_provide": "ask_if_provide.mp3",
    "provide_medicine": "provide_medicine.mp3",
    "ask_again": "ask_again.mp3",
    "no_recommendation": "no_recommendation.mp3",
    "unclear_answer": "unclear_answer.mp3"
}

# --- 약 설명 데이터 ---
MEDICINE_EXPLANATIONS = {
    "famotidine": "파모티딘은 위산을 억제하여 속쓰림이나 위산 과다에 효과가 있습니다. 하루 한 번, 자기 전에 복용하는 것이 일반적입니다.",
    "somnifacient": "수면유도제는 일시적인 불면 증상을 완화하는 데 도움을 줍니다. 잠들기 30분 전에 한 알을 복용하시고, 복용 후에는 운전과 같은 기계 조작을 피해야 합니다.",
    "allergy": "알러지약은 재채기, 콧물, 가려움증 같은 알러지 반응을 완화합니다. 하루 한 번 복용하며, 졸음이 올 수 있으니 주의가 필요합니다."
}

# --- 약 이름 한글 매핑 ---
MEDICINE_KOREAN_NAMES = {
    "famotidine": "파모티딘",
    "somnifacient": "자멘쏙",
    "allergy": "아루진"
}

def text_to_speech(text, filename):
    """텍스트를 음성으로 변환하고 재생합니다."""
    tts = gTTS(text, lang='ko')
    tts.save(filename)
    playsound(filename)

# --- Wake Word 클래스 ---
class WakeupWord:
    def __init__(self, buffer_size):
        from openwakeword.model import Model
        self.model = Model(wakeword_models=[MODEL_PATH])
        self.model_name = MODEL_NAME
        self.stream = None
        self.buffer_size = buffer_size

    def set_stream(self, stream):
        self.stream = stream

    def is_wakeup(self):
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs.get(self.model_name, 0.0)
        print(f"Confidence for {self.model_name}: {confidence:.2f}")
        return confidence > 0.15

# --- 의약품 추천 클래스 ---
class ExtractMedicine:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4", temperature=0.1, openai_api_key=OPENAI_API_KEY
        )
        
        prompt_content = """
        당신은 처방전 없이 구매 가능한 일반 의약품 추천 전문가입니다. 
        사용자가 말하는 증상을 듣고, 추천 가능한 의약품 목록에 해당하는 경우에만 약을 추천합니다.

        <매우 중요한 규칙>
        - 이 시나리오는 사용자가 "처방전이 없는 경우"이므로, 처방전이 필요한 의약품은 절대 추천해서는 안 됩니다.
        - 따라서 'penzal', 'sky', 'tg', 'zaide'는 사용자의 증상과 관계없이 무조건 False로 설정해야 합니다.

        <추천 가능한 의약품 목록 및 관련 증상>
        - famotidine: 속쓰림, 위산 과다, 위가 아픔
        - somnifacient: 잠이 안 옴, 불면증
        - allergy: 가려움, 알러지성 콧물, 재채기

        <출력 형식>
        - 반드시 다음 형식에 맞춰 각 의약품의 필요 여부를 True 또는 False로만 답해야 합니다.
        - 다른 설명 없이, 아래 형식만 정확히 지켜 한 줄로 출력해주세요.
        penzal:bool,sky:bool,tg:bool,zaide:bool,famotidine:bool,somnifacient:bool,allergy:bool

        <사용자 증상>
        "{user_input}"
        """

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=prompt_content
        )
        self.lang_chain = self.prompt_template | self.llm

    def extract(self, user_symptom):
        response = self.lang_chain.invoke({"user_input": user_symptom})
        response_text = response.content.strip()
        print(f"LLM 응답: {response_text}")

        medicine_flags = {}
        try:
            parts = response_text.split(',')
            for part in parts:
                key, value = part.split(':')
                medicine_flags[key.strip()] = value.strip().lower() == 'true'
            return medicine_flags
        except Exception as e:
            warnings.warn(f"LLM 응답 파싱 실패: {e}\n응답: {response_text}")
            return {
                "penzal": False, "sky": False, "tg": False, "zaide": False, 
                "famotidine": False, "somnifacient": False, "allergy": False
            }

# --- 사용자 응답 처리 ---
def get_user_response(stt_instance):
    while True:
        user_answer = stt_instance.speech2text()
        if any(keyword in user_answer for keyword in ["예", "네", "응", "어"]):
            return "yes"
        elif any(keyword in user_answer for keyword in ["아니요", "아니", "싫어"]):
            return "no"
        else:
            text_to_speech("답변이 명확하지 않습니다. 예 또는 아니오로 다시 말씀해 주세요.", SOUND_FILES["unclear_answer"])

# --- 비처방 의약품 추천 시나리오 (수정된 부분) ---
def run_drug_store(stt_instance):
    """증상을 듣고 비처방 의약품을 추천하는 전체 프로세스를 실행합니다."""
    text_to_speech("증상을 말씀해주세요.", SOUND_FILES["symptom_request"])
    
    user_symptom = stt_instance.speech2text()
    if not user_symptom:
        print("증상 음성 인식에 실패했습니다.")
        return

    extractor = ExtractMedicine()
    medicine_flags = extractor.extract(user_symptom)

    try:
        file_name = "decision.json"
        with open(file_name, 'w', encoding='utf-8') as json_file:
            json.dump(medicine_flags, json_file, ensure_ascii=False, indent=4)
        print(f"추천 결과가 '{file_name}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"JSON 파일 저장 중 오류 발생: {e}")

    recommended_medicines = [med for med, needed in medicine_flags.items() if needed]

    if not recommended_medicines:
        text_to_speech("추천해 드릴 의약품이 없습니다.", SOUND_FILES["no_recommendation"])
        return

    # --- ▼▼▼ TTS를 위한 한글 이름 변환 로직 추가 ▼▼▼ ---
    korean_names = [MEDICINE_KOREAN_NAMES.get(med, med) for med in recommended_medicines]
    recommended_text_for_tts = ", ".join(korean_names)

    # 추천 약품을 한글 이름으로 안내
    text_to_speech(f"{recommended_text_for_tts}를 추천드립니다.", SOUND_FILES["recommendation"])
    
    text_to_speech("추천드린 의약품을 드릴까요? 예 또는 아니오로 답해주세요.", SOUND_FILES["ask_if_provide"])
    user_choice = get_user_response(stt_instance)

    if user_choice == "yes":
        # 약을 전달할 때도 한글 이름으로 안내
        text_to_speech(f"{recommended_text_for_tts}를 드리겠습니다.", SOUND_FILES["provide_medicine"])
        
        time.sleep(0.5)
        for med in recommended_medicines:
            if med in MEDICINE_EXPLANATIONS:
                explanation = MEDICINE_EXPLANATIONS[med]
                explanation_filename = f"explanation_{med}.mp3"
                text_to_speech(explanation, explanation_filename)
                SOUND_FILES[f"explanation_{med}"] = explanation_filename
    else:
        text_to_speech("알겠습니다. 도움이 필요하시면 다시 말씀해주세요.", SOUND_FILES["ask_again"])

# --- QR 코드 처방전 처리 시나리오 ---
def handle_prescription_qr():
    """웹캠으로 QR코드를 인식하고 결과를 음성 안내합니다."""
    text_to_speech("처방전을 카메라에 인식시켜 주세요.", SOUND_FILES["prescription_instructions"])
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다.")
        return

    is_valid = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            qrcodes = decode(frame)
            if qrcodes:
                qr_data = qrcodes[0].data.decode('utf-8')
                print(f"인식된 QR코드 데이터: {qr_data}")
                is_valid = qr_data.lower() == 'true'
                break

            cv2.imshow('QR Code Scanner', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if is_valid is True:
        text_to_speech("유효한 처방전입니다. 약을 준비해 드리겠습니다.", SOUND_FILES["valid_prescription"])
    elif is_valid is False:
        text_to_speech("유효하지 않은 처방전입니다.", SOUND_FILES["invalid_prescription"])
    else:
        text_to_speech("인식이 취소되었습니다.", SOUND_FILES["recognition_cancelled"])

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    try:
        Mic = MicController.MicController()
        Mic.open_stream()

        wakeup = WakeupWord(Mic.config.buffer_size)
        wakeup.set_stream(Mic.stream)
        print("대기 중... 'hello rokey'를 말씀해주세요.")
        while not wakeup.is_wakeup():
            pass
        
        # playsound("start_sound.mp3")
        print("호출어가 감지되었습니다!")

        stt = STT(OPENAI_API_KEY)
        text_to_speech("처방전이 있으신가요? 예 또는 아니오로 답해주세요.", SOUND_FILES["prescription_query"])
        user_response = get_user_response(stt)

        if user_response == "yes":
            handle_prescription_qr()
        else:
            run_drug_store(stt)
            
        print("프로세스가 종료되었습니다.")

    except Exception as e:
        print(f"오류가 발생하여 프로그램을 종료합니다: {e}")
    finally:
        for sound_file in SOUND_FILES.values():
            if os.path.exists(sound_file):
                os.remove(sound_file)
        print("임시 음성 파일들을 삭제했습니다.")