import os
from dotenv import load_dotenv
from openai import OpenAI
from STT import STT
import warnings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
class ExtractMedicine:
    def __init__(self):
        load_dotenv(dotenv_path=".env")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(
            model="gpt-4", temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        prompt_content = """
            당신은 의약품 추천 전문가입니다. 사용자가 말하는 증상에 대해 적절한 의약품을 추천해주세요.
            <목표>
            - 사용자가 말하는 증상에 대해 의약품을 추천해야 합니다.
            - 의약품 리스트에 맞는 추천을 하세요.
            <의약품 리스트>
            - 감기약, 진통제, 파스, 목캔디, 소화제, 해열제, 비타민, 근육이완제, 두통약, 수면유도제
            <출력 형식>
            - 다음 형식으로 의약품과 증상을 구분하여 출력하세요: [의약품1 의약품2 ... / 증상1 증상2 ...]
            - 의약품과 증상은 각각 공백으로 구분하고, 의약품이 없으면 공백 없이 비워둡니다.
            <특수 규칙>
            - "목이 아프다"는 증상에 대해 "목캔디", "감기약" 등을 추천합니다.
            - "두통이 있다", "머리가 아프다"는 증상에 대해 "두통약", "진통제" 등을 추천합니다.
            - "근육이 아프다", "뻐근하다", "넘어졌다", "결리다"는 증상에 대해 "파스", "근육이완제" 등을 추천합니다.
            - "소화불량"은 "소화제" 등을 추천합니다.
            - "잠이 안온다"는 "수면유도제"를 추천합니다
            <사용자 증상>
            "{user_input}"
        """
        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=prompt_content
        )
        self.lang_chain = self.prompt_template | self.llm
    def extract_medicine(self, output_message):
        response = self.lang_chain.invoke({"user_input": output_message})
        result = response.content.strip().split("/")
        if len(result) != 2:
            warnings.warn("추천된 의약품과 증상이 더 많이 있습니다.")
            return None
        medicine, symptom = result[0], result[1]
        medicine = medicine.split()
        symptom = symptom.split()
        print(f"llm의 의약품 추천: {medicine}")
        print(f"llm의 증상 추출: {symptom}")
        return medicine, symptom
    def text_to_speech(self, text, filename="recommendation.mp3"):
        """OpenAI TTS 사용"""
        with self.client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        ) as response:
            response.stream_to_file(filename)
        os.system(f"play {filename}")  # 리눅스 환경이면 play, 윈도우면 다른 재생 방식 필요
    def get_user_response(self):
        self.text_to_speech("추천드린 의약품을 드리시겠습니까? 예 또는 아니오로 답해주세요.")
        stt = STT(os.getenv("OPENAI_API_KEY"))
        user_response = stt.speech2text()
        if "예" in user_response or "네" in user_response:
            return "yes"
        elif "아니요" in user_response or "아니" in user_response:
            return "no"
        else:
            return None
if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    stt = STT(openai_api_key)
    output_message = stt.speech2text()
    extract_medicine = ExtractMedicine()
    medicine, symptoms = extract_medicine.extract_medicine(output_message)
    if medicine:
        recommended_medicines = " ".join(medicine)
        extract_medicine.text_to_speech(f"{recommended_medicines}를 추천드립니다.")
        user_response = extract_medicine.get_user_response()
        if user_response == "yes":
            extract_medicine.text_to_speech(f"{recommended_medicines}를 드리겠습니다.")
        elif user_response == "no":
            extract_medicine.text_to_speech("다시 증상을 말씀해 주세요.")
            output_message = stt.speech2text()
            medicine, symptoms = extract_medicine.extract_medicine(output_message)
            recommended_medicines = " ".join(medicine)
            extract_medicine.text_to_speech(f"{recommended_medicines}를 추천드립니다.")
    else:
        extract_medicine.text_to_speech("추천할 의약품이 없습니다.")
