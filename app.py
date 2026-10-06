import streamlit as st
import numpy as np
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import io

# --- KHỞI TẠO MODEL AI (Chỉ chạy 1 lần) ---
@st.cache_resource
def load_whisper_model():
    # Dùng model "base" cho nhẹ và nhanh
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_whisper_model()

# --- CÁC HÀM XỬ LÝ ÂM THANH VÀ DỊCH ---

def process_audio(audio_bytes):
    """Nhận file âm thanh, làm sạch nhiễu và trả về mảng numpy"""
    audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))
    
    # Chuyển sang mono nếu là stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
        
    # Làm sạch nhiễu (Noise Reduction)
    reduced_noise = nr.reduce_noise(y=audio_data, sr=samplerate, prop_decrease=0.8)
    
    return reduced_noise, samplerate

def speech_to_text(audio_data, samplerate):
    """Chuyển âm thanh thành text tiếng Anh (STT)"""
    segments, info = model.transcribe(audio_data, beam_size=5, language="en")
    text = " ".join([segment.text for segment in segments])
    return text.strip()

def clean_english_text(text):
    """Tự động chỉnh lại câu tiếng Anh hoàn chỉnh"""
    if not text:
        return ""
    # Viết hoa chữ cái đầu
    text = text[0].upper() + text[1:]
    # Thêm dấu chấm cuối câu nếu chưa có
    if text[-1] not in ['.', '?', '!']:
        text += '.'
    return text

def translate_to_vietnamese(text):
    """Dịch sang tiếng Việt"""
    if not text:
        return ""
    try:
        translated = GoogleTranslator(source='en', target='vi').translate(text)
        return translated
    except Exception as e:
        return f"Lỗi dịch thuật: {e}"

# --- GIAO DIỆN WEB (STREAMLIT) ---
st.title("🎙️ Demo Dịch Realtime Anh - Việt")
st.write("Hệ thống: Micro -> Làm sạch nhiễu -> STT -> Sửa câu -> Dịch tiếng Việt")

# Khởi tạo session_state để lưu trữ kết quả và tránh xử lý lại nhiều lần
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None
if "english_result" not in st.session_state:
    st.session_state.english_result = ""
if "vietnamese_result" not in st.session_state:
    st.session_state.vietnamese_result = ""

# Widget ghi âm từ Micro
audio_value = st.audio_input("Nhấn nút Micro để bắt đầu nói tiếng Anh:")

if audio_value:
    st.audio(audio_value, format="audio/wav")
    
    # Lấy dữ liệu bytes của âm thanh hiện tại
    audio_bytes = audio_value.getvalue()
    
    # Chỉ xử lý khi có file ghi âm MỚI (tránh việc bấm play audio làm chạy lại AI)
    if audio_bytes != st.session_state.last_audio_bytes:
        st.session_state.last_audio_bytes = audio_bytes
        
        with st.spinner("Đang xử lý âm thanh và nhận diện giọng nói..."):
            # Bước 1 & 2: Làm sạch âm thanh
            cleaned_audio, sr = process_audio(audio_bytes)
            
            # Bước 3: Speech to Text
            raw_text = speech_to_text(cleaned_audio, sr)
            
            # Bước 4: Làm sạch câu tiếng Anh
            final_english = clean_english_text(raw_text)
            
            # Bước 5: Dịch sang tiếng Việt
            vietnamese_text = translate_to_vietnamese(final_english)
            
            # Lưu kết quả vào session state
            st.session_state.english_result = final_english
            st.session_state.vietnamese_result = vietnamese_text

    # Hiển thị kết quả (lấy từ session_state ra)
    if st.session_state.english_result:
        st.success("Hoàn thành!")
        st.subheader("🇬🇧 Tiếng Anh (Đã làm sạch):")
        st.info(st.session_state.english_result)
        
        st.subheader("🇻🇳 Tiếng Việt:")
        st.success(st.session_state.vietnamese_result)