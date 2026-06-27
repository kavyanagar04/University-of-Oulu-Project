# Emotion-Aware Robot Behavior Engine 🤖🎙️
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/kavyanagar04/University-of-Oulu-Project/blob/main/Oulu_Project_Colab_Final_edited.ipynb)

An end-to-end AI pipeline that processes raw speech audio, transcribes it, detects the speaker's exact emotion using a fine-tuned deep learning model, and generates an empathetic, emotionally-aligned vocal response.

## 🚀 Architecture
- **Stage 0 (Speech-to-Text):** OpenAI Whisper Base 
- **Stage 1 (Speech Emotion Recognition):** Fine-tuned `facebook/wav2vec2-base`
- **Stage 2 (Empathetic Generation):** LLaMA 3.1 8B (via Groq API)
- **Stage 3 (Text-to-Speech):** Google TTS (gTTS)

## 📊 Performance
- **Baseline SVM Accuracy:** ~60% (using 84 acoustic features like MFCCs & Pitch)
- **Wav2Vec2 Weighted Accuracy (WA):** ~64.4% on IEMOCAP Session 5 (4-class)
- **End-to-End Latency:** ~1.85s on Google Colab T4 GPU

## 🛠️ How to Run on Colab
Because this project requires a GPU, it is designed to run seamlessly on Google Colab.

1. Download the pre-trained Wav2Vec2 model and the IEMOCAP parquet dataset from [https://drive.google.com/drive/folders/1KlnbkRpVwGxzQ__EN3YlnmuMXaVQTi8K?usp=sharing].
2. Upload the unzipped files into a folder named `Oulu Project` in the root of your Google Drive.
3. Open `notebooks/Emotion_Aware_Robot_Pipeline.ipynb` in Google Colab.
4. Input your Groq API key when prompted and click **Run All**.
