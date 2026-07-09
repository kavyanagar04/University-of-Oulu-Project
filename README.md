# Emotion-Aware Robot Behavior Engine 🤖🎙️

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/kavyanagar04/University-of-Oulu-Project/blob/main/Oulu_Project_Colab_Final_edited.ipynb)

An end-to-end AI pipeline designed to process raw human speech audio, transcribe it, detect the speaker's exact emotional state using a fine-tuned deep learning model, and dynamically generate an empathetic, emotionally-aligned vocal response in real-time. 

*Inspired by the ReSIn-HR (Real-Time Synchronized Interaction Framework for Emotion-Aware Humanoid Robots) research framework.*

---

## 🚀 The 4-Stage Inference Pipeline
The system is built on a modular, 4-stage architecture designed to mimic natural human empathetic conversation with ultra-low latency:

1. **Stage 0 (Speech-to-Text):** OpenAI Whisper Base model transcribes the raw audio waveform into semantic text.
2. **Stage 1 (Speech Emotion Recognition):** A `facebook/wav2vec2-base` transformer, fine-tuned on the IEMOCAP dataset, analyzes spatiotemporal acoustic features to predict the user's emotion (Angry, Happy, Neutral, Sad) and Valence-Arousal scores.
3. **Stage 2 (Empathetic Generation):** LLaMA 3.1 8B (hosted via the **Groq API**) receives the transcript and emotional label. Acting as an empathetic conversational agent, it generates a psychologically appropriate text response.
4. **Stage 3 (Text-to-Speech):** 
   - **Baseline:** A robust `gTTS` fallback mechanism guarantees pipeline stability.
   - **Advanced:** A custom **Microsoft Edge Neural TTS (`edge-tts`)** integration explicitly maps the predicted emotion to vocal prosody parameters (Pitch, Rate, and Volume), achieving highly expressive synthesis without paid cloud APIs.

---

## 📊 Performance & Latency Metrics
* **Baseline SVM Accuracy:** ~60% (trained on 84 manually extracted acoustic descriptors like MFCCs, Pitch, and ZCR).
* **Deep Learning Superiority (Wav2Vec2):** Achieved **~64.4% Weighted Accuracy (WA)** on the held-out IEMOCAP Session 5 test set.
* **End-to-End Latency:** Achieved a blistering total inference time of **~1.48 seconds** on a standard Google Colab T4 GPU (Audio In → Audio Out).
  * *LLM Generation (Groq):* ~0.2s
  * *Neural TTS Rendering:* ~1.1s

---

## 🧪 Generalization Testing (RAVDESS Zero-Shot)
To evaluate the true acoustic invariance of the fine-tuned Wav2Vec2 model, a massive Zero-Shot generalization test was conducted on a completely unseen, out-of-domain dataset (RAVDESS).

* **Test Scope:** 1,728 filtered 4-class audio samples.
* **Results:** Achieved **~29% Weighted Accuracy** (26% Unweighted).
* **Research Analysis:** The model exhibited a heavy bias towards predicting "Angry" (99% recall). This highlights a severe acoustic domain shift: IEMOCAP contains natural, conversational dialogue in a room, while RAVDESS contains exaggerated, theatrical speech in a sterile studio. The model successfully overfit to IEMOCAP's specific acoustic profile. Future real-world robotic deployments require multi-corpus training (e.g., IEMOCAP + RAVDESS + CREMA-D) to survive massive acoustic domain shifts.

---

## 🛠️ How to Run on Colab
Because this project utilizes heavy Transformer models and requires a GPU, it is designed to run seamlessly on Google Colab.

1. Download the pre-trained Wav2Vec2 model, the IEMOCAP parquet dataset, and the RAVDESS zero-shot dataset (`Ravdess Dataset.zip`) from [https://drive.google.com/drive/folders/1KlnbkRpVwGxzQ__EN3YlnmuMXaVQTi8K?usp=sharing].
2. Upload all the unzipped model folders and the `Ravdess Dataset.zip` file into a folder named `Oulu Project` in the root of your Google Drive.
3. Open `Oulu_Project_Colab_Final_edited.ipynb` in Google Colab.
4. Input your **Groq API key** when prompted in the setup cell.
5. Click **Run All** to execute the training, feature extraction, and the real-time inference pipeline!
