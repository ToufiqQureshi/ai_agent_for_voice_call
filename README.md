# 🎙️ AI Voice Sales Agent

A smart, voice-enabled AI sales assistant built using `Vocode`, `LangChain`, `Streamlit`, and real-time voice interaction APIs (Azure TTS, Deepgram ASR, OpenAI). This AI helps potential clients understand **RevMerito’s IT services** and captures leads through natural, dynamic conversations.

---

## 🚀 Features

- 🎤 Real-time Speech-to-Text (Deepgram ASR)  
- 🧠 Sales-focused conversational AI (LangChain + OpenAI)  
- 🗣️ Voice output with Azure Text-to-Speech  
- 🧾 Built-in company details, testimonials, and value propositions  
- 🔄 Consultative & Direct sales modes  
- 📊 Interactive Streamlit dashboard  
- 🐳 Fully Dockerized and Kubernetes-ready  

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- Vocode SDK
- LangChain
- OpenAI GPT-4 API
- Deepgram ASR
- Azure TTS

---

## 📁 Project Structure

. ├── app.py # Main Streamlit application ├── requirements.txt # Python dependencies ├── Dockerfile # Docker setup ├── .env # Environment variables (not pushed to GitHub) ├── k8s/ │ ├── deployment.yaml # Kubernetes Deployment manifest │ └── service.yaml # Kubernetes Service manifest

yaml
Copy
Edit

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ToufiqQureshi/ai_agent_for_voice_call.git
cd ai_agent_for_voice_call
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Set Environment Variables
Create a .env file with your API keys:

env
Copy
Edit
OPENAI_API_KEY=your_openai_key
AZURE_SPEECH_KEY=your_azure_key
AZURE_SPEECH_REGION=eastus
DEEPGRAM_API_KEY=your_deepgram_key
🐳 Docker Usage
Build the Docker Image
bash
Copy
Edit
docker build -t revmerito-sales-ai .
Run the Container
bash
Copy
Edit
docker run -p 8501:8501 --env-file .env revmerito-sales-ai
☸️ Kubernetes Deployment
1. Apply Kubernetes Manifests
bash
Copy
Edit
kubectl apply -f k8s/
2. Access the App
Get the external IP:

bash
Copy
Edit
kubectl get service revmerito-voice-app
🛠️ Customization
You can modify the following:

Product Focus: Update in Streamlit UI (app.py)

Sales Style: Choose between "consultative" or "direct"

Company Details: Edit the REVMERITO_INFO dictionary
