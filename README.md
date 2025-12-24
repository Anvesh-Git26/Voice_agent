
📜 Objective
To build a voice-first, agentic AI system capable of autonomously reasoning, planning, and acting in a native Indian language (e.g., Telugu, Hindi, Marathi). The system acts as a Government Welfare Service Agent, helping users identify eligible public schemes through natural conversation, adhering to a "Planner-Executor-Evaluator" agentic workflow.
🚀 Key Features
 * 🎙️ Voice-First Interface: Complete hands-free interaction. The agent listens (STT), processes, and speaks back (TTS) in the user's native language.
 * 🧠 True Agentic Workflow: Uses a reasoning loop (Planner → Executor → Evaluator) rather than simple chatbot logic.
 * 🛠️ Multi-Tool Integration:
   * Scheme Retriever: Fetches scheme details from schemes.json.
   * Eligibility Engine: Compares user data (age, income, location) against scheme criteria.
 * 🌏 Native Language Support: Full pipeline support (STT/LLM/TTS) for Indian languages.
 * 💾 Conversation Memory: Remembers user details (e.g., "I am 25 years old") across turns to handle context and contradictions.
 * 🛡️ Robust Failure Handling: Gracefully handles unrecognized voice inputs or missing data by asking clarifying questions.
🏗️ Architecture & Approach
The system follows a modular pipeline designed for low latency and high accuracy in vernacular languages.
1. The Pipeline
 * Input (STT): User speech is captured and transcribed using Google Speech Recognition / OpenAI Whisper (configured for the specific Indian language).
 * Agent Brain (LLM): The core logic is powered by LLM (e.g., GPT-4o / Groq Llama-3) via LangChain. The system uses a system prompt that enforces:
   * Identity: Government Service Assistant.
   * Language: Output strictly in the target native language.
   * Reasoning: Decide which tool to call based on user intent.
 * Tool Execution:
   * If the user asks "What schemes are there?", the agent calls the Scheme_Retrieval_Tool.
   * If the user asks "Am I eligible?", the agent calls the Eligibility_Calculator_Tool.
 * Response Generation: The LLM synthesizes the tool outputs into a natural, conversational response.
 * Output (TTS): The text response is converted back to audio using gTTS (Google Text-to-Speech) or ElevenLabs and played to the user.
2. Decision Flow (Agent Loop)
 * Plan: User intent is analyzed (e.g., "User wants to apply for Rythu Bandhu").
 * Execute: Check if required info (Age, Land size) is present in memory.
   * If Missing: Ask the user for details.
   * If Present: Query the database (schemes.json).
 * Evaluate: Verify if the retrieved scheme matches the user's constraints before answering.
🛠️ Tech Stack
 * Language: Python 3.10+
 * Orchestration: LangChain / Phidata (Agent structure)
 * LLM: OpenAI GPT-4o / Groq (Llama 3.1) / Gemini Flash
 * Speech-to-Text (STT): SpeechRecognition (Google API) / Whisper
 * Text-to-Speech (TTS): gTTS / pyttsx3
 * Data Source: schemes.json (Structured database of govt schemes)
 * Interface: Streamlit (for Web UI) or Terminal
📂 File Structure
Voice_agent/
├── main.py             # Main application entry point (Streamlit/CLI app)
├── schemes.json        # Database containing government scheme details
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

⚡ Setup & Installation
Prerequisites
 * Python 3.8 or higher
 * An OpenAI API Key (or Groq/Gemini Key depending on implementation)
Step 1: Clone the Repository
git clone https://github.com/Anvesh-Git26/Voice_agent.git
cd Voice_agent

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Configure Environment
Create a .env file in the root directory and add your API keys:
OPENAI_API_KEY="your_api_key_here"
# GROQ_API_KEY="if_using_groq"

Step 4: Run the Application
# For Streamlit Interface
streamlit run main.py

# OR for Terminal Interface
python main.py

☁️ Running on Google Colab
Since Google Colab runs on a remote server, we cannot access your local microphone directly. We use ngrok to tunnel the Streamlit UI to a web address where you can use your browser's microphone.
1. Copy and paste the following code block into a Google Colab cell:
# 1. Install dependencies
!pip install streamlit pyngrok openai gtts speechrecognition pyaudio

# 2. Clone the repo (if not already uploaded)
!git clone https://github.com/Anvesh-Git26/Voice_agent.git
%cd Voice_agent

# 3. Add your API Key (Replace with actual key)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# 4. Authenticate ngrok (Required to expose the server)
# Sign up at ngrok.com to get your authtoken
!ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN

# 5. Run Streamlit in the background
get_ipython().system_raw('streamlit run main.py &')

# 6. Expose via ngrok
from pyngrok import ngrok
public_url = ngrok.connect(8501).public_url
print(f"🚀 Application running at: {public_url}")

2. Click the public_url generated.
3. Interact with the Voice Agent directly in the browser.
🧪 Evaluation & Testing Scenarios
We tested the agent against the following mandatory scenarios:
| Scenario | User Query | Agent Action | Status |
|---|---|---|---|
| Discovery | "Tell me about farming schemes." | Calls Scheme_Search tool, retrieves "Rythu Bandhu", answers in native language. | ✅ Pass |
| Eligibility | "Am I eligible for this?" | Checks memory for Age/Income. If missing, asks user: "What is your annual income?" | ✅ Pass |
| Edge Case | "I want to apply for NASA." | Replies: "I only handle Government Welfare Schemes. Can I help you with a ration card?" | ✅ Pass |
| Contradiction | User says "Age 20" then "Age 60" | Agent detects conflict: "Previously you said 20. Which is correct?" | ✅ Pass |

