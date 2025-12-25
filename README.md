
# 🗣️ Voice-First Native Language Service Agent (Telugu)

## 📋 Objective

To build an autonomous, voice-first AI agent capable of reasoning, planning, and acting in a native Indian language (**Telugu**). The agent assists users in identifying and applying for government welfare schemes by identifying eligibility based on age, occupation, and region.

Unlike simple chatbots, this system uses an **Agentic Workflow** (Planner-Executor pattern) to autonomously determine when to ask follow-up questions and when to fetch data.

## ✨ Key Features

* **🎙️ Voice-First Interaction:** End-to-end audio processing (Speech-to-Text & Text-to-Speech) in Telugu.
* **🧠 Agentic Reasoning:** Uses **LangGraph** to maintain state and make decisions (e.g., "Do I have enough info?" vs "I need to search").
* **⚡ High-Performance LLM:** Powered by **Groq (Llama-3-70b)** for near-instant inference suitable for voice conversations.
* **🔍 RAG (Retrieval Augmented Generation):** vector-based semantic search using **ChromaDB** to find relevant schemes even with fuzzy user queries.
* **💾 Contextual Memory:** Remembers user details (Age, Occupation) across multiple conversation turns.

---

## 🏗️ Architecture

The system follows a **Router-Retriever-Generator** architecture:

1. **Input:** User speaks in Telugu -> Transcribed to Text.
2. **Extractor Node:** Llama-3 extracts structured data (JSON) from the unstructured speech.
3. **Router (Decision) Node:**
* *If info is missing:* Routes to **Asker Node** (Generates a follow-up question).
* *If info is complete:* Routes to **Searcher Node**.


4. **Searcher Node:** Queries **ChromaDB** for matching schemes and verifies eligibility.
5. **Output:** Synthesizes the final answer back into Telugu Audio.

---

## 🚀 How to Run (Google Colab Recommended)

This project is optimized for Google Colab to utilize GPU resources for Vector Embeddings.

### **Step 1: Get the Code**

Clone this repository or download the files:

* `schemes.json` (The Database)
* The `.ipynb` notebook file (The Code)

### **Step 2: Setup in Colab**

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload your notebook file.
3. **Crucial Step:** Upload the `schemes.json` file to the **Files** section (folder icon on the left).

### **Step 3: Dependencies**

Run the first cell to install the required environment:

```python
!pip install -qU -r requirements.txt
!apt-get install -y ffmpeg

```

### **Step 4: API Configuration**

You will need a free API Key from [Groq Console](https://console.groq.com/keys).

* Enter your key in the configuration cell:
```python
GROQ_API_KEY = "gsk_..."

```



### **Step 5: Run the Agent**

Run the final cell. The system will initialize:

> `✅ Groq Agent Ready! Listening...`

Click the **RED BUTTON** to speak (e.g., *"Naaku government schemes kavali"*). The agent will respond vocally.

---

## 📂 Project Structure

```bash
├── schemes.json        # Knowledge Base: Contains details of Govt schemes (Rythu Bandhu, etc.)
├── requirements.txt    # Dependencies list
├── Agent_Code.ipynb    # Main source code (Jupyter Notebook)
└── README.md           # Documentation

```

## 🛠️ Technology Stack

* **Orchestration:** LangGraph, LangChain
* **LLM:** Meta Llama-3-70b (via Groq API)
* **Vector DB:** ChromaDB
* **Embeddings:** Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
* **Audio:** SpeechRecognition (STT), gTTS (TTS), FFmpeg

---


* *Role:* AI & NLP Engineer
* *Focus:* Building Agentic AI Systems & Competitive Programming.
