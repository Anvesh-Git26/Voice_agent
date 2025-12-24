import os
import sys
import re
import json
import time
import base64
import getpass
import torch
import chromadb
import speech_recognition as sr
from typing import TypedDict, List, Dict
from gtts import gTTS
from IPython.display import HTML, Audio, display, Javascript, clear_output
from google.colab import output

# LangChain / Graph Imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from chromadb.utils import embedding_functions

# ==========================================
# PART 1: SETUP & CONFIGURATION
# ==========================================


GOOGLE_API_KEY = 'Enter your api here'

# 2. INTELLIGENT MODEL (Cloud - Gemini)
print("☁️  Initializing Cloud Model (Gemini)...")
CLOUD_MODEL_NAME = "gemini-2.5-flash-lite"
try:
    llm_cloud = ChatGoogleGenerativeAI(
        model=CLOUD_MODEL_NAME, 
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    print(f"❌ Cloud Model Init Failed: {e}")
    llm_cloud = None

# 3. FAST MODEL (Local - Hugging Face on T4 GPU)
LOCAL_MODEL_ID = "google/gemma-2-9b-it" 
print(f"🔄 Loading Local GPU Model: {LOCAL_MODEL_ID}...")

try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        return_full_text=False
    )

    llm_local = HuggingFacePipeline(pipeline=pipe)
    print("✅ Local GPU Model Loaded Successfully!")

except Exception as e:
    print(f"⚠️ Failed to load local model on GPU. Error: {e}")
    print("⚠️ Falling back to Cloud Model for extraction.")
    llm_local = llm_cloud

# 4. Initialize ChromaDB & LOAD JSON DATA
print("💾 Initializing Knowledge Base...")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient(path="./chroma_db")
# Note: Changing collection name to force refresh if schema changed
collection = client.get_or_create_collection(name="telangana_schemes", embedding_function=emb_fn)

if collection.count() == 0:
    json_file_path = "schemes.json" # <--- Ensure this file exists
    
    if os.path.exists(json_file_path):
        print(f"📂 Found '{json_file_path}'. Loading data...")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                schemes_data = json.load(f)
            
            documents = []
            ids = []
            metadatas = []
            
            for i, item in enumerate(schemes_data):
                # IMPROVED PARSING FOR YOUR JSON STRUCTURE
                scheme_id = item.get('scheme_id', f"scheme_{i}")
                name = item.get('scheme_name_te', item.get('scheme_id', 'Unknown'))
                desc = item.get('description_te', '')
                benefits = item.get('benefits', '')
                
                # Flatten eligibility criteria into readable text for the Vector DB
                eligibility = item.get('eligibility_criteria', {})
                elig_text = ", ".join([f"{k}: {v}" for k, v in eligibility.items()])
                
                # Create a rich text description for embedding
                content = f"Scheme Name: {name}. Description: {desc}. Benefits: {benefits}. Eligibility Criteria: {elig_text}"
                
                documents.append(content)
                ids.append(scheme_id)
                metadatas.append({"category": item.get('category', 'General')})
            
            if documents:
                collection.add(documents=documents, ids=ids, metadatas=metadatas)
                print(f"✅ Successfully added {len(documents)} schemes from JSON.")
                
        except Exception as e:
            print(f"❌ Error reading JSON file: {e}")
            
    else:
        print(f"⚠️ '{json_file_path}' not found. Please upload it.")
else:
    print(f"✅ Database loaded with {collection.count()} schemes.")

# ==========================================
# PART 2: AUDIO FUNCTIONS (JAVASCRIPT)
# ==========================================

RECORD_JS = """
var sleep  = time => new Promise(resolve => setTimeout(resolve, time))
var b2text = blob => new Promise(resolve => {
  var reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  
  var btn = document.createElement('button')
  btn.innerHTML = "🟥 STOP RECORDING"
  btn.style.background = "red"
  btn.style.color = "white"
  btn.style.border = "none"
  btn.style.padding = "10px"
  btn.style.borderRadius = "5px"
  btn.style.fontSize = "16px"
  btn.style.margin = "10px"
  btn.onclick = () => { recorder.stop() }
  document.body.append(btn)
  
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    btn.remove()
    resolve(text)
  }
})
"""

def record_audio(filename='input.wav'):
    """Records audio from browser, converts WebM to WAV via ffmpeg."""
    display(Javascript(RECORD_JS))
    print("👇 Click the RED button above to stop recording...")
    s = output.eval_js('record(0)')
    b = base64.b64decode(s.split(',')[1])
    
    with open('temp_audio.webm', 'wb') as f:
        f.write(b)
    
    # Convert to PCM WAV (Required for SpeechRecognition)
    os.system(f"ffmpeg -y -i temp_audio.webm -ac 1 -ar 16000 {filename} -loglevel error")
    return filename

def transcribe_audio(filename):
    """Transcribes audio using Google Web Speech API (Free)."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="te-IN")
        print(f"🎤 You said: {text}")
        return text
    except Exception as e:
        print(f"⚠️ Transcription Error: {e}")
        return ""

def speak_audio(text):
    """Synthesizes text to speech and plays it in Colab."""
    print(f"🤖 Agent: {text}")
    if not text: return
    
    # --- CLEANING STEP ---
    # Remove asterisks (*), hashes (#), and underscores (_) used for markdown
    # Replace with space to avoid merging words
    clean_text = re.sub(r'[*#_`]', ' ', text)
    # Remove excess whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    try:
        tts = gTTS(text=clean_text, lang='te')
        tts.save('response.mp3')
        display(Audio('response.mp3', autoplay=True))
        time.sleep(len(text) / 12 + 1.5)
    except Exception as e:
        print(f"TTS Error: {e}")

# ==========================================
# PART 3: AGENT GRAPH NODES
# ==========================================

class AgentState(TypedDict):
    messages: List[BaseMessage]
    profile: Dict[str, str]
    next_step: str
    final_response: str

def info_extractor_node(state: AgentState):
    """Extracts profile info using the Local/Cloud LLM."""
    messages = state['messages']
    current_profile = state.get('profile', {}) or {}
    last_user_input = messages[-1].content if messages else ""
    
    if not last_user_input:
        return {"profile": current_profile}

    # Updated Prompt to capture more fields relevant to your JSON (caste, gender)
    system_prompt = f"""<start_of_turn>user
You are a data extraction assistant. Extract details from text into JSON.
Translate Telugu terms to English.
Fields: 'age', 'occupation', 'region', 'caste', 'income', 'gender', 'marital_status'.

Rules:
1. Return ONLY valid JSON.
2. If a value is missing, do not include the key.

Examples:
Input: "Naa vayasu 40 nenu raithu"
Output: {{ "age": "40", "occupation": "farmer" }}

Input: "Nenu SC kulam"
Output: {{ "caste": "SC" }}

Input: "Nenu pelli kani ammayini"
Output: {{ "gender": "female", "marital_status": "unmarried" }}

Task:
Current Profile: {json.dumps(current_profile)}
User Input: "{last_user_input}"
<end_of_turn>
<start_of_turn>model
"""
    try:
        response = llm_local.invoke(system_prompt)
        text = response if isinstance(response, str) else response.content
        clean_json = text.strip()
        
        if "{" in clean_json and "}" in clean_json:
            start = clean_json.find("{")
            end = clean_json.rfind("}") + 1
            clean_json = clean_json[start:end]
            
        extracted_data = json.loads(clean_json)
        updated_profile = current_profile.copy()
        for k, v in extracted_data.items():
            if v and str(v).lower() not in ["unknown", "none", "null"]:
                updated_profile[k] = v
        print(f"DEBUG (Extractor): {extracted_data}")
    except Exception as e:
        print(f"Extraction Error: {e}")
        updated_profile = current_profile

    return {"profile": updated_profile}

def decision_node(state: AgentState):
    profile = state.get('profile', {})
    # Mandatory fields before we allow a search
    mandatory_fields = ['occupation', 'age', 'region']
    missing = [field for field in mandatory_fields if field not in profile or not profile[field]]
    return {"next_step": "ask" if missing else "search"}

def question_generator_node(state: AgentState):
    profile = state.get('profile', {})
    missing_fields = [k for k in ['occupation', 'age', 'region'] if k not in profile]
    prompt = f"You are a helpful assistant. Current Profile: {profile}. Missing: {missing_fields}. Ask ONE polite question in TELUGU to get missing info. Just only give telugu response not telugu in english"
    response = llm_cloud.invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content}

def scheme_search_node(state: AgentState):
    profile = state['profile']
    # Create a query that targets the 'Eligibility' section of your JSON docs
    query_text = f"scheme for {profile.get('occupation', '')} {profile.get('age', '')} years old {profile.get('gender', '')} {profile.get('caste', '')}"
    print(f"DEBUG: RAG Query -> {query_text}")
    
    # Retrieve top 3 matches
    results = collection.query(query_texts=[query_text], n_results=3)
    
    docs = results['documents'][0] if results['documents'] else []
    
    if not docs:
        context = "No specific schemes found."
        print("DEBUG: No documents returned.")
    else:
        context = "\n\n".join([f"- {d}" for d in docs])
        print(f"DEBUG: Retrieved Context:\n{context}")
    
    prompt = f"""
    User Profile: {profile}
    
    Matched Schemes from Database:
    {context}
    
    Task: Explain the matched schemes in TELUGU. 
    Focus on WHY they are eligible (e.g., "Because you are a farmer...") and keep it short and just brief.
    If the database content says 'No specific schemes', apologize politely in Telugu that I am not able to currently get schemes for your profile.
    
    IMPORTANT: Provide the output in PLAIN TEXT. Do NOT use bold (**), italics, or markdown symbols (asterisks).
    """
    response = llm_cloud.invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content}

def output_node(state: AgentState):
    return {"messages": [AIMessage(content=state['final_response'])]}

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)
workflow.add_node("extractor", info_extractor_node)
workflow.add_node("decider", decision_node)
workflow.add_node("asker", question_generator_node)
workflow.add_node("searcher", scheme_search_node)
workflow.add_node("speaker", output_node)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "decider")
workflow.add_conditional_edges("decider", lambda x: x["next_step"], {"ask": "asker", "search": "searcher"})
workflow.add_edge("asker", "speaker")
workflow.add_edge("searcher", "speaker")
workflow.add_edge("speaker", END)

app = workflow.compile()

# ==========================================
# PART 4: MAIN EXECUTION LOOP
# ==========================================

def run_interactive_session():
    print("\n✅ SYSTEM READY! Initializing Conversation...")
    speak_audio("నమస్కారం. ప్రభుత్వ పథకాల గురించి నన్ను అడగండి.")
    
    messages = []
    user_profile = {} 
    
    while True:
        try:
            time.sleep(1)
            print("\n--- NEW TURN ---")
            
            # 1. Record
            audio_file = record_audio()
            
            # 2. Transcribe
            user_text = transcribe_audio(audio_file)
            
            if not user_text:
                print("Could not hear you. Please try again.")
                continue
            
            if "exit" in user_text.lower() or "bye" in user_text.lower():
                speak_audio("ధన్యవాదాలు!")
                break
                
            # 3. Agent Execution
            messages.append(HumanMessage(content=user_text))
            
            state = {
                "messages": messages, 
                "profile": user_profile, 
                "next_step": "decider"
            }
            
            result = app.invoke(state)
            
            # 4. Update State & Speak
            messages = result['messages']
            user_profile = result.get('profile', user_profile)
            
            last_response = messages[-1].content
            speak_audio(last_response)
            
        except KeyboardInterrupt:
            print("Stopped by User.")
            break
        except Exception as e:
            print(f"Runtime Error: {e}")
            break

if __name__ == "__main__":
    run_interactive_session()
