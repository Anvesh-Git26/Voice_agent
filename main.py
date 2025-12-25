import os
import sys
import re
import json
import time
import base64
import chromadb
import speech_recognition as sr
from typing import TypedDict, List, Dict
from gtts import gTTS
from IPython.display import Audio, display, Javascript
from google.colab import output

# LangChain / Graph Imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq  # <--- NEW IMPORT
from chromadb.utils import embedding_functions

# ==========================================
# PART 1: SETUP & CONFIGURATION (GROQ)
# ==========================================

mandatory_fields = ['occupation', 'age', 'region']

# 🔴 PASTE YOUR GROQ API KEY HERE
GROQ_API_KEY = "Enter_your_API" 

if GROQ_API_KEY.startswith("gsk_..."):
    print("⚠️ WARNING: Please replace 'gsk_...' with your actual Groq API Key!")

# Initialize Groq Model
# Llama-3-70b is very smart and handles Telugu reasonably well
print("⚡ Initializing Groq Model (Llama-3)...")
try:
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile", # Or "llama3-70b-8192"
        groq_api_key=GROQ_API_KEY
    )
    print("✅ Groq Model Ready!")
except Exception as e:
    print(f"❌ Groq Init Failed: {e}")
    llm = None

# Initialize Database
print("💾 Initializing Knowledge Base...")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection("telangana_schemes") # Reset DB
except:
    pass
collection = client.get_or_create_collection(name="telangana_schemes", embedding_function=emb_fn)

# Load JSON Data
if os.path.exists("schemes.json"):
    with open("schemes.json", 'r', encoding='utf-8') as f:
        schemes_data = json.load(f)
    
    documents = []
    ids = []
    
    for i, item in enumerate(schemes_data):
        text = f"{item['scheme_name_te']}: {item['description_te']} Benefits: {item['benefits']}"
        documents.append(text)
        ids.append(str(i))
    
    if documents:
        collection.add(documents=documents, ids=ids)
        print(f"✅ Loaded {len(documents)} schemes.")

# ==========================================
# PART 2: AUDIO FUNCTIONS
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
  btn.style.padding = "10px"
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
    display(Javascript(RECORD_JS))
    print("👇 Click RED button to record...")
    s = output.eval_js('record(0)')
    b = base64.b64decode(s.split(',')[1])
    with open('temp_audio.webm', 'wb') as f: f.write(b)
    os.system(f"ffmpeg -y -i temp_audio.webm -ac 1 -ar 16000 {filename} -loglevel error")
    return filename

def transcribe_audio(filename):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language="te-IN")
        print(f"🎤 You: {text}")
        return text
    except:
        return ""

def speak_audio(text):
    print(f"🤖 Agent: {text}")
    clean_text = re.sub(r'[*#_`]', ' ', text).strip()
    try:
        tts = gTTS(text=clean_text, lang='te')
        tts.save('response.mp3')
        display(Audio('response.mp3', autoplay=True))
        time.sleep(len(text)/10 + 1)
    except:
        pass

# ==========================================
# PART 3: AGENT LOGIC (GROQ OPTIMIZED)
# ==========================================

# ==========================================
# PART 3: AGENT LOGIC (WITH AUTO-EXIT)
# ==========================================

class AgentState(TypedDict):
    messages: List[BaseMessage]
    profile: Dict[str, str]
    next_step: str
    final_response: str
    is_complete: bool  # <--- NEW FLAG: Tells the loop to stop

def info_extractor(state: AgentState):
    msg = state['messages'][-1].content
    profile = state.get('profile', {})
    
    # Simple extraction
    sys_msg = f"""
    Extract user details (age, occupation). Return JSON ONLY.
    Input: "{msg}"
    Current Profile: {json.dumps(profile)}
    """
    try:
        res = llm.invoke(sys_msg).content
        if "{" in res:
            res = res[res.find("{"):res.rfind("}")+1]
        new_data = json.loads(res)
        profile.update({k:v for k,v in new_data.items() if v})
    except:
        pass
    
    return {"profile": profile}

def router(state: AgentState):
    profile = state['profile']
    # If we have Occupation OR Age, we assume we can search
    if 'occupation' in profile or 'age' in profile:
        return {"next_step": "search"}
    return {"next_step": "ask"}

def asker(state: AgentState):
    return {"final_response": "మీ వృత్తి మరియు వయస్సు చెప్పండి? (What is your occupation and age?)", "is_complete": False}

def searcher(state: AgentState):
    profile = state['profile']
    
    # 1. Search
    results = collection.query(query_texts=["scheme"], n_results=3)
    docs = [d for d in results['documents'][0]]
    context = "\n".join(docs)
    
    # 2. Generate Answer
    prompt = f"""
    User: {profile}
    Schemes: {context}
    Pick the best scheme for this user and explain in TELUGU.
    """
    res = llm.invoke(prompt).content
    
    # 3. SET COMPLETION FLAG TO TRUE
    return {"final_response": res, "is_complete": True}

def speaker(state: AgentState):
    return {"messages": [AIMessage(content=state['final_response'])]}

# Graph Construction
workflow = StateGraph(AgentState)
workflow.add_node("extract", info_extractor)
workflow.add_node("decide", router)
workflow.add_node("ask", asker)
workflow.add_node("search", searcher)
workflow.add_node("speak", speaker)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "decide")
workflow.add_conditional_edges("decide", lambda x: x['next_step'], {"ask": "ask", "search": "search"})
workflow.add_edge("ask", "speak")
workflow.add_edge("search", "speak")
workflow.add_edge("speak", END)

app = workflow.compile()

# ==========================================
# PART 4: RUN LOOP (MODIFIED)
# ==========================================

def run_chat():
    print("✅ System Ready! Speak now...")
    speak_audio("నమస్కారం! ప్రభుత్వ పథకాల కోసం నన్ను అడగండి.")
    
    msgs = []
    profile = {}
    
    while True:
        try:
            # 1. Listen
            audio = record_audio()
            text = transcribe_audio(audio)
            
            if not text:
                print("No voice detected. Trying again...")
                continue
                
            msgs.append(HumanMessage(content=text))
            
            # 2. Think & Act
            res = app.invoke({"messages": msgs, "profile": profile})
            profile = res.get('profile', {})
            
            # 3. Speak Response
            response_text = res['messages'][-1].content
            speak_audio(response_text)
            
            # 4. CHECK TERMINATION
            # If the agent marked the task as complete (found a scheme), we exit.
            if res.get('is_complete', False):
                print("\n✅ Task Completed. Terminating Session.")
                break
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    run_chat()
