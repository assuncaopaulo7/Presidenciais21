import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.query import Query
import orjson
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import struct 
import time
import os
import sys
import threading
import queue
import gc
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai 
import torch
import warnings

# Ignorar avisos de SSL
warnings.filterwarnings("ignore")

# 🛡️ OTIMIZAÇÃO DE CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ==============================================================================
# ⚙️ CONFIGURAÇÕES CRÍTICAS
# ==============================================================================
REDIS_HOST = "localhost"
REDIS_PORT = 6380
INDEX_NAME = "presidenciais_idx"

# ⚠️ CONFIRMA O NOME DO FICHEIRO AQUI (Se é EAWP9 ou EAWP37)
CDXJ_PATH = "../EAWP9.cdxj" 

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GOOGLE_API_KEY = "AIzaSyAgXghBCfErZJZppCqMyE9YsmfU5B_OfQQ"

# 🚀 ACELERADOR (FAST FORWARD)
# Se estiveres a 0 aceites há muito tempo, aumenta isto.
# Pula as primeiras N linhas instantaneamente (geralmente lixo técnico).
SKIP_INITIAL_LINES = 50000 

MAX_DOCUMENTS = 3000     # Quantos documentos BONS queremos encontrar
DOWNLOAD_THREADS = 6     # Aumentei um pouco para compensar a latência
QUEUE_SIZE = 15          
TEXT_LIMIT = 3000        

# ==============================================================================
# 🛠️ INICIALIZAÇÃO
# ==============================================================================

print(f"📥 A carregar IA (Device: {'cuda' if torch.cuda.is_available() else 'cpu'})...")
try:
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model = SentenceTransformer(MODEL_NAME, device=device)
except ImportError:
    print("❌ Falta instalar libs.")
    sys.exit(1)

if "INSIRA" not in GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_llm = genai.GenerativeModel('gemini-pro')
else:
    model_llm = None

try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    r.ping()
except:
    print("❌ Redis desligado.")
    sys.exit(1)

# ==============================================================================
# 🧩 FUNÇÕES
# ==============================================================================

def convert_vector_to_bytes(vector):
    return struct.pack('<{}f'.format(len(vector)), *vector)

def create_index():
    try:
        r.ft(INDEX_NAME).info()
    except:
        print("🔨 Criando índice...")
        schema = [
            TextField("url"),
            TextField("content"), 
            VectorField("embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"})
        ]
        r.ft(INDEX_NAME).create_index(schema)

class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.scanned = 0
        self.skipped_start = 0
        self.ignored = 0
        self.accepted = 0

stats = Stats()

def fetch_content(url, timestamp):
    try:
        archive_url = f"https://arquivo.pt/wayback/{timestamp}/{url}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
        
        # Timeout curto para falhar rápido em sites mortos
        resp = requests.get(archive_url, headers=headers, timeout=5, verify=False)
        if resp.status_code != 200: return None

        soup = BeautifulSoup(resp.content, 'html.parser')
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'svg', 'noscript']):
            tag.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        
        if len(text) < 200: return None # Ignora textos muito curtos
            
        low = text.lower()
        
        # ⚠️ KEYWORDS GENÉRICAS (Funcionam para 2016 E 2021)
        # Removi nomes específicos (Ventura, Mayan) para não bloquear dados de 2016
        keywords = [
            "portugal", "eleição", "election", "presiden", "candidat", 
            "voto", "vote", "urna", "campanha", "debate", "partido",
            "marcelo", "sampaio", "nóvoa", "matias", "belém" # Nomes comuns a várias eleições
        ]
        
        if not any(k in low for k in keywords):
            return None
            
        return text[:TEXT_LIMIT]
    except:
        return None

# ==============================================================================
# 🏗️ PIPELINE DE ALTA VELOCIDADE
# ==============================================================================

def producer_task(cdxj_file, data_queue, stop_event):
    submission_semaphore = threading.Semaphore(20)
    
    print(f"📂 A ler: {cdxj_file}")
    print(f"⏩ MODO FAST FORWARD: A saltar as primeiras {SKIP_INITIAL_LINES} linhas...")
    
    with open(cdxj_file, "r", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as executor:
            
            for line in f:
                if stop_event.is_set(): break
                
                # 🚀 LÓGICA DE SALTO RÁPIDO
                if stats.skipped_start < SKIP_INITIAL_LINES:
                    stats.skipped_start += 1
                    if stats.skipped_start % 10000 == 0:
                        print(f"⏩ A saltar lixo... {stats.skipped_start}/{SKIP_INITIAL_LINES}", end='\r')
                    continue

                with stats.lock:
                    stats.scanned += 1
                    curr_scan = stats.scanned
                    curr_acc = stats.accepted
                
                if curr_scan % 100 == 0:
                    print(f"👀 A Analisar: {curr_scan} | 🗑️ Ignorados: {stats.ignored} | ✅ ACEITES: {curr_acc}...", end='\r')

                parts = line.split(" ", 2)
                if len(parts) < 3: continue
                
                timestamp, json_part = parts[1], parts[2].strip()
                try: rec = orjson.loads(json_part)
                except: continue

                if rec.get("mime") != "text/html": 
                    with stats.lock: stats.ignored += 1
                    continue
                
                url = rec.get("url", "")
                if not url.startswith("http"): url = "http://" + url
                
                doc_key = f"doc:{timestamp}:{rec.get('digest', 'nodigest')}"

                if r.exists(doc_key):
                    data_queue.put(("EXISTING", None, None))
                    continue

                submission_semaphore.acquire()

                def download_job(u, t, k, sem):
                    try:
                        if stop_event.is_set(): return
                        result = fetch_content(u, t)
                        
                        if result:
                            # SUCESSO
                            print(f"\n✅ ENCONTRADO: {u[:60]}...")
                            with stats.lock: stats.accepted += 1
                            data_queue.put(("NEW", k, {"url": u, "content": result}))
                        else:
                            with stats.lock: stats.ignored += 1
                    finally:
                        sem.release()
                
                executor.submit(download_job, url, timestamp, doc_key, submission_semaphore)
                time.sleep(0.002)

    data_queue.put("DONE")

def consumer_task():
    data_queue = queue.Queue(maxsize=QUEUE_SIZE)
    stop_event = threading.Event()
    
    prod_thread = threading.Thread(target=producer_task, args=(CDXJ_PATH, data_queue, stop_event))
    prod_thread.start()

    print(f"🚀 Início (Queue: {QUEUE_SIZE})...")
    pbar = tqdm(total=MAX_DOCUMENTS, unit="docs")
    indexed = 0

    try:
        while True:
            item = data_queue.get()
            if item == "DONE": break
            
            status, key, data = item
            
            if status == "EXISTING":
                pbar.update(1)
                indexed += 1
            elif status == "NEW":
                embedding = local_model.encode(data["content"])
                r.hset(key, mapping={
                    "url": data["url"],
                    "content": data["content"],
                    "embedding": convert_vector_to_bytes(embedding)
                })
                pbar.update(1)
                indexed += 1
                del embedding
                del data
            
            data_queue.task_done()
            if indexed >= MAX_DOCUMENTS:
                print("\n🎯 Meta atingida!")
                stop_event.set()
                break
            if indexed % 50 == 0: gc.collect()

    except KeyboardInterrupt:
        print("\n🛑 A parar...")
        stop_event.set()
    
    pbar.close()
    prod_thread.join(timeout=2)

def rag_pipeline(query):
    print(f"\n🔍 A pesquisar: '{query}'...")
    q_vec = local_model.encode(query).tolist()
    redis_q = Query(f'(*)=>[KNN 5 @embedding $vector AS score]').sort_by('score').return_fields('content', 'url').dialect(2)
    try: res = r.ft(INDEX_NAME).search(redis_q, query_params={"vector": convert_vector_to_bytes(q_vec)})
    except: return

    if not res.docs: print("❌ Nada encontrado."); return

    ctx = "\n".join([f"Fonte ({d.url}): {d.content[:400]}" for d in res.docs])
    if model_llm:
        print("🤖 Gemini a responder...")
        try:
            prompt = f"Contexto:\n{ctx}\nPergunta: {query}\nResposta em Português:"
            ans = model_llm.generate_content(prompt)
            print("-" * 40 + f"\n{ans.text}\n" + "-" * 40)
        except: pass
    else:
        print(ctx)

if __name__ == "__main__":
    create_index()
    if os.path.exists(CDXJ_PATH):
        if input("Processar dados (s/n)? ").lower() == 's':
            consumer_task()
    
    while True:
        q = input("\nPergunta (sair): ")
        if q in ['sair', 'exit']: break
        rag_pipeline(q)