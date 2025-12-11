import redis
import orjson
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import struct 
import time
from typing import List, Optional, Dict, Any
from redis.commands.search.field import TextField, VectorField
from concurrent.futures import ThreadPoolExecutor
import threading
import sys

# ================================
# FUNÇÃO AUXILIAR: Conversão de Vetor para Bytes 
# ================================
def convert_vector_to_bytes(vector: List[float]) -> bytes:
    """Converte uma lista de floats para o formato de bytes FLOAT32 necessário pelo Redis."""
    # O '<' indica little-endian, 'f' é float de 4 bytes (FLOAT32)
    return struct.pack('<{}f'.format(len(vector)), *vector)


# ================================
# CONFIGURAÇÃO
# ================================
CDXJ_PATH = "EAWP37.cdxj"   
REDIS_HOST = "localhost"
REDIS_PORT = 6380
INDEX_NAME = "cdxj_index"

# 🔄 CONFIGURAÇÕES DE VELOCIDADE
MAX_DOCUMENTS = 10000 # Limite o número de documentos VÁLIDOS a indexar
MAX_THREADS = 10     # Número de threads para processamento paralelo (ajuste conforme a sua CPU e limites da API)


# 🔄 MUDANÇA PARA HUGGING FACE INFERENCE API
HF_API_TOKEN = "token"
# Modelo de Embedding leve e eficiente para Inglês
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384 
HF_INFERENCE_API_URL = f"https://api-inference.huggingface.co/models/{EMBEDDING_MODEL}"

# Conexão Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT) 
try:
    r.ping()
    print("Conexão Redis estabelecida com sucesso.")
except redis.exceptions.ConnectionError as e:
    print(f"Erro ao conectar ao Redis: {e}")
    sys.exit(1) # Sai se não conseguir conectar

# Variáveis globais para rastreamento de progresso e bloqueio
indexed_count = 0
lock = threading.Lock()

# ================================
# CRIAR ÍNDICE VETORIAL NO REDIS
# ================================
def create_index():
    try:
        r.ft(INDEX_NAME).info()
        print("Índice já existe.")
        return
    except:
        print("A criar índice vetorial...")

        schema = [
            # Campos textuais
            TextField("url"),
            TextField("mime"),
            TextField("status"),
            TextField("content"), 

            # Campo vetorial (FLOAT32)
            VectorField(
                "embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM, 
                    "DISTANCE_METRIC": "COSINE"
                }
            )
        ]

        try:
            r.ft(INDEX_NAME).create_index(schema)
            print("Índice criado com sucesso.")
        except Exception as e:
            print(f"Erro ao criar índice: {e}") 
            sys.exit(1) # Sai se não conseguir criar índice

# ================================
# FUNÇÃO: gerar embedding (HUGGING FACE)
# ================================
def generate_embedding(text: str) -> Optional[List[float]]:
    """Gera um embedding usando a Hugging Face Inference API."""
    if not HF_API_TOKEN or HF_API_TOKEN == "SUA_CHAVE_API_HUGGINGFACE_AQUI":
        return None

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}
    
    # Implementação de backoff exponencial simples para lidar com rate limits
    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(HF_INFERENCE_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status() 
            
            embeddings = response.json()
            if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
                return embeddings[0]
            
            # Avisos sobre modelo a carregar ou erro na Inference API
            if isinstance(embeddings, dict) and 'error' in embeddings:
                # print(f"Aviso HF: {embeddings['error']}")
                pass # Apenas tentamos novamente
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(5) 
            else:
                return None
            
        except requests.exceptions.RequestException as e:
            # print(f"Erro de conexão/API: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt) 
            else:
                return None
        except Exception:
            return None
    return None


# ================================
# FUNÇÃO: buscar conteúdo arquivado
# ================================
def fetch_archived_content(url: str, timestamp: str) -> Optional[str]:
    """
    Constrói a URL do Arquivo.pt e faz a requisição para obter o HTML, 
    extraindo apenas o texto limpo.
    """
    
    ARCHIVE_BASE_URL = "https://arquivo.pt/wayback/"
    archive_url = f"{ARCHIVE_BASE_URL}{timestamp}/{url}"
    
    try:
        response = requests.get(archive_url, timeout=10)
        response.raise_for_status() 

        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'noscript', 'meta']):
            script_or_style.decompose()

        text = soup.get_text(separator=' ', strip=True)
        
        # O modelo MiniLM é eficiente com textos mais curtos
        return text[:3000]
        
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None


# ================================
# PROCESSAMENTO DE LINHA (PARALELO)
# ================================
def process_line_parallel(line: str, pbar: tqdm):
    """Processa uma única linha do CDXJ, gera o embedding e guarda no Redis."""
    global indexed_count

    # 1. Verificar o limite de documentos
    with lock:
        if indexed_count >= MAX_DOCUMENTS:
            return

    # 2. Parsing e Filtragem
    parts = line.split(" ", 2)
    if len(parts) < 3:
        return
    
    timestamp = parts[1] 
    json_part = parts[2].strip()

    try:
        record = orjson.loads(json_part)
    except:
        return

    if record.get("mime") != "text/html":
        return

    url = record.get("url", "")
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"

    mime = record.get("mime", "")
    status = str(record.get("status", ""))
    
    # 3. Fase 1: Obter Conteúdo
    content_text = fetch_archived_content(url, timestamp)
    
    if not content_text or len(content_text.strip()) < 50: 
        return 

    # 4. Fase 2: Geração de Embedding
    text_for_embedding = content_text 
    embedding = generate_embedding(text_for_embedding)
    
    if embedding is None:
        return 

    # 5. Fase 3: Armazenamento no Redis
    try:
        vector_bytes = convert_vector_to_bytes(embedding)
        key = f"cdx:{record['digest']}"

        r.hset(key, mapping={
            "url": url,
            "mime": mime,
            "status": status,
            "content": content_text, 
            "embedding": vector_bytes
        })
        
        # 6. Atualizar contador e barra de progresso
        with lock:
            indexed_count += 1
            pbar.update(1)
            if indexed_count >= MAX_DOCUMENTS:
                pbar.close() # Fecha a barra quando o limite é atingido
                print(f"\nLimite de {MAX_DOCUMENTS} documentos VÁLIDOS alcançado.")
                # Não é necessário chamar sys.exit, pois o loop principal vai terminar naturalmente.

    except Exception:
        # Erro de armazenamento no Redis (raro, mas possível)
        pass


# ================================
# LOOP PRINCIPAL (Paralelo)
# ================================
def process_cdxj_parallel():
    print("A processar ficheiro CDXJ em paralelo...")
    
    if not HF_API_TOKEN or HF_API_TOKEN == "SUA_CHAVE_API_HUGGINGFACE_AQUI":
        print("ERRO: O token da Hugging Face não foi configurado.")
        return

    # Usamos ThreadPoolExecutor para paralelizar chamadas HTTP e API
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        with open(CDXJ_PATH, "r", encoding="utf-8") as f:
            # Não podemos usar `tqdm(f)` diretamente se quisermos parar o loop
            # e a barra deve contar apenas os documentos indexados com sucesso.
            
            # A barra de progresso será atualizada manualmente dentro do worker
            pbar = tqdm(total=MAX_DOCUMENTS, desc="A indexar (Válidos)")
            
            # Disparamos as tarefas de processamento de linhas
            for line in f:
                if indexed_count >= MAX_DOCUMENTS:
                    break
                executor.submit(process_line_parallel, line, pbar)

    print("Processamento do ficheiro concluído (pode não ter atingido o final do ficheiro CDXJ).")


# ================================
# FASE 4: Busca no Redis
# (Mantida a mesma lógica)
# ================================
def search_index(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Gera o embedding da query e busca os K documentos mais similares no Redis."""
    print(f"\nA buscar os {k} documentos mais relevantes para: '{query}'")

    if not HF_API_TOKEN or HF_API_TOKEN == "SUA_CHAVE_API_HUGGINGFACE_AQUI":
        print("ERRO: O token da Hugging Face não está configurado. A busca falhou.")
        return []

    # 1. Gerar embedding da query usando o mesmo modelo HF
    query_embedding = generate_embedding(query)
    
    if query_embedding is None:
        return []

    # 2. Converter o vetor da query para bytes
    query_vector_bytes = convert_vector_to_bytes(query_embedding)

    # 3. Construir a query de busca por similaridade vetorial (KNN)
    search_query = (
        f'(*)=>[KNN {k} @embedding $vector AS score]'
        f' RETURN {k} url content score'
    )

    # 4. Preparar e executar o comando FT.SEARCH
    try:
        results = r.ft(INDEX_NAME).search(
            redis.commands.search.query.Query(search_query)
            .sort_by('score') 
            .dialect(2), 
            query_params={'vector': query_vector_bytes}
        )
    except Exception as e:
        print(f"Erro ao executar busca no Redis: {e}")
        return []

    # 5. Processar e formatar os resultados
    search_results = []
    for doc in results.docs:
        search_results.append({
            'url': doc.url,
            'content': doc.content,
            'score': float(doc.score)
        })
        
    return search_results

# ================================
# EXECUÇÃO PRINCIPAL
# ================================
if __name__ == "__main__":
    create_index()
    time.sleep(1) 
    
    # 🚀 Usamos a função paralela para indexação
    process_cdxj_parallel() 
    print("✔️ Indexação Concluída!")
    
    # Exemplo da Fase 4: Busca 
    TEST_QUERY = "Who was the winner of the 2021 presidential election in Portugal?"
    results = search_index(TEST_QUERY, k=5)
    
    print("\n--- Resultados da Busca (Fase 4) ---")
    for i, res in enumerate(results):
        print(f"#{i+1}: Score: {res['score']:.4f} | URL: {res['url']}")
        print(f"   Conteúdo (Excerto): {res['content'][:150]}...\n")
    
    print("------------------------------------------------------------------")