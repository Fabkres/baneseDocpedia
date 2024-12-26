Observe esse código:


import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import tempfile
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Configurações do Streamlit
st.set_page_config(page_title="Converse com documentos 📚", page_icon="📚")
st.title("Converse com documentos 📚")

model_class = "openai"  # Apenas OpenAI será utilizado

# Função para configurar o modelo
def model_openai(model="gpt-4", temperature=0.1):
    llm = ChatOpenAI(model=model, temperature=temperature)
    return llm

# Função para configurar o retriever
def config_retriever(uploads):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 4}
    )

    return retriever

# Função para configurar a RAG Chain
def config_rag_chain(retriever):
    llm = model_openai()
    qa_prompt_template = """Você é um assistente virtual prestativo e está respondendo perguntas gerais.
    Use os seguintes pedaços de contexto recuperado para responder à pergunta.
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(qa_prompt_template)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

# Função para extrair as 5 principais palavras-chave
def extract_keywords(documents):
    llm = model_openai()
    keyword_prompt = """
    Extraia as 5 principais palavras-chave do seguinte texto:
    {text}
    """
    
    # Concatenando o conteúdo dos documentos para formar um único texto
    full_text = "\n".join([doc.page_content for doc in documents])
    
    # Certifique-se de que o texto seja passado corretamente no prompt
    prompt = keyword_prompt.format(text=full_text)
    
    # Chamando o modelo com o prompt de extração de palavras-chave
    result = llm([HumanMessage(content=prompt)])
    
    # A resposta do modelo será uma string contendo as palavras-chave
    return result.content.strip()

# Inicializando variáveis de sessão
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou o seu assistente virtual! Como posso ajudar você?"),
    ]
if "keywords" not in st.session_state:
    st.session_state.keywords = []

# Interface para upload e processamento
with st.sidebar:
    st.subheader("Envie seus documentos")
    uploads = st.file_uploader(
        label="Enviar arquivos", type=["pdf"],
        accept_multiple_files=True
    )
    if st.button("Processar documentos") and uploads:
        with st.spinner("Processando documentos..."):
            start_time = time.time()  # Medir tempo de processamento
            st.session_state.retriever = config_retriever(uploads)
            st.session_state.rag_chain = config_rag_chain(st.session_state.retriever)
            
            # Extração das palavras-chave
            documents = st.session_state.retriever.get_relevant_documents('')
            st.session_state.keywords = extract_keywords(documents)
            
            elapsed_time = time.time() - start_time
            st.success(f"Documentos processados com sucesso! Tempo: {elapsed_time:.2f} segundos")

# Exibindo palavras-chave extraídas
if st.session_state.keywords:
    st.subheader("Principais palavras-chave extraídas:")
    st.write(st.session_state.keywords)

# Exibindo histórico de chat
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Entrada de texto do usuário
user_query = st.chat_input("Digite sua mensagem aqui...")
if user_query and st.session_state.rag_chain:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        start_time = time.time()  # Medir tempo de resposta
        result = st.session_state.rag_chain.invoke(
            {"input": user_query, "chat_history": st.session_state.chat_history}
        )
        elapsed_time = time.time() - start_time

        resp = result['answer']
        st.write(resp)
        st.caption(f"Resposta gerada em {elapsed_time:.2f} segundos")
        st.session_state.chat_history.append(AIMessage(content=resp))

        # Exibindo fontes do contexto
        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'Página não especificada')

            ref = f":link: Fonte {idx + 1}: *{file} - p. {page}*"
            with st.popover(ref):
                st.caption(doc.page_content)



como faço para pegar partes do código que está relacionada com a plavra chave use o resumo para isso siga os seguintes passos:
1- Aproveite o código de split para pegar e verificar as partes do docuemnto que se relaciona com as palavras chaves selecionadas
tem um exemplo abaixo de como pegar os dados 

obsereve o codigo baaixo 
from keywords import extract_keywords
import re
from collections import defaultdict

def generate_mind_map_structure(text, num_keywords=5):
    keywords = extract_keywords(text, num_keywords)
    
    mind_map = {"root": "Main Topic", "branches": []}
    sentences = re.split(r'(?<=\.)\s+', text)
    
    keyword_map = defaultdict(list)
    for keyword in keywords:
        for sentence in sentences:
            if keyword in sentence.lower():
                keyword_map[keyword].append(sentence.strip())
    
    # Constrói os ramos do mapa mental
    for keyword, related_sentences in keyword_map.items():
        mind_map["branches"].append({
            "keyword": keyword,
            "details": related_sentences
        })
    
    return mind_map

def export_mind_map_to_json(mind_map, output_file="mind_map.json"):
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mind_map, f, ensure_ascii=False, indent=4)



preciso que as palvras chaves sejam enviadas para a função e use o mesmo processaemnto Mapeamento de palavras-chave para frases: Para cada palavra-chave extraída, o código verifica quais frases do texto contêm a palavra-chave (em minúsculo, para garantir que a busca seja insensível a maiúsculas/minúsculas). As frases que contêm a palavra-chave são armazenadas em um dicionário keyword_map, onde a chave é a palavra-chave e o valor é uma lista de frases que a contêm.
Construção da estrutura do mapa mental: A estrutura do mapa mental é criada com um "root" (tópico principal) e uma lista de "branches" (ramos), onde cada ramo corresponde a uma palavra-chave e suas frases relacionadas.
precisa de alguma explicação?

OBS. MOSTRE SEMPRE O CODIOG COMPLETO AQUI NO CHAT