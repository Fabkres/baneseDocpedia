# Imagem base
FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar os arquivos para o container
COPY ./app /app

# Instalar dependências
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta 8000
EXPOSE 8000

# Configurar a execução do servidor FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
