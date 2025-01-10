from langchain_aws import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings

def get_debrock_embedding_function():
  embeddings = BedrockEmbeddings(
     credentials_profile_name='default', region_name='us-east-1'
  )
  return embeddings

def get_ollama_embedding_function():
  embeddings = OllamaEmbeddings(model="nomic-embed-text")
  return embeddings
