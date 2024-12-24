# split 설정
chunk_size = 1000
chunk_overlap = 0

# model 설정
embedding_name = 'text-embedding-3-large'
model_name = 'gpt-4o-mini'

# Chroma DB 설정
collection_name_person = "korean_history"
persist_directory_person = "vector_store/korean_history_db"

collection_name_case = "korean_history2"
persist_directory_case = "vector_store/korean_history2_db"
