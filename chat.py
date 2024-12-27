import re
import os
from glob import glob

from gtts import gTTS
from playsound import playsound

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
MODEL_NAME  = 'gpt-4o'
EMBEDDING_NAME = 'text-embedding-3-large'

COLLECTION_NAME = 'korean_history'
PERSIST_DIRECTORY= 'vector_store/korean_history_db'
category = ["사건", "인물"]
name = ["고대", "고려", "근대", "조선", "현대"]

# Split
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name=MODEL_NAME,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# db 연결
embedding_model = OpenAIEmbeddings(model=EMBEDDING_NAME)
vector_store = Chroma(collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)

def qna(query):
    retriever = vector_store.as_retriever(search_type="mmr")
    # Prompt Template 생성
    messages = [
            ("ai", """
        너는 한국사에 대해서 해박한 지식을 가진 역사전문가야야.
        내가 역사적 인물 또는 사건에 대해 말하면 그 인물과 사건을 이해가 쉽게, 흥미를 잃지 않게 쉬운용어로 풀어서 설명해주면 돼.

        문서에 없는 내용은 답변할 수 없어. 모르면 모른다고 답변해.

        인물의 이름 :
        시대 :
        인물에 대해 알고 싶은 것 :
    {context}"""),
            ("human", "{question}"),
        ]
    prompt_template = ChatPromptTemplate(messages)

    # 모델
    model = ChatOpenAI(model=MODEL_NAME)

    # output parser
    parser = StrOutputParser()

    # Chain 구성 retriever(관련 문서 조회) -> prompt_template(prompt 생성) model(정답) -> output parser
    chain = {"context":retriever, "question":RunnablePassthrough()} | prompt_template | model | parser
    return chain.invoke(query)

def TTS(query):
    text = qna(query)
    tts = gTTS(text=text, lang='ko')
    tts.save("tts.mp3")
    print(text)
    playsound('tts.mp3')

if __name__ == "__main__":
    while True:
        query = input("질문을 입력하세요(종료: exit) > ")
        if query == "exit":
            break
        elif len(query) > 5:
            break

    # query = "도시락 폭탄을 던진 사람은 누구인가요?"    
    TTS(query)