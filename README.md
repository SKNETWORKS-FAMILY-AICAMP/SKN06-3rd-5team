# SKN06-3rd-5Team

3차 단위 프로젝트 - 안형진, 조해원, 전수연, 임연경, 박미현


## 갑진 파이브 ✋🏻
### 팀원
| 안형진 | 조해원 | 전수연 | 
|:----------:|:----------:|:----------:|
|<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-3rd-5team/blob/main/%EC%82%AC%EC%A7%84/%ED%98%95%EC%A7%84.png" alt="image" width="200" height="250"/> |<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-3rd-5team/blob/main/%EC%82%AC%EC%A7%84/%ED%95%B4%EC%9B%90.png" alt="image" width="200" height="250"/>|<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-3rd-5team/blob/main/%EC%82%AC%EC%A7%84/%EC%88%98%EC%97%B0.jpg" alt="image" width="350" height="250"/>|
| 나 정말 **"개로왕~🤴🏻"** | 코드 잘 좀 **"해종~🤴🏻"** | 우리 오늘 밤  **"세종.🤴🏻"** |

</br>

| 임연경 | 박미현 |
|:----------:|:----------:|
|<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-3rd-5team/blob/main/%EC%82%AC%EC%A7%84/%EC%97%B0%EA%B2%BD.png" alt="image" width="200" height="250"/>|<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-3rd-5team/blob/main/%EC%82%AC%EC%A7%84/%EB%AF%B8%ED%98%84.png" alt="image" width="200" height="250"/>|
| **"우왕~🤴🏻"** 된당~ 흐히히 | 우리 모델 좋은거 **"인종~?🤴🏻"** |

# 🤴🏻👑 역사적 인물이나 사건에 대해 알려주는 교육 챗봇 🤖

### ✔ 개발 기간                                                  
2024.12.24 ~ 2024.12.26 (총 2일)

## 프로젝트 개요
✔ **목표**
   > RAG기반 질의응답을 이용한 역사적 인물이나 사건에 대해 질문하면 대답하는 챗봇 개발
   >
   1. 역사에 대한 관심 증대
      > 학생들과 일반 사용자가 역사적 인물과 사건에 대해 쉽고 흥미롭게 접근할 수 있도록 돕는 것을 목표
      >
      > 이를 통해 역사가 지닌 중요성과 재미를 발견할 수 있는 기회를 제공
      
  2. 지식의 대중화
      > 한국사를 포함한 다양한 역사적 주제를 간단하고 명료하게 설명하여, 누구나 쉽게 이해하고 학습할 수 있는 환경을 조성
      >
      > 이는 특히 역사를 잘 모르는 사용자들에게 유용한 정보를 제공하는 데 초점을 맞춤
      
  
  3. 교육적 활용 가능성
      > 교육 현장에서 학생들의 학습을 지원하는 도구로 활용
      >
      > 예를 들어, 역사적 인물의 업적, 사건의 배경과 결과 등을 질문하고 답변하는 과정을 통해 학습 효율을 높이고 호기심 자극
   
    
  4. 문화유산 보존
      > 역사적 자료와 정보를 디지털화하여 미래 세대가 지속적으로 사용할 수 있는 자산으로 보존
      > 
      > 특히, 한국사를 비롯한 전 세계의 다양한 역사적 이야기를 기록하고 전달하는 데 기여
     
      
  5. 맞춤형 정보 제공
       > 사용자의 질문에 따라 역사적 정보를 정확하고 빠르게 제공함으로써, 개개인의 필요와 관심사에 적합한 정보를 제공
       
       
✔ **주요 작업**
   1. 데이터 수집

        > 프로젝트 주제에 맞는 데이터를 수집하는데, 정보에 허구가 없고 신뢰도가 높은 자료를 인용
        > 
        > 시각화 자료를 제외하고, 텍스트 위주의 자료만 받아서 사용
        > 
        > 국사편찬위원회의 자료 사용 - http://contents.history.go.kr/front/kc/main.do#self
        > 
       
   2. 전처리
      
      > 줄바꿈, 한자어 포함된 것을 주의하여 데이터 가공
      > 

   3. RAG 기반 벡터 데이터베이스 저장
      >  데이터 벡터화, 저장소 구성, 메타데이터 추가를 통해 검색을 할 수 있는 데이터베이스를 구축
      > 
   
   4. LLM 연동
      
      > ![image](https://github.com/user-attachments/assets/85efb451-f6c8-4688-892f-28b6688cde6e)






## 프로젝트 진행 과정
**❗️ 화면 구성은 생략하고 RAG를 위한 백엔드 시스템 구축에 집중 ❗️**
</br>
### 1. 데이터 수집 및 전처리

#### 1.1 데이터 수집
  - **목표**
    > 프로젝트 주제에 맞는 데이터를 수집
    > 
  - **방법**
    > 공개 데이터셋 다운로드(우리역사넷 한국사연대기 PDF자료)
    >
    > 추가로 request를 이용하여 크롤링 진행 후 저장
    > 

#### 1.2 데이터 전처리
  - **작업**
    > 한자 제거, 문장 말머리 번호 제거, 중복 제거, 줄바꿈 제거 등
    >
  - **코드 (Python)**
    ```
    full_text = [doc.page_content for doc in load_docs] 
    full_text = ''.join(full_text)
    full_text = re.sub(r"관련사료", "", full_text)
    full_text = re.sub(r"\([一-龥]+\)", "", full_text)
    full_text = re.sub(r"\n", "", full_text)
    ```



### 2. 벡터화 및 벡터 데이터베이스 저장

#### 2.1 텍스트 임베딩 생성
  - **목표**
    > 텍스트 데이터를 벡터 형태로 변환
    >  

  - **코드**
    ```
      # Split
      splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
          model_name=MODEL_NAME,
          chunk_size=CHUNK_SIZE,
          chunk_overlap=CHUNK_OVERLAP,
      )
      
      # Embedding 모델 초기화
      embedding_model = OpenAIEmbeddings(model=EMBEDDING_NAME)
    ```



#### 2.2 벡터 데이터베이스 저장
  - **목표**
    > 생성된 벡터를 벡터 데이터베이스에 저장
    > 

  - **코드**
    ```
    # Vector store 연결
    vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model
    )
    ```



### 3. RAG 기반 질의 응답 시스템 구현

#### 3.1 LLM 연동 및 RAG Chain 구성
  - **목표**
    > LLM과 벡터 데이터베이스를 연동하여 질의 응답 시스템 구축
    > 
  - **도구**
    > OpenAI API, LangChain 등
    > 

  - **코드**
    ```
    # LLM 연동
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    MODEL_NAME = 'gpt-4o'

    # LLM 모델 초기화
    model = ChatOpenAI(model=MODEL_NAME)

    # Prompt 템플릿 생성
    messages = [
       ("ai", """
        너는 한국사에 대해서 해박한 지식을 가진 역사전문가야.
        내가 역사적 인물 또는 사건에 대해 말하면 그 인물과 사건을 이해가 쉽게, 흥미를 잃지 않게 쉬운용어로 풀어서 설명해주면 돼.

        문서에 없는 내용은 답변할 수 없어. 모르면 모른다고 답변해.

        인물의 이름 :
        시대 :
        인물에 대해 알고 싶은 것 :
    {context}"""),
        ("human", "{question}"),
       ]
    prompt_template = ChatPromptTemplate(messages=messages)

    # Output parser
    parser = StrOutputParser()

    # Langchain 구성
    from langchain_core.runnables import RunnablePassthrough

    # Vector 데이터베이스에서 검색 수행
    retriever = vector_store.as_retriever(search_type="mmr")

    # Chain 구성 retriever(관련 문서 조회) -> prompt_template(prompt 생성) model(정답) -> output parser
    chain = {"context":retriever, "question":RunnablePassthrough()} | prompt_template | model | parser
    ```



### 4. 성능 테스트 및 개선

#### 4.1 평가 작업
   1. PDF 문서 로드 및 전처리
      > PDF 파일을 로드하고 필요한 텍스트를 추출하여 전처리.
      >
      > 텍스트를 문서 형식(Document)으로 변환.

   2. 텍스트 분할
      > 추출된 텍스트를 RecursiveCharacterTextSplitter를 사용하여 적절한 크기로 분할.
      > 
   3. 평가용 데이터 생성
      > 전처리된 문서에서 샘플 컨텍스트를 무작위로 선택.
      >
      > ChatOpenAI 모델을 활용하여 질문-정답 쌍을 생성.
   
   4. RAG(Recovery-Augmented Generation) 체인 생성
      > 검색 모델(retriever)과 LLM을 연결하여 사용자 입력에 대한 답변과 검색된 컨텍스트를 생성.
      > 
   5. 모델 평가 메트릭 설정
      > LLMContextRecall, LLMContextPrecisionWithReference, Faithfulness, AnswerRelevancy 등의 메트릭 설정.
      > 
   6. 평가 수행
      > 생성된 질문-정답 쌍을 사용하여 모델의 응답을 평가.
      > 
#### 4.2 평가 과정
   1. PDF 문서 로드 및 전처리
      ```
      # Split
      splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      model_name=MODEL_NAME,
      chunk_size=CHUNK_SIZE,
      chunk_overlap=CHUNK_OVERLAP,
      )

      document_list = []
      for j in category:
         for i in name:
            file = glob(f"./pdf/{j}/{i}/*.pdf")

            for path in file:
               loader = PyMuPDFLoader(path)
               load_docs = loader.load()
               full_text = [doc.page_content for doc in load_docs]
               full_text = ''.join(full_text)
               full_text = re.sub(r"관련사료", "", full_text)
               full_text = re.sub(r"\([一-龥]+\)", "", full_text)
               full_text = re.sub(r"\n", "", full_text)

               docs = splitter.split_text(full_text)
               metadata = {
                  "id": "_".join(os.path.splitext(os.path.basename(path))[0].split('_')[:2]),
                  "title": os.path.splitext(os.path.basename(path))[0].split('_')[-1],
               }
   
               for doc in docs:
                   _doc = Document(metadata=metadata, page_content=doc)
                   document_list.append(_doc)
      ```
         
   2. 평가 데이터 생성
      ```
      class EvalDatasetSchema(BaseModel):
          user_input: str = Field(..., title="질문(question)")
          qa_context: list[str] = Field(..., title="질문-답변 쌍을 만들 때 참조한 context.")
          reference: str = Field(..., title="질문의 정답(ground truth)")
      
      parser = JsonOutputParser(pydantic_object=EvalDatasetSchema)
      eval_model = ChatOpenAI(model="gpt-4o")
      prompt_template = PromptTemplate.from_template(
          template=dedent("""
              당신은 RAG 평가를 위해 질문과 정답 쌍을 생성하는 인공지능 비서입니다.
              다음 [Context] 에 문서가 주어지면 해당 문서를 기반으로 {num_questions}개 질문-정답 쌍을 생성하세요. 
              질문과 정답을 생성한 후 아래의 출력 형식 GUIDE 에 맞게 생성합니다.
              ...
          """),
          partial_variables={"format_instructions": parser.get_format_instructions()}
      )
      
      eval_dataset_generator = prompt_template | eval_model | parser
      eval_data_list = []
      num_questions = 5
      for context in eval_context_list:
          _eval_data_list = eval_dataset_generator.invoke({"context": context, "num_questions": num_questions})
          eval_data_list.extend(_eval_data_list)
      ```
   3. RAG 체인 생성
      ```
      rag_chain = (
         RunnablePassthrough()
         | {"context": retriever, "question": RunnablePassthrough()}
         | {
               "source_context": itemgetter("context") | RunnableLambda(str_from_documents),
               "llm_answer": {
                     "context": RunnableLambda(format_docs),
                     "question": itemgetter("question"),
                 } | prompt_template | model | StrOutputParser(),
             }
         )
         ```
   4. 모델 평가
         ```
         from ragas.metrics import (
             LLMContextRecall, Faithfulness, LLMContextPrecisionWithReference, AnswerRelevancy
         )
         from ragas.llms import LangchainLLMWrapper
         from ragas.embeddings import LangchainEmbeddingsWrapper
         
         metrics = [
             LLMContextRecall(llm=eval_llm),
             LLMContextPrecisionWithReference(llm=eval_llm),
             Faithfulness(llm=eval_llm),
             AnswerRelevancy(llm=eval_llm, embeddings=eval_embedding),
         ]
         
         result = evaluate(dataset=eval_dataset, metrics=metrics)
         ```
### 평가 결과
   ![image](https://github.com/user-attachments/assets/6e38430a-7864-4ac0-9907-6bb0e9eaf34b)

   1. **context_recall: 0.7000** </br>
      해석: 점수가 70%로 중간 정도 성능을 보였습니다. 모델이 컨텍스트 내 정보를 어느 정도 활용하고 있지만, 일부 중요한 정보를 놓치고 있을 가능성이 있습니다.</br>

   2. **llm_context_precision_with_reference: 0.8833** </br>
      해석: 점수가 88%로 높은 수준을 보이고 있습니다. 모델은 제공된 문맥 내 정보를 정확히 이해하고, 기준에 부합하는 답변을 생성하고 있음을 보여줍니다.</br>
   
   3. **faithfulness: 0.8358**</br>
      해석: 83%로 신뢰할 만한 수준이지만, 간혹 정보 왜곡이 있을 가능성을 나타냅니다.</br>
   
   4. **answer_relevancy: 0.6326**</br>
      해석: 63%로 상대적으로 낮은 점수입니다. 질문과 관련 없는 내용이 포함되거나, 질문에 대한 직접적인 답변을 생성하지 못했을 가능성이 큽니다.</br>
      
   #### 개선 방향:
   - **컨텍스트 선택 최적화:** </br>
      - 더 적합한 정보를 검색하고 모델에 제공하기 위한 컨텍스트 필터링 방식을 개선하세요.</br>
      - 중요한 정보와 덜 중요한 정보를 구분해, 관련성이 높은 데이터만 모델에 입력하도록 설계하세요.
   - **질문 이해 능력 강화:**

      - 모델이 질문의 의도를 더 잘 이해할 수 있도록 학습 데이터를 개선하거나 fine-tuning을 적용하세요.
      - 질문의 구조를 분석하고, 의도를 추출하는 별도의 처리 단계를 추가하세요.
   - **RAG 워크플로우 개선:**

      - 벡터 데이터베이스에서 검색된 정보의 품질을 높이기 위해 데이터셋의 벡터화를 정교하게 다듬으세요.
      - 관련성이 낮은 검색 결과를 걸러내는 후처리 알고리즘을 추가적으로 설계하세요.
   - **결과 모니터링 및 반복 개선:**
      
      - 여러 지표를 모니터링하며, 모델 성능의 균형을 맞추는 과정을 반복하세요.
      - 특히, 낮은 점수를 보이는 answer_relevancy 지표 개선에 우선적으로 집중하는 것이 중요합니다.
      


### Stack
- Environment
  
    ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white)
    ![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)

- Develpoment
  
    ![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)
- Communication
  
    ![Discord](https://img.shields.io/badge/discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)  

### 회고
- 형진
  > 아아
  >
- 해원
  > 평가를 하는 과정에서 어려움이 많이 발생한 것같다.
  >
- 수연
  > RAG 성능평가가 제일 어려웠다. 그 부분을 좀 더 공부해봐야할 것 같다.
  > 
- 연경
  > 이번 프로젝트는 건강 때문에 더 열심히 참여하지 못해 아쉬움이 남는 것같다. 더욱 열심히 공부해서 복습을 해봐야할 것같다.
  >
- 미현
  > 이제 기계를 믿을 수 있나?
  > 


