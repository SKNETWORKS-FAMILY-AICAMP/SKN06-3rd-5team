# SKN06-3rd-5Team

3차 단위 프로젝트 - 안형진, 조해원, 전수연, 임연경, 박미현


## (팀명) 
### 팀원
| 안형진 | 조해원 | 전수연 | 임연경 | 박미현 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
|<img width="140px" src="![형진]"/>|<img width="140px" src="![해원]" />|<img width="140px" src="![수연]" />|<img width="140px" src="!![연경]" />|<img width="140px" src="!![미현]" />|
| 1 | 2 | 3 | 4 | 5 | 
</br>

# 🤖 역사적 인물이나 사건에 대해 알려주는 교육 챗봇 🤖

### ✔ 개발 기간                                                  
2024.12.24 ~ 2024.12.26 (총 2일)

## 프로젝트 개요
✔ **목표**
   > RAG기반 질의응답을 이용한 역사적 인물이나 사건에 대해 질문하면 대답하는 챗봇 개발
   >
   1. 역사에 대한 관심 증대
      > 학생들과 일반 사용자가 역사적 인물과 사건에 대해 쉽고 흥미롭게 접근할 수 있도록 돕는 것을 목표로 한다. 이를 통해 역사가 지닌 중요성과 재미를 발견할 수 있는 기회를 제공
      
  2. 지식의 대중화
      > 한국사를 포함한 다양한 역사적 주제를 간단하고 명료하게 설명하여, 누구나 쉽게 이해하고 학습할 수 있는 환경을 조성한다. 이는 특히 역사를 잘 모르는 사용자들에게 유용한 정보를 제공하는 데 초점을 맞춤
      
  
  3. 교육적 활용 가능성
      > 교육 현장에서 학생들의 학습을 지원하는 도구로 활용될 수 있다. 예를 들어, 역사적 인물의 업적, 사건의 배경과 결과 등을 질문하고 답변하는 과정을 통해 학습 효율을 높이고 호기심 자극
   
    
  4. 문화유산 보존
      > 역사적 자료와 정보를 디지털화하여 미래 세대가 지속적으로 사용할 수 있는 자산으로 보존
     
     
      > 특히, 한국사를 비롯한 전 세계의 다양한 역사적 이야기를 기록하고 전달하는 데 기여
     
      
  5. 맞춤형 정보 제공
       > 사용자의 질문에 따라 역사적 정보를 정확하고 빠르게 제공함으로써, 개개인의 필요와 관심사에 적합한 정보를 제공
       
       
✔ **주요 작업**
   1. 데이터 수집

        > 프로젝트 주제에 맞는 데이터를 수집하는데, 정보에 허구가 없고 신뢰도가 높은 자료를 인용할 것.
        > 시각화 자료를 제외하고, 텍스트 위주의 자료만 받아서 사용한다. 
        > 국사편찬위원회의 자료 사용 - http://contents.history.go.kr/front/kc/main.do#self  
       
   2. 전처리
      
         > 줄바꿈, 한자어 포함된 것을 주의하여 데이터 가공을 한다.

   3. RAG 기반 벡터 데이터베이스 저장
      >  
   
   4. LLM 연동
      
      >


![image](https://github.com/user-attachments/assets/85efb451-f6c8-4688-892f-28b6688cde6e)

   
✔ **산출물**
   > PDF문서
   >
   > 데이터셋
   >
   > 시스템 구조
   >
   > 코드
   > 


   > 평가 결과
      - context_recall': 0.7000

      - llm_context_precision_with_reference': 0.8833  
      
      - faithfulness': 0.8358  
      
      - answer_relevancy': 0.6326  
      


## 프로젝트 진행 과정
❗️ 화면 구성은 생략하고 RAG를 위한 백엔드 시스템 구축에 집중 ❗️
</br>
### 1. 데이터 수집 및 전처리

#### 1.1 데이터 수집
  - **목표**
    > 프로젝트 주제에 맞는 데이터를 수집
    > 
  - **방법**
    > 공개 데이터셋 다운로드(우리역사넷 한국사연대기 PDF자료)
    > 

#### 1.2 데이터 전처리
  - **작업**
    > 텍스트 정제, 중복 제거, 불필요한 정보 삭제 등
    >
  - **코드 (Python)**
    ```
    ```



### 2. 벡터화 및 벡터 데이터베이스 저장

#### 2.1 텍스트 임베딩 생성
  - **목표**
    > 텍스트 데이터를 벡터 형태로 변환
    > 
  - **도구**
    > Sentence Transformers, Hugging Face Transformers 등
    > 

  - **코드 (Python)**
    ```
    ```



#### 2.2 벡터 데이터베이스 저장
  - **목표**
    > 생성된 벡터를 벡터 데이터베이스에 저장
    > 
  - **도구**
    > FAISS, Pinecone 등
    > 

  - **코드 (Python with FAISS?)**
    ```
    ```



### 3. RAG 기반 질의 응답 시스템 구현

#### 3.1 LLM 연동 및 RAG Chain 구성
  - **목표**
    > LLM과 벡터 데이터베이스를 연동하여 질의 응답 시스템 구축
    > 
  - **도구**
    > OpenAI API, LangChain 등
    > 

  - **코드 (Python with LangChain)**
    ```
    ```



### 4. 성능 테스트 및 개선

#### 4.1 성능 테스트
  - **목표**
    > RAG 시스템의 질의 응답 성능 평가
    > 
  - **방법**
    > 정확성, 응답 속도 등 평가 지표 설정 및 측정
    > 

#### 4.2 개선 및 최적화
  - **작업**
    > 모델 튜닝, 하이퍼파라미터 조정 등
    > 


### Stack
- Environment
  
    ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white)
    ![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)

- Develpoment
  
    ![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)
- Communication
  
    ![Discord](https://img.shields.io/badge/discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)  

### 회고
형진 
>
>
해원 
>
>
수연 
>
>
연경 
>
>
미현 
>
>

