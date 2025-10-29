from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pathlib import Path
import yaml

load_dotenv()

class State(TypedDict):
  messages: List[HumanMessage]
  doc_type: Optional[str]
  template_content: Optional[str]



class CreateDocumentAgent:
  def __init__(self, model='gpt-4o', temperature=0.7):

    # LLM 초기화
    self.llm = ChatOpenAI(model=model, temperature=temperature)

    # YAML 파일에서 템플릿 로드
    self.doc_prompts = self._load_templates()

    # 그래프 초기화
    self.app = self._build_graph()

  def _load_templates(self):
    try:
        # 현재 스크립트와 같은 디렉토리에서 templates.yaml 파일 찾기
        current_dir = Path(__file__).parent
        template_path = current_dir / "templates.yaml"
        
        # 템플릿 파일 존재 여부 확인
        if not template_path.exists():
            print(f"[WARNING] 템플릿 파일을 찾을 수 없습니다: {template_path}")
            return {}
        
        # YAML 파일 읽기 및 파싱
        with open(template_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data.get('templates', {})
            
    except Exception as e:
        print(f"[ERROR] 템플릿 로드 중 오류 발생: {e}")
        return {}

  def classify_doc_type(self, state: State) -> State:
    """
    사용자 입력에서 문서 타입을 분류하는 함수.
    """
    user_message = state["messages"][-1].content

    classification_prompt = ChatPromptTemplate.from_messages([
      ('system', """
      사용자의 요청을 분석하여 다음 문서 타입 중 하나로 분류해주세요:
      1. 영업방문 결과보고서 - 고객 방문, 영업 활동 관련
      2. 제품설명회 시행 신청서 - 제품설명회 진행 계획, 신청 관련
      3. 제품설명회 시행 결과보고서 - 제품설명회 완료 후 결과 보고 관련

      반드시 위 3가지 중 하나의 정확한 문서 타입 이름만 응답해주세요.
      앞에 숫자는 제거하고 문서명만 출력하세요.
      """),
      ('human', '영업 방문 결과보고서를 작성해줘'),
      ('assistant', '영업방문 결과보고서'),
      ('human', '실적분석레포트를 작성해줘'),
      ('assistant', '지원하지 않는 문서 타입입니다.'),
      ('human', '{user_request}'),
    ])

    response = self.llm.invoke(classification_prompt.format_messages(user_request=user_message))
    content = response.content
    
    if content not in ['영업방문 결과보고서', '제품설명회 시행 신청서', '제품설명회 시행 결과보고서']:
      # 지원하지 않는 문서로 분류된 경우, False
      state['doc_type'] = False
      print(f"분류된 문서 타입: {content}")
      return state
    else:
      # 지원하는 문서로 분류된 경우, 문서명과 해당 template
      state['doc_type'] = content
      print(f"분류된 문서 타입: {content}")
      state["template_content"] = self.doc_prompts[content]["document_information"]
      print(f"{content} 템플릿 추가 완료: ")
      return state

  def _build_graph(self):
    workflow = StateGraph(State)

    # 노드 추가
    workflow.add_node('classify_doc_type', self.classify_doc_type)

    workflow.set_entry_point('classify_doc_type')
    workflow.add_edge('classify_doc_type', END)

    return workflow.compile()

  def run(self, user_input: str):
    # 초기 상태 설정
    initial_state = {
        'messages': [HumanMessage(content=user_input)],
        'doc_type': None,
        'template_content': None,
    }
    result = self.app.invoke(initial_state)
    return result

if __name__ == "__main__":
    # 통합 문서 작성 시스템 실행
    agent = CreateDocumentAgent()
    result = agent.run(user_input="제품 설명회 시행 결과 보고서 작성할거야")
    print("\n=== 최종 결과 ===")
    print(result)