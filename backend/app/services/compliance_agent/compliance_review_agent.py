from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
import json

load_dotenv()

class State(TypedDict):
  doc_type: Optional[str]
  item_content: Optional[str]
  review_item: Optional[str]
  review_result: Optional[str]

class ComplianceReviewAgent:
  def __init__(self, model='gpt-4o', temperature=0.7):
    self.llm = ChatOpenAI(model=model, temperature=temperature)

#========================================================================================================================
# 노드
#======================================================================================================================== 
  def prepare_review_content(self, state: State) -> State:
    """
    문서별 리뷰 과정에 사용될 컨텐츠 준비
    """
    doc_type = state['doc_type']
    original_item_content = state['item_content']
    if doc_type == '영업방문 결과보고서':
      review_item = ['고객사개요', '프로젝트개요', '방문및협의내용', '향후계획및일정', '협조사항및공유사항']
    elif doc_type == '제품설명회 시행 신청서':
      review_item = ['제품설명회시행목적', '제품설명회주요내용']
    elif doc_type == '제품설명회 시행 결과보고서':
      review_item = ['제품설명회시행목적', '제품설명회주요내용', '지급내역', '금액', '1인금액', '주류']
    else:
      review_item = None
    reivew_item = state['review_item']
    return state

#========================================================================================================================
# 그래프 빌드
#========================================================================================================================

#========================================================================================================================
# 그래프 실행
#========================================================================================================================
  def run(self, doc_type: str, item_content: dict):
    # 초기 상태 설정
    initial_state = {
        'doc_type': doc_type,
        'item_content': item_content,
        'review_result': None,
    }
    result = self.app.invoke(initial_state)
    return result