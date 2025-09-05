import os
import time
import json
import csv
import argparse
import logging
from datetime import datetime

import google.generativeai as genai
from datasets import load_dataset
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# --- 기본 설정 (Basic Setup) ---

# 로깅 설정: 터미널에 진행 상황을 명확하게 보여주기 위함
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NLTK 데이터 다운로드 (METEOR 평가 및 토큰화에 필요)
# 처음 실행 시에만 다운로드합니다.
REQUIRED_NLTK_PACKAGES = ['punkt', 'wordnet'] # 'punkt_tab'은 보통 'punkt'에 포함되거나 필요 없는 경우가 많습니다.
try:
    for package in REQUIRED_NLTK_PACKAGES:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
except LookupError:
    print(f"NLTK 데이터({', '.join(REQUIRED_NLTK_PACKAGES)})를 다운로드합니다. 잠시만 기다려주세요...")
    for package in REQUIRED_NLTK_PACKAGES:
        nltk.download(package, quiet=True)
    print("NLTK 데이터 다운로드 완료.")


# --- 핵심 기능 함수 (Core Functions) ---

def configure_gemini():
    """Gemini API 키를 환경 변수에서 로드하고 모델을 설정합니다."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. API 키를 설정해주세요.")
    genai.configure(api_key=api_key)
    # 답변 생성에 사용할 모델 설정
    model = genai.GenerativeModel('gemini-2.5-flash') # 모델명을 최신 모델로 변경
    return model

def load_hotpotqa_dataset(split='validation'):
    """
    Hugging Face에서 HotpotQA 데이터셋을 로드합니다.
    로컬 'datasets' 디렉토리에 데이터셋을 저장하고, 이미 존재하면 다운로드 없이 불러옵니다.
    """
    cache_directory = "datasets"
    logging.info(f"hotpot_qa 데이터셋 ('fullwiki' 설정, '{split}' 스플릿)을 확인/로드합니다.")
    logging.info(f"데이터셋 저장 및 캐시 디렉토리: '{cache_directory}'")
    return load_dataset("hotpot_qa", "fullwiki", split=split, cache_dir=cache_directory)

def generate_answer_with_context(model, question, context):
    """
    제공된 컨텍스트(Supporting Facts)를 기반으로 질문에 대한 답변을 생성합니다.
    이것이 'Retrieval-Augmented Generation (RAG)'의 Generation 단계에 해당합니다.
    """
    prompt = f"""
    You are a helpful assistant that answers questions based ONLY on the provided context.
    Do not use any external knowledge.
    
    IMPORTANT: Provide a SHORT and CONCISE answer. If the answer is a simple yes/no, just say "yes" or "no".
    If it's a name, date, or short fact, provide only that information.
    Do not add explanations unless specifically asked.
    
    Context:
    ---
    {context}
    ---
    Question: {question}

    Answer: 


    Example1:



    Question: Which magazine was started first Arthur's Magazine or First for Women?

    Answer: Arthur's Magazine



    Context:

    ---

    ...

    ---



    Example2:


    Question: The Oberoi family is part of a hotel company that has a head office in what city?

    Answer: Delhi


    Context:

    ---

    ...

    ---
    """
    try:
        # 안전 설정 추가 (콘텐츠 필터링 관련 오류 방지)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API 호출 중 오류 발생: {e}")
        return "[답변 생성 실패]"

def main(num_queries):
    """메인 실행 함수"""
    start_time = time.time()
    logging.info("HotpotQA 데이터셋 평가 스크립트를 시작합니다.")
    logging.info(f"처리할 쿼리 수: {num_queries}")

    # --- 1. 환경 설정 및 모델/데이터 로딩 ---
    try:
        model = configure_gemini()
    except ValueError as e:
        logging.error(e)
        return

    dataset = load_hotpotqa_dataset()
    dataset_name = "hotpotqa"

    # --- 2. 결과 저장 디렉토리 생성 ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"결과를 저장할 '{results_dir}' 디렉토리를 확인/생성했습니다.")

    # --- 3. 데이터 처리 및 평가 ---
    generation_results = []
    total_bleu = 0
    total_meteor = 0
    processed_count = 0

    num_samples_to_process = min(num_queries, len(dataset))
    if num_samples_to_process < num_queries:
        logging.warning(
            f"요청된 쿼리 수({num_queries})가 데이터셋 스플릿의 실제 크기({len(dataset)})보다 큽니다. "
            f"{num_samples_to_process}개만 처리합니다."
        )

    # tqdm을 사용하여 터미널에 진행률 표시
    for example in tqdm(dataset.select(range(num_samples_to_process)), total=num_samples_to_process, desc="쿼리 처리 중"):
        question = example['question']
        ground_truth_answer = example['answer']

        # === 모든 컨텍스트를 프롬프트에 포함 ===
        print(f"\n{'='*80}")
        print(f"질문 ID: {example.get('id', 'N/A')}")
        print(f"질문: {question}")
        print(f"정답: {ground_truth_answer}")
        print(f"{'='*80}")
        
        # 모든 컨텍스트를 하나의 문자열로 합치기
        all_context = ""
        try:
            if isinstance(example['context'], dict) and 'title' in example['context'] and 'sentences' in example['context']:
                titles = example['context']['title']
                sentences_lists = example['context']['sentences']
                
                print(f"컨텍스트 문서 수: {len(titles)}")
                
                for i, (title, sentences) in enumerate(zip(titles, sentences_lists)):
                    all_context += f"\n=== 문서 {i+1}: {title} ===\n"
                    for j, sentence in enumerate(sentences):
                        all_context += f"{j+1}. {sentence}\n"
                    all_context += "\n"
                
                print(f"전체 컨텍스트 길이: {len(all_context)} 문자")
                print(f"컨텍스트 미리보기 (처음 500자): {all_context[:500]}...")
                
            else:
                print("경고: 예상치 못한 컨텍스트 구조")
                all_context = str(example['context'])
                
        except Exception as e:
            print(f"오류: 컨텍스트 처리 중 오류 발생: {e}")
            all_context = str(example['context'])
        
        # Gemini를 통해 답변 생성 (모든 컨텍스트 포함)
        generated_answer = generate_answer_with_context(model, question, all_context)
        
        # 답변 길이 정보 출력
        print(f"정답 길이: {len(ground_truth_answer)} 문자, 생성 답변 길이: {len(generated_answer)} 문자")
        print(f"정답: '{ground_truth_answer}'")
        print(f"생성: '{generated_answer}'")

        # 평가 점수 계산
        reference_tokens = word_tokenize(ground_truth_answer.lower())
        candidate_tokens = word_tokenize(generated_answer.lower())

        bleu = 0.0
        meteor = 0.0
        if reference_tokens and candidate_tokens:
            try:
                # Smoothing Function을 사용하여 더 안정적인 BLEU 점수 계산
                smoothing = SmoothingFunction().method1
                
                # 짧은 답변에 더 적합한 가중치 설정
                if len(candidate_tokens) <= 3:
                    # 짧은 답변: 1-gram과 2-gram만 사용
                    bleu = sentence_bleu([reference_tokens], candidate_tokens, 
                                       weights=(0.7, 0.3, 0.0, 0.0), 
                                       smoothing_function=smoothing)
                elif len(candidate_tokens) <= 6:
                    # 중간 길이: 1,2,3-gram 사용
                    bleu = sentence_bleu([reference_tokens], candidate_tokens, 
                                       weights=(0.6, 0.3, 0.1, 0.0), 
                                       smoothing_function=smoothing)
                else:
                    # 긴 답변: 모든 n-gram 사용
                    bleu = sentence_bleu([reference_tokens], candidate_tokens, 
                                       weights=(0.4, 0.3, 0.2, 0.1), 
                                       smoothing_function=smoothing)
                
                meteor = meteor_score([reference_tokens], candidate_tokens)
            except ZeroDivisionError:
                 logging.warning(f"ID {example.get('id', 'N/A')} - 점수 계산 중 ZeroDivisionError 발생.")
        
        total_bleu += bleu
        total_meteor += meteor
        processed_count += 1
        
        generation_results.append({
            'id': example['id'],
            'question': question,
            'ground_truth_answer': ground_truth_answer,
            'context': all_context,
            'generated_answer': generated_answer,
            'bleu_score': bleu,
            'meteor_score': meteor
        })

    # --- 4. 최종 결과 계산 및 저장 ---
    end_time = time.time()
    execution_time = end_time - start_time

    avg_bleu = total_bleu / processed_count if processed_count > 0 else 0
    avg_meteor = total_meteor / processed_count if processed_count > 0 else 0

    logging.info("=" * 50)
    logging.info("평가 완료!")
    logging.info(f"총 실행 시간: {execution_time:.2f}초")
    logging.info(f"처리된 쿼리 수: {processed_count}")
    logging.info(f"평균 BLEU 점수: {avg_bleu:.4f}")
    logging.info(f"평균 METEOR 점수: {avg_meteor:.4f}")
    logging.info("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"{dataset_name}_generation_results_{timestamp}.csv")
    json_filename = os.path.join(results_dir, f"{dataset_name}_evaluation_scores_{timestamp}.json")
    
    if generation_results:
        logging.info(f"생성 결과를 CSV 파일로 저장합니다: {csv_filename}")
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=generation_results[0].keys())
            writer.writeheader()
            writer.writerows(generation_results)
    else:
        logging.warning("처리된 결과가 없어 CSV 파일을 생성하지 않습니다.")

    evaluation_scores = {
        'dataset_name': dataset_name,
        'model_used': 'gemini-2.5-flash',
        'num_queries_processed': processed_count,
        'total_execution_time_seconds': round(execution_time, 2),
        'average_bleu_score': round(avg_bleu, 4),
        'average_meteor_score': round(avg_meteor, 4),
    }
    logging.info(f"평가 점수를 JSON 파일로 저장합니다: {json_filename}")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(evaluation_scores, f, indent=4, ensure_ascii=False)

    logging.info("모든 작업이 성공적으로 완료되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HotpotQA 데이터셋에 대해 RAG 파이프라인을 실행하고 성능을 평가합니다.")
    parser.add_argument(
        '--num_queries',
        type=int,
        default=10,
        help='평가할 쿼리의 개수 (API 비용 관리를 위해 조절)'
    )
    args = parser.parse_args()
    
    main(args.num_queries)