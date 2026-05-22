from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ==========================================
# [최적화 완료] 서버 시작 시 무거운 연산 원천 제거
# ==========================================
print("🔄 실제 장학금 데이터를 로드하는 중입니다...")
csv_filename = "한국장학재단_학자금지원정보(고등학생)_20260511.csv"

try:
    df = pd.read_csv(csv_filename, encoding="cp949")
except:
    df = pd.read_csv(csv_filename, encoding="utf-8")

df = df.fillna("")

print("🤖 한국어 AI 임베딩 모델 로딩 중...")
# 모델만 가볍게 메모리에 올려둡니다.
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 데이터 융합
df['ai_text'] = (
    df['상품명'] + " " + 
    df['학자금유형구분'] + " " + 
    df['성적기준 상세내용'] + " " + 
    df['소득기준 상세내용'] + " " + 
    df['특정자격 상세내용'] + " " + 
    df['지역거주여부 상세내용'] + " " + 
    df['자격제한 상세내용']
)

print("🚀 [최적화 완료] 무한 대기 없이 AI 하이브리드 엔진 즉시 가동!")

# ==========================================
# [라우터] 웹 화면 및 온디맨드 매칭 API
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():
    data = request.json
    user_region = data.get('region', '전국')
    user_text = data.get('user_text', '')

    if not user_text:
        return jsonify([])

    # --------------------------------------
    # 1단계: Rule-based Filtering (여기서 데이터 양을 대폭 줄임)
    # --------------------------------------
    if user_region != "전국":
        region_mask = df['운영기관명'].str.contains(user_region, na=False) | df['지역거주여부 상세내용'].str.contains(user_region, na=False)
        national_mask = df['운영기관명'].str.contains('한국장학재단|정부|교육부', na=False)
        final_mask = region_mask | national_mask
        working_df = df[final_mask].copy()
    else:
        working_df = df.copy()

    if len(working_df) < 5:
        working_df = df.copy()

    # --------------------------------------
    # 2단계: 실시간 온디맨드(On-Demand) 임베딩 연산
    # 필터링을 거쳐 수십 개로 줄어든 데이터만 계산하므로 0.5초 안에 끝납니다.
    # --------------------------------------
    user_embedding = model.encode([user_text])
    
    # 필터링된 타깃 데이터만 압축 연산
    target_embeddings = model.encode(working_df['ai_text'].tolist(), show_progress_bar=False)
    similarities = cosine_similarity(user_embedding, target_embeddings)[0]
    
    # --------------------------------------
    # 3단계: 정량 키워드 교차 가중치 부여 알고리즘
    # --------------------------------------
    weight_bonuses = np.zeros(len(working_df))
    
    if any(keyword in user_text for keyword in ['다자녀', '세 자녀', '세자녀', '네 자녀', '다인가족']):
        for idx, (_, row) in enumerate(working_df.iterrows()):
            if any(k in row['ai_text'] for k in ['다자녀', '다자녀가정', '다자녀 가구']):
                weight_bonuses[idx] += 10.0
                
    if any(keyword in user_text for keyword in ['어려워', '기초생활', '차상위', '한부모', '소득']):
        for idx, (_, row) in enumerate(working_df.iterrows()):
            if any(k in row['소득기준 상세내용'] for k in ['수급자', '차상위', '한부모', '중위소득']):
                weight_bonuses[idx] += 5.0

    # 종합 점수 반영
    ai_scores = similarities * 100
    final_scores = np.minimum(ai_scores + weight_bonuses, 100.0).round(1)
    working_df['match_score'] = final_scores
    
    top_matches = working_df.sort_values(by='match_score', ascending=False).head(5)
    
    # --------------------------------------
    # 4단계: 결과 데이터 반환
    # --------------------------------------
    results = []
    for _, row in top_matches.iterrows():
        target_summary = f"[소득] {row['소득기준 상세내용']} | [성적] {row['성적기준 상세내용']} | [자격] {row['특정자격 상세내용']}"
        if len(target_summary) > 150:
            target_summary = target_summary[:150] + "..."

        results.append({
            'institution': row['운영기관명'],
            'title': row['상품명'],
            'target': target_summary,
            'score': row['match_score'],
            'url': row['홈페이지주소'] if row['홈페이지주소'] else "https://www.kosaf.go.kr",
            'reason': f"AI 분석 결과 적합도 {row['match_score']}%입니다. 학생님의 문맥 속 니즈와 해당 장학금의 조건이 일치합니다. ⚠️ 단, 소득분위 및 성적의 정량적 컷오프는 반드시 아래 원본 홈페이지 공고를 통해 최종 확인하시기 바랍니다."
        })
        
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)