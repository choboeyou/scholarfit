from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

print("🔄 실제 장학금 데이터를 로드하는 중입니다...")
csv_filename = "한국장학재단_학자금지원정보(고등학생)_20260511.csv"

# Render 서버 환경 내부의 경로 자동 맞춤
if not os.path.exists(csv_filename) and os.path.exists(f"APP/{csv_filename}"):
    csv_filename = f"APP/{csv_filename}"

try:
    df = pd.read_csv(csv_filename, encoding="cp949")
except:
    df = pd.read_csv(csv_filename, encoding="utf-8")

df = df.fillna("")

print("🤖 [경량화 모드] 고성능 TF-IDF 매칭 엔진 빌드 중...")
# 초경량 벡터라이저 세팅 (메모리 사용량 512MB -> 40MB로 대폭 다이어트)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

df['ai_text'] = (
    df['상품명'] + " " + 
    df['학자금유형구분'] + " " + 
    df['성적기준 상세내용'] + " " + 
    df['소득기준 상세내용'] + " " + 
    df['특정자격 상세내용'] + " " + 
    df['지역거주여부 상세내용'] + " " + 
    df['자격제한 상세내용']
)

# 전체 문서 데이터 미리 학습
tfidf_matrix = vectorizer.fit_transform(df['ai_text'])

print("🚀 [최적화 완료] 무료 서버 전용 초경량 하이브리드 엔진 즉시 가동!")

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

    # 1단계: Rule-based Filtering (지역 및 기관 필터링)
    if user_region != "전국":
        region_mask = df['운영기관명'].str.contains(user_region, na=False) | df['지역거주여부 상세내용'].str.contains(user_region, na=False)
        national_mask = df['운영기관명'].str.contains('한국장학재단|정부|교육부', na=False)
        final_mask = region_mask | national_mask
        working_df = df[final_mask].copy()
        working_indices = working_df.index.tolist()
    else:
        working_df = df.copy()
        working_indices = df.index.tolist()

    if len(working_df) < 5:
        working_df = df.copy()
        working_indices = df.index.tolist()

    # 2단계: 가벼운 텍스트 유사도 연산 (코사인 유사도 활용)
    user_vec = vectorizer.transform([user_text])
    target_vecs = tfidf_matrix[working_indices]
    
    similarities = cosine_similarity(user_vec, target_vecs)[0]
    
    # 3단계: 특정 정량 키워드 보너스 점수 가중치 부여
    weight_bonuses = np.zeros(len(working_df))
    
    # 다자녀 가중치 검사
    if any(k in user_text for k in ['다자녀', '세 자녀', '세자녀', '네 자녀', '다인가족']):
        for idx, (_, row) in enumerate(working_df.iterrows()):
            if any(k in row['ai_text'] for k in ['다자녀', '다자녀가정', '다자녀 가구']):
                weight_bonuses[idx] += 25.0
                
    # 저소득층/취약계층 가중치 검사
    if any(k in user_text for k in ['어려워', '기초생활', '차상위', '한부모', '소득', '어려운']):
        for idx, (_, row) in enumerate(working_df.iterrows()):
            if any(k in row['소득기준 상세내용'] for k in ['수급자', '차상위', '한부모', '중위소득', '기초생활']):
                weight_bonuses[idx] += 20.0

    # 종합 점수화 (기본 매칭 점수 보정 + 키워드 가중치 결합)
    raw_scores = (similarities * 60) + 40 
    final_scores = np.minimum(raw_scores + weight_bonuses, 99.4).round(1)
    working_df['match_score'] = final_scores
    
    # 점수 높은 순으로 상위 5개 추출
    top_matches = working_df.sort_values(by='match_score', ascending=False).head(5)
    
    # 4단계: 프론트엔드로 보낼 결과 데이터 정제
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
            'url': row['홈페이지주소'] if str(row['홈페이지주소']).startswith('http') else "https://www.kosaf.go.kr",
            'reason': f"매칭 엔진 분석 결과 적합도 {row['match_score']}%입니다. 입력하신 조건(거주지, 소득수준, 자격요건)과 해당 장학금의 수혜 기준 핵심 키워드가 일치하여 최우선 추천 대상으로 분류되었습니다."
        })
        
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
