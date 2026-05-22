from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import re
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# [핵심 수정] 한 칸 상위 폴더에 위치한 templates 폴더를 바라보도록 절대 경로 설정 변경
base_dir = os.path.dirname(os.path.abspath(__file__)) # APP 폴더 위치
parent_dir = os.path.dirname(base_dir)                # scholarfit (상위 폴더) 위치

app = Flask(__name__, template_folder=os.path.join(parent_dir, 'templates'))

print("🔄 장학금 데이터 로딩 프로세스 시작...")
csv_filename = "한국장학재단_학자금지원정보(고등학생)_20260511.csv"

# 여러 경로에서 파일 탐색
possible_paths = [
    os.path.join(parent_dir, csv_filename),
    os.path.join(base_dir, csv_filename),
    csv_filename,
    f"APP/{csv_filename}"
]

target_path = None
for path in possible_paths:
    if os.path.exists(path):
        target_path = path
        break

try:
    if target_path:
        print(f"📍 데이터 파일을 찾았습니다: {target_path}")
        try:
            df = pd.read_csv(target_path, encoding="cp949")
        except Exception:
            df = pd.read_csv(target_path, encoding="utf-8")
    else:
        # 상위 폴더나 현재 폴더 내 아무 CSV나 탐색
        csv_files = [f for f in os.listdir(parent_dir) if f.endswith('.csv')] or [f for f in os.listdir(base_dir) if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(os.path.join(parent_dir, csv_files[0]), encoding="cp949")
        else:
            print("⚠️ CSV 파일을 찾을 수 없어 빈 테이블로 엔진을 시작합니다.")
            df = pd.DataFrame(columns=['운영기관명', '상품명', '학자금유형구분', '성적기준 상세내용', '소득기준 상세내용', '특정자격 상세내용', '지역거주여부 상세내용', '자격제한 상세내용', '홈페이지주소'])
except Exception as e:
    df = pd.DataFrame(columns=['운영기관명', '상품명', '학자금유형구분', '성적기준 상세내용', '소득기준 상세내용', '특정자격 상세내용', '지역거주여부 상세내용', '자격제한 상세내용', '홈페이지주소'])

df = df.fillna("")

print("🤖 매칭 엔진 빌드 중...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

if not df.empty:
    df['ai_text'] = (
        df['상품명'].astype(str) + " " + 
        df['학자금유형구분'].astype(str) + " " + 
        df['성적기준 상세내용'].astype(str) + " " + 
        df['소득기준 상세내용'].astype(str) + " " + 
        df['특정자격 상세내용'].astype(str) + " " + 
        df['지역거주여부 상세내용'].astype(str) + " " + 
        df['자격제한 상세내용'].astype(str)
    )
    tfidf_matrix = vectorizer.fit_transform(df['ai_text'])
else:
    df['ai_text'] = ""
    tfidf_matrix = None

print("🚀 엔진 세팅 완료!")

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        error_msg = f"<h3>❌ 렌더링 에러 발생!</h3>" \
                    f"<p><b>이유:</b> {str(e)}</p>" \
                    f"<p><b>현재 지정된 templates 주소:</b> {os.path.join(parent_dir, 'templates')}</p>" \
                    f"<pre>{traceback.format_exc()}</pre>"
        return error_msg, 500

@app.route('/match', methods=['POST'])
def match():
    if df.empty or tfidf_matrix is None:
        return jsonify([])
        
    try:
        data = request.json
        user_region = data.get('region', '전국')
        user_text = data.get('user_text', '')

        if not user_text:
            return jsonify([])

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

        user_vec = vectorizer.transform([user_text])
        target_vecs = tfidf_matrix[working_indices]
        similarities = cosine_similarity(user_vec, target_vecs)[0]
        
        weight_bonuses = np.zeros(len(working_df))
        if any(k in user_text for k in ['다자녀', '세 자녀', '세자녀', '네 자녀', '다인가족']):
            for idx, (_, row) in enumerate(working_df.iterrows()):
                if any(k in row['ai_text'] for k in ['다자녀', '다자녀가정', '다자녀 가구']):
                    weight_bonuses[idx] += 25.0
                    
        if any(k in user_text for k in ['어려워', '기초생활', '차상위', '한부모', '소득', '어려운']):
            for idx, (_, row) in enumerate(working_df.iterrows()):
                if any(k in row['소득기준 상세내용'] for k in ['수급자', '차상위', '한부모', '중위소득', '기초생활']):
                    weight_bonuses[idx] += 20.0

        raw_scores = (similarities * 60) + 40 
        final_scores = np.minimum(raw_scores + weight_bonuses, 99.4).round(1)
        working_df['match_score'] = final_scores
        
        top_matches = working_df.sort_values(by='match_score', ascending=False).head(5)
        
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
                'reason': f"매칭 엔진 분석 결과 적합도 {row['match_score']}%입니다."
            })
            
        return jsonify(results)
    except Exception as e:
        return jsonify([{"title": "오류 발생", "target": str(e), "score": 0}])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
