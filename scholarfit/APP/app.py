from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

app = Flask(__name__, template_folder=os.path.join(parent_dir, 'templates'))

print("🔄 고도화된 장학금 데이터 로딩 프로세스 시작...")
csv_filename = "한국장학재단_학자금지원정보_키워드포함.csv"

possible_paths = [
    os.path.join(parent_dir, csv_filename),
    os.path.join(base_dir, csv_filename),
    csv_filename,
    f"APP/{csv_filename}",
    os.path.join(parent_dir, "한국장학재단_학자금지원정보(고등학생)_20260511.csv"),
    os.path.join(base_dir, "한국장학재단_학자금지원정보(고등학생)_20260511.csv")
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
        print("⚠️ 지정된 CSV 파일을 찾을 수 없어 빈 테이블로 엔진을 시작합니다.")
        df = pd.DataFrame(columns=['운영기관명', '상품명', '학자금유형구분', '성적기준 상세내용', '소득기준 상세내용', '특정자격 상세내용', '지역거주여부 상세내용', '자격제한 상세내용', '홈페이지주소', '추천_키워드'])
except Exception as e:
    print(f"❌ 데이터 로드 중 오류 발생: {str(e)}")
    df = pd.DataFrame(columns=['운영기관명', '상품명', '학자금유형구분', '성적기준 상세내용', '소득기준 상세내용', '특정자격 상세내용', '지역거주여부 상세내용', '자격제한 상세내용', '홈페이지주소', '추천_키워드'])

df = df.fillna("")

if '추천_키워드' not in df.columns:
    df['추천_키워드'] = ""

print("🤖 1,140개 맞춤형 키워드를 융합한 매칭 엔진 빌드 중...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

if not df.empty:
    df['ai_text'] = (
        df['상품명'].astype(str) + " " + 
        df['학자금유형구분'].astype(str) + " " + 
        df['성적기준 상세내용'].astype(str) + " " + 
        df['소득기준 상세내용'].astype(str) + " " + 
        df['특정자격 상세내용'].astype(str) + " " + 
        df['지역거주여부 상세내용'].astype(str) + " " + 
        df['자격제한 상세내용'].astype(str) + " " +
        df['추천_키워드'].astype(str)
    )
    tfidf_matrix = vectorizer.fit_transform(df['ai_text'])
else:
    df['ai_text'] = ""
    tfidf_matrix = None

print("🚀 엔진 세팅 완료! 메모리 가성비 극대화 모드로 가동 중입니다.")

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
        
        raw_scores = (similarities * 60) + 40 
        final_scores = np.minimum(raw_scores, 99.4).round(1)
        working_df['match_score'] = final_scores
        
        top_matches = working_df.sort_values(by='match_score', ascending=False).head(5)
        
        results = []
        for _, row in top_matches.iterrows():
            target_summary = f"[소득] {row['소득기준 상세내용']} | [성적] {row['성적기준 상세내용']} | [자격] {row['특정자격 상세내용']}"
            if len(target_summary) > 150:
                target_summary = target_summary[:150] + "..."

            reason_msg = f"입력하신 조건과 장학 제도의 핵심 키워드가 분석되어 적합도 {row['match_score']}%로 추천되었습니다."
            if row['추천_키워드']:
                reason_msg = f"키워드 [{row['추천_키워드']}] 기반 분석 결과, 적합도 {row['match_score']}%로 추천되었습니다."

            results.append({
                'institution': row['운영기관명'],
                'title': row['상품명'],
                'target': target_summary,
                'score': row['match_score'],
                'url': row['홈페이지주소'] if str(row['홈페이지주소']).startswith('http') else "https://www.kosaf.go.kr",
                'reason': reason_msg
            })
            
        return jsonify(results)
    except Exception as e:
        return jsonify([{"title": "오류 발생", "target": str(e), "score": 0}])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
