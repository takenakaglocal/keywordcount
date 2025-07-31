import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from collections import defaultdict, Counter
import itertools
import re
import math
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    st.warning("MeCabがインストールされていません。形態素解析機能が制限されます。")

# OpenAI設定
try:
    from openai import OpenAI
    import json
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("OpenAIライブラリがインストールされていません。AI抽出機能は使用できません。")

# ページ設定
st.set_page_config(
    page_title="地方自治体文書分析システム（共起ネットワーク版）",
    page_icon="📊",
    layout="wide"
)

# タイトル
st.title("📊 地方自治体文書分析システム（共起ネットワーク版）")
st.markdown("---")

# セッション状態の初期化
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = False
if 'df_with_locations' not in st.session_state:
    st.session_state.df_with_locations = None
if 'custom_keywords' not in st.session_state:
    st.session_state.custom_keywords = []
if 'cooccurrence_data' not in st.session_state:
    st.session_state.cooccurrence_data = None
if 'ai_keywords_cache' not in st.session_state:
    st.session_state.ai_keywords_cache = {}

# ORG_CODE_MAPPINGの定義（一部のみ表示）
ORG_CODE_MAPPING = {
    # 例として一部のみ記載
    "011002": "札幌市", "012025": "函館市", "012033": "小樽市",
    "131016": "東京都千代田区", "131024": "東京都中央区", "131032": "東京都港区",
    "141003": "横浜市", "141305": "川崎市", "141500": "相模原市",
    "231002": "名古屋市", "271004": "大阪市", "281000": "神戸市",
    "401005": "福岡市", "401307": "北九州市",
    # 実際にはすべての自治体コードを含める必要があります
}

# 都道府県コードと名前のマッピング
PREFECTURE_MAPPING = {
    "01": "北海道", "02": "青森県", "03": "岩手県", "04": "宮城県", "05": "秋田県",
    "06": "山形県", "07": "福島県", "08": "茨城県", "09": "栃木県", "10": "群馬県",
    "11": "埼玉県", "12": "千葉県", "13": "東京都", "14": "神奈川県", "15": "新潟県",
    "16": "富山県", "17": "石川県", "18": "福井県", "19": "山梨県", "20": "長野県",
    "21": "岐阜県", "22": "静岡県", "23": "愛知県", "24": "三重県", "25": "滋賀県",
    "26": "京都府", "27": "大阪府", "28": "兵庫県", "29": "奈良県", "30": "和歌山県",
    "31": "鳥取県", "32": "島根県", "33": "岡山県", "34": "広島県", "35": "山口県",
    "36": "徳島県", "37": "香川県", "38": "愛媛県", "39": "高知県", "40": "福岡県",
    "41": "佐賀県", "42": "長崎県", "43": "熊本県", "44": "大分県", "45": "宮崎県",
    "46": "鹿児島県", "47": "沖縄県"
}

# 除外する一般的な単語リスト
EXCLUDE_WORDS = {
    # 一般的な動詞・形容詞
    'する', 'ある', 'なる', 'いる', 'できる', 'れる', 'られる', 'よる', 'おる', 'いう', 'もつ',
    'いく', 'くる', 'みる', 'おこなう', '行う', '思う', '考える', '出る', '入る', 'わかる',
    # 一般的な名詞
    'こと', 'もの', 'ため', 'ところ', 'とき', 'ひと', '人', '方', 'とおり', 'まま',
    'よう', 'ほう', 'ほか', 'それ', 'これ', 'あれ', 'どれ', 'ここ', 'そこ', 'あそこ',
    # 行政用語（一般的すぎるもの）
    '施策', '計画', '政策', '課', '部', '局', '室', '係', '担当', '実施', '推進',
    '事業', '業務', '取組', '取り組み', '対応', '実現', '確保', '向上', '促進',
    '強化', '充実', '整備', '活用', '支援', '提供', '構築', '形成', '創出',
    # 接続詞・助詞など
    'および', 'また', 'ただし', 'なお', 'さらに', 'ほか', 'など', '等', 'より',
    'から', 'まで', 'について', 'に関する', 'における', 'による', 'ための',
    # 数字・記号関連
    '年', '月', '日', '第', '条', '項', '号', '章', '節', '款', '目',
    # その他
    'あり', 'なし', 'でき', 'こちら', 'それぞれ', '各', '当', '本', '今', '次'
}

def add_location_columns(df):
    """都道府県名と市区町村名の列を追加"""
    df['prefecture_code'] = df['code'].astype(str).str[:2]
    df['prefecture_name'] = df['prefecture_code'].map(PREFECTURE_MAPPING)
    df['municipality_name'] = df['code'].astype(str).map(ORG_CODE_MAPPING)
    
    # 市区町村名から都道府県名を除去（都道府県庁の場合を除く）
    def clean_municipality_name(row):
        if pd.isna(row['municipality_name']):
            return None
        if row['municipality_name'].endswith('庁'):
            return row['municipality_name']
        return row['municipality_name']
    
    df['municipality_name'] = df.apply(clean_municipality_name, axis=1)
    
    # fiscal_year_start列から年度を抽出（文字列として処理）
    try:
        # fiscal_year_start列を文字列に変換
        df['fiscal_year_str'] = df['fiscal_year_start'].astype(str)
        
        # 年度を抽出する関数
        def extract_fiscal_year(val):
            if pd.isna(val) or val == 'nan':
                return None
            
            val_str = str(val).strip()
            
            # カンマを除去
            val_str = val_str.replace(',', '')
            
            # 1. まず4桁の数字だけの場合（例: "2023", "2,023"）
            if val_str.isdigit() and len(val_str) == 4:
                year = int(val_str)
                if 1900 <= year <= 2100:
                    return year
            
            # 2. 日付形式の場合（例: "2023-04-01", "2023/4/1"）
            import re
            # YYYY-MM-DD または YYYY/MM/DD 形式
            date_match = re.match(r'^(\d{4})[-/](\d{1,2})[-/](\d{1,2})', val_str)
            if date_match:
                year = int(date_match.group(1))
                month = int(date_match.group(2))
                # 4月以降なら年度はそのまま、3月以前なら前年度
                if month >= 4:
                    return year
                else:
                    return year - 1
            
            # 3. 年度表記の場合（例: "2023年度", "令和5年度"）
            year_match = re.search(r'(\d{4})年度', val_str)
            if year_match:
                return int(year_match.group(1))
            
            # 4. その他の4桁数字を含む場合
            four_digit_match = re.search(r'(\d{4})', val_str)
            if four_digit_match:
                year = int(four_digit_match.group(1))
                if 1900 <= year <= 2100:
                    return year
            
            return None
        
        # 年度を抽出
        df['fiscal_year'] = df['fiscal_year_str'].apply(extract_fiscal_year)
        
        # NaNの数を確認
        na_count = df['fiscal_year'].isna().sum()
        if na_count > 0:
            st.info(f"年度を抽出できなかったレコードが{na_count}件あります。")
        
        # Int64型に変換（NaNを含む場合）
        df['fiscal_year'] = df['fiscal_year'].astype('Int64')
        
    except Exception as e:
        st.error(f"年度の抽出でエラーが発生しました: {e}")
        # エラーの場合はカンマを除去して数値に変換
        try:
            df['fiscal_year'] = df['fiscal_year_start'].astype(str).str.replace(',', '').astype(float).astype('Int64')
        except:
            df['fiscal_year'] = pd.NA
    
    return df

def calculate_word_importance_score(word, freq, total_docs):
    """
    単語の重要度スコアを計算
    - 複合語や長い単語を優先
    - 頻度だけでなく単語の特性も考慮
    """
    # 基本スコア（頻度の対数）
    base_score = math.log(freq + 1)
    
    # 長さボーナス
    length_bonus = 1.0
    if len(word) >= 4:
        length_bonus = 1.5
    if len(word) >= 6:
        length_bonus = 2.0
    if len(word) >= 8:
        length_bonus = 3.0
    
    # 文字種の複雑さボーナス
    has_katakana = any('ァ' <= c <= 'ヶ' for c in word)
    has_kanji = any('\u4e00' <= c <= '\u9fff' for c in word)
    has_alpha = any(c.isalpha() and ord(c) < 128 for c in word)
    
    complexity_bonus = 1.0
    char_types = sum([has_katakana, has_kanji, has_alpha])
    if char_types >= 2:
        complexity_bonus = 1.5
    
    # 短い単語にペナルティ
    short_penalty = 1.0
    if len(word) <= 2:
        short_penalty = 0.2
    elif len(word) == 3:
        short_penalty = 0.5
    
    # 総合スコア
    importance_score = (
        base_score * 
        length_bonus * 
        complexity_bonus * 
        short_penalty
    )
    
    return importance_score

def extract_keywords_with_ai(text, api_key, max_keywords=30, sample_mode=False):
    """
    AIを使用してテキストから重要なキーワードを抽出する
    
    Parameters:
    - text: 分析対象のテキスト
    - api_key: OpenAI APIキー
    - max_keywords: 抽出する最大キーワード数
    - sample_mode: サンプリングモードを使用するか
    """
    if not OPENAI_AVAILABLE or not api_key:
        return []
    
    try:
        client = OpenAI(api_key=api_key)
        
        # サンプリングモードの場合、テキストを短縮
        if sample_mode:
            # テキストの最初、中間、最後から抽出
            if len(text) > 3000:
                parts = []
                parts.append(text[:1000])  # 最初の1000文字
                mid_start = len(text) // 2 - 500
                parts.append(text[mid_start:mid_start + 1000])  # 中間の1000文字
                parts.append(text[-1000:])  # 最後の1000文字
                text = ' '.join(parts)
        else:
            # 通常モード：テキストが長すぎる場合は切り詰める
            if len(text) > 4000:
                text = text[:4000]
        
        prompt = f"""
以下のテキストから、最も重要で特徴的なキーワードやフレーズを{max_keywords}個まで抽出してください。

抽出基準：
1. そのテキストの主題や内容を最もよく表す単語・フレーズ
2. 専門用語、固有名詞、重要な概念
3. 複合語も積極的に抽出（例：「関係人口」「地域活性化」「デジタル田園都市」など）
4. 一般的すぎる単語（する、ある、こと、など）は除外
5. 「施策」「計画」「政策」「課」などの一般的な行政用語は除外
6. そのテキスト特有の、他と差別化できる特徴的な用語を優先

出力形式：
キーワードを1行に1つずつ、改行で区切って出力してください。
JSONフォーマットは使用しないでください。
番号や記号は付けないでください。

テキスト：
{text}
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは日本語のテキスト分析の専門家です。地方自治体の文書から特徴的で重要なキーワードを抽出します。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        result = response.choices[0].message.content
        
        # 改行で分割してキーワードを抽出
        keywords = []
        for line in result.split('\n'):
            line = line.strip()
            # 空行、番号付き行、特殊文字を除去
            line = re.sub(r'^[\d\.\-\*\・]+\s*', '', line)
            line = re.sub(r'[{}\[\]"\'`]', '', line)
            
            # JSONっぽい文字列を除外
            if line and len(line) >= 2 and not any(x in line.lower() for x in ['json', 'keywords', '...', '{', '}', '[', ']']):
                # 最終的な除外チェック
                if line not in EXCLUDE_WORDS:
                    keywords.append(line)
        
        return keywords[:max_keywords]
        
    except Exception as e:
        st.error(f"AI抽出エラー: {e}")
        return []

def tokenize_text(text, use_mecab=True, use_compound=True):
    """単一テキストを単語に分解（エラーハンドリング強化版）"""
    if pd.isna(text) or text == '':
        return []
    
    text = str(text)
    
    if use_mecab and MECAB_AVAILABLE:
        try:
            tagger = MeCab.Tagger()
            tagger.parse('')  # 初期化
            
            words = []
            compounds = []
            
            node = tagger.parseToNode(text)
            prev_node = None
            
            while node:
                features = node.feature.split(',')
                pos = features[0]
                
                if pos in ['名詞', '動詞', '形容詞']:
                    word = features[6] if len(features) > 6 and features[6] != '*' else node.surface
                    
                    if word and word not in EXCLUDE_WORDS and len(word) > 1 and not word.isdigit():
                        words.append(word)
                        
                        if use_compound and pos == '名詞' and prev_node:
                            prev_features = prev_node.feature.split(',')
                            if prev_features[0] == '名詞':
                                prev_word = prev_features[6] if len(prev_features) > 6 and prev_features[6] != '*' else prev_node.surface
                                if prev_word and prev_word not in EXCLUDE_WORDS and len(prev_word) > 1:
                                    compound = prev_word + word
                                    if len(compound) <= 10:
                                        compounds.append(compound)
                
                prev_node = node if pos == '名詞' else None
                node = node.next
            
            if use_compound:
                words.extend(list(set(compounds)))
            
            return words
        except Exception as e:
            st.warning(f"MeCab処理エラー: {e}")
            # フォールバック処理
            return tokenize_text(text, use_mecab=False)
    else:
        # 簡易解析
        pattern = r'[ァ-ヴー]+|[ぁ-ん]+|[一-龥]+|[a-zA-Z]+'
        words = re.findall(pattern, text)
        words = [w for w in words if w not in EXCLUDE_WORDS and len(w) > 1]
        return words

def tokenize_text_batch(texts, use_mecab=True, use_compound=True):
    """複数テキストを一括で単語に分解（バッチ処理用）"""
    results = []
    
    if use_mecab and MECAB_AVAILABLE:
        try:
            tagger = MeCab.Tagger()
            tagger.parse('')  # 初期化（一度だけ）
            
            for text in texts:
                if pd.isna(text) or text == '':
                    results.append([])
                    continue
                
                text = str(text)
                words = []
                compounds = []
                
                node = tagger.parseToNode(text)
                prev_node = None
                
                while node:
                    features = node.feature.split(',')
                    pos = features[0]
                    
                    if pos in ['名詞', '動詞', '形容詞']:
                        word = features[6] if len(features) > 6 and features[6] != '*' else node.surface
                        
                        if word and word not in EXCLUDE_WORDS and len(word) > 1 and not word.isdigit():
                            words.append(word)
                            
                            if use_compound and pos == '名詞' and prev_node:
                                prev_features = prev_node.feature.split(',')
                                if prev_features[0] == '名詞':
                                    prev_word = prev_features[6] if len(prev_features) > 6 and prev_features[6] != '*' else prev_node.surface
                                    if prev_word and prev_word not in EXCLUDE_WORDS and len(prev_word) > 1:
                                        compound = prev_word + word
                                        if len(compound) <= 10:
                                            compounds.append(compound)
                    
                    prev_node = node if pos == '名詞' else None
                    node = node.next
                
                if use_compound:
                    words.extend(list(set(compounds)))
                
                results.append(words)
        except:
            # エラー時は簡易解析にフォールバック
            for text in texts:
                results.append(tokenize_text(text, use_mecab=False))
    else:
        # 簡易解析
        for text in texts:
            if pd.isna(text) or text == '':
                results.append([])
                continue
            
            text = str(text)
            pattern = r'[ァ-ヴー]+|[ぁ-ん]+|[一-龥]+|[a-zA-Z]+'
            words = re.findall(pattern, text)
            words = [w for w in words if w not in EXCLUDE_WORDS and len(w) > 1]
            results.append(words)
    
    return results

def calculate_cooccurrence(df, min_count=5, top_n_words=100, use_ai=False, api_key=None, sample_size=None):
    """共起頻度を計算（エラーハンドリング強化版）"""
    import time
    start_time = time.time()
    
    # データの検証
    if len(df) == 0:
        st.error("データが空です")
        return {}, {}, []
    
    # content_text列の存在確認
    if 'content_text' not in df.columns:
        st.error("content_text列が見つかりません")
        return {}, {}, []
    
    # サンプリングの実施
    if sample_size and len(df) > sample_size:
        st.info(f"データ量が多いため、{sample_size}件をランダムサンプリングして処理します。")
        df = df.sample(n=sample_size, random_state=42)
    
    # 単語の出現回数をカウント
    word_counts = Counter()
    # 共起回数をカウント
    cooccurrence_counts = defaultdict(int)
    
    progress_bar = st.progress(0)
    total_docs = len(df)
    
    if use_ai and api_key:
        # AI抽出
        processed_count = 0
        for idx, row in df.iterrows():
            processed_count += 1
            if processed_count % 10 == 0:
                progress_bar.progress(processed_count / total_docs)
            
            text = row['content_text']
            if pd.isna(text):
                continue
                
            text_hash = hash(str(text)[:100])
            
            if text_hash in st.session_state.ai_keywords_cache:
                words = st.session_state.ai_keywords_cache[text_hash]
            else:
                words = extract_keywords_with_ai(text, api_key, sample_mode=True)
                st.session_state.ai_keywords_cache[text_hash] = words
            
            unique_words = list(set(words))
            for word in unique_words:
                word_counts[word] += 1
            
            for word1, word2 in itertools.combinations(sorted(unique_words), 2):
                cooccurrence_counts[(word1, word2)] += 1
    else:
        # 通常の形態素解析
        batch_size = 100
        all_words_list = []
        
        # バッチ処理で形態素解析
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            texts = batch_df['content_text'].tolist()
            
            # バッチで形態素解析
            batch_words = tokenize_text_batch(texts, use_mecab=MECAB_AVAILABLE, use_compound=True)
            all_words_list.extend(batch_words)
            
            # プログレスバー更新
            progress = min((i + batch_size) / total_docs, 1.0)
            progress_bar.progress(progress)
        
        # 単語カウントと共起計算
        for words in all_words_list:
            if not words:  # 空のリストをスキップ
                continue
                
            unique_words = list(set(words))
            
            # 単語の出現回数をカウント
            for word in unique_words:
                word_counts[word] += 1
            
            # 共起回数をカウント
            if len(unique_words) > 1:
                sorted_words = sorted(unique_words)
                for i in range(len(sorted_words)):
                    for j in range(i + 1, len(sorted_words)):
                        cooccurrence_counts[(sorted_words[i], sorted_words[j])] += 1
    
    progress_bar.empty()
    
    # 処理時間を表示
    elapsed_time = time.time() - start_time
    st.info(f"形態素解析完了: {elapsed_time:.1f}秒")
    
    # データが空の場合の処理
    if not word_counts:
        st.warning("単語が抽出できませんでした")
        return {}, {}, []
    
    # 重要度スコアを計算
    word_scores = []
    for word, count in word_counts.items():
        if count >= min_count:
            score = calculate_word_importance_score(word, count, total_docs)
            word_scores.append((word, count, score))
    
    # スコアでソート
    word_scores.sort(key=lambda x: x[2], reverse=True)
    
    # 上位N語を選択
    top_words = [item[0] for item in word_scores[:top_n_words]]
    top_words_set = set(top_words)
    
    # 共起データをフィルタリング
    filtered_cooccurrence = {}
    for (word1, word2), count in cooccurrence_counts.items():
        if word1 in top_words_set and word2 in top_words_set and count >= min_count:
            filtered_cooccurrence[(word1, word2)] = count
    
    # 選択された単語のカウントのみを保持
    filtered_word_counts = {word: word_counts[word] for word in top_words}
    
    return filtered_word_counts, filtered_cooccurrence, top_words

def create_cooccurrence_network(word_counts, cooccurrence_data, top_words, layout_type='spring', 
                              community_resolution=1.0, edge_threshold=0.5):
    """共起ネットワークを作成（エラーハンドリング強化版）"""
    # データの検証
    if not word_counts or not top_words:
        st.error("単語データが空です")
        return [], {}, nx.Graph(), {}
    
    # NetworkXグラフの作成
    G = nx.Graph()
    
    # ノードの追加
    for word in top_words:
        if word in word_counts:
            G.add_node(word, count=word_counts[word])
    
    # エッジの追加（重みの正規化）
    edge_weights = []
    for (word1, word2), count in cooccurrence_data.items():
        if word1 in word_counts and word2 in word_counts:
            # ゼロ除算を防ぐ
            denominator = word_counts[word1] + word_counts[word2] - count
            if denominator > 0:
                weight = count / denominator
            else:
                weight = 0
            G.add_edge(word1, word2, weight=weight, raw_count=count)
            edge_weights.append(weight)
    
    # エッジがない場合の処理
    if not edge_weights:
        st.warning("共起関係が見つかりませんでした")
        edge_weights = [0]  # デフォルト値
    
    # コミュニティ検出
    partition = {}
    try:
        # python-louvainがインストールされている場合
        import community.community_louvain as community_louvain
        if len(G.nodes()) > 0 and len(G.edges()) > 0:
            partition = community_louvain.best_partition(G, resolution=community_resolution)
        else:
            partition = {node: 0 for node in G.nodes()}
    except ImportError:
        # community-louvainがインストールされていない場合
        partition = {node: 0 for node in G.nodes()}
        st.warning("コミュニティ検出ライブラリがインストールされていません。")
    except Exception as e:
        # その他のエラー
        partition = {node: 0 for node in G.nodes()}
        st.warning(f"コミュニティ検出でエラーが発生しました: {e}")
    
    # コミュニティごとにノードをグループ化
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    # グラフが空の場合の処理
    if len(G.nodes()) == 0:
        return [], {'num_nodes': 0, 'num_edges': 0, 'num_communities': 0, 'avg_degree': 0, 'density': 0}, G, partition
    
    # レイアウトの計算
    try:
        if layout_type == 'spring':
            # 初期位置の設定
            pos_init = {}
            num_communities = max(len(communities), 1)
            
            for i, comm_id in enumerate(communities.keys()):
                angle = 2 * np.pi * i / num_communities
                center_x = 2 * np.cos(angle)
                center_y = 2 * np.sin(angle)
                
                nodes = communities[comm_id]
                for j, node in enumerate(nodes):
                    if len(nodes) == 1:
                        pos_init[node] = (center_x, center_y)
                    else:
                        sub_angle = 2 * np.pi * j / len(nodes)
                        radius = 0.5
                        pos_init[node] = (
                            center_x + radius * np.cos(sub_angle),
                            center_y + radius * np.sin(sub_angle)
                        )
            
            pos = nx.spring_layout(G, pos=pos_init, k=2, iterations=50, weight='weight')
        elif layout_type == 'circular':
            pos = nx.circular_layout(G)
        elif layout_type == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.random_layout(G, seed=42)
    except Exception as e:
        st.warning(f"レイアウト計算でエラーが発生しました: {e}")
        pos = nx.random_layout(G, seed=42)
    
    # カラーパレットの設定
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set3
    community_colors = {}
    for i, comm_id in enumerate(sorted(communities.keys())):
        community_colors[comm_id] = colors[i % len(colors)]
    
    # エッジデータの準備
    intra_edges = []
    inter_edges = []
    
    max_weight = max(edge_weights) if edge_weights else 1
    threshold = edge_threshold * max_weight
    
    for edge in G.edges(data=True):
        if 'weight' in edge[2] and edge[2]['weight'] >= threshold:
            edge_data = {
                'x': [pos[edge[0]][0], pos[edge[1]][0], None],
                'y': [pos[edge[0]][1], pos[edge[1]][1], None],
                'weight': edge[2]['weight'],
                'count': edge[2].get('raw_count', 0)
            }
            
            if partition.get(edge[0], 0) == partition.get(edge[1], 0):
                intra_edges.append(edge_data)
            else:
                inter_edges.append(edge_data)
    
    # Plotlyのトレースを作成
    traces = []
    
    # エッジの描画
    for edge in inter_edges:
        traces.append(go.Scatter(
            x=edge['x'], y=edge['y'],
            mode='lines',
            line=dict(width=0.5, color='rgba(200,200,200,0.3)'),
            hoverinfo='text',
            hovertext=f"共起回数: {edge['count']}",
            showlegend=False
        ))
    
    for edge in intra_edges:
        traces.append(go.Scatter(
            x=edge['x'], y=edge['y'],
            mode='lines',
            line=dict(width=edge['weight']*5, color='rgba(100,100,100,0.5)'),
            hoverinfo='text',
            hovertext=f"共起回数: {edge['count']}",
            showlegend=False
        ))
    
    # ノードの描画
    for comm_id, nodes in communities.items():
        if not nodes:
            continue
            
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        hover_text = []
        
        for node in nodes:
            if node in pos:
                node_x.append(pos[node][0])
                node_y.append(pos[node][1])
                node_text.append(node)
                count = G.nodes[node].get('count', 1)
                node_size.append(np.log(count + 1) * 10)
                hover_text.append(f"{node}<br>出現回数: {count}<br>コミュニティ: {comm_id}")
        
        if node_x:  # ノードが存在する場合のみ描画
            traces.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                name=f'コミュニティ {comm_id}',
                marker=dict(
                    size=node_size,
                    color=community_colors.get(comm_id, 'gray'),
                    line=dict(width=2, color='white')
                ),
                text=node_text,
                textposition="top center",
                hovertext=hover_text,
                hoverinfo='text'
            ))
    
    # 統計情報
    stats = {
        'num_nodes': len(G.nodes()),
        'num_edges': len(G.edges()),
        'num_communities': len(communities),
        'avg_degree': sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0,
        'density': nx.density(G) if len(G.nodes()) > 1 else 0
    }
    
    return traces, stats, G, partition

def parse_search_query(query):
    """検索クエリを解析してキーワードと検索タイプを抽出"""
    import re
    
    # AND検索のチェック
    if ' AND ' in query:
        keywords = []
        parts = query.split(' AND ')
        for part in parts:
            part = part.strip()
            # ダブルクォートで囲まれている場合は完全一致
            if part.startswith('"') and part.endswith('"'):
                keywords.append(('exact', part[1:-1]))
            else:
                keywords.append(('partial', part))
        return 'AND', keywords
    
    # OR検索のチェック
    elif ' OR ' in query:
        keywords = []
        parts = query.split(' OR ')
        for part in parts:
            part = part.strip()
            # ダブルクォートで囲まれている場合は完全一致
            if part.startswith('"') and part.endswith('"'):
                keywords.append(('exact', part[1:-1]))
            else:
                keywords.append(('partial', part))
        return 'OR', keywords
    
    # 単一キーワード
    else:
        if query.startswith('"') and query.endswith('"'):
            return 'SINGLE', [('exact', query[1:-1])]
        else:
            return 'SINGLE', [('partial', query)]

def count_keyword_occurrences(df, keyword):
    """キーワードの出現回数をカウント（拡張版）"""
    import re
    
    # 検索クエリを解析
    search_type, keywords = parse_search_query(keyword)
    
    # テキストをNaN対応
    df['content_text_safe'] = df['content_text'].fillna('')
    
    if search_type == 'SINGLE':
        match_type, kw = keywords[0]
        if match_type == 'exact':
            # 完全一致（単語境界を考慮）
            pattern = r'\b' + re.escape(kw) + r'\b'
            df['keyword_count'] = df['content_text_safe'].str.count(pattern)
            df['has_keyword'] = df['keyword_count'] > 0
        else:
            # 部分一致
            df['keyword_count'] = df['content_text_safe'].str.count(re.escape(kw))
            df['has_keyword'] = df['keyword_count'] > 0
    
    elif search_type == 'AND':
        # AND検索：すべてのキーワードを含む場合のみカウント
        df['keyword_count'] = 0
        df['has_keyword'] = True
        
        for match_type, kw in keywords:
            if match_type == 'exact':
                pattern = r'\b' + re.escape(kw) + r'\b'
                temp_count = df['content_text_safe'].str.count(pattern)
            else:
                temp_count = df['content_text_safe'].str.count(re.escape(kw))
            
            df['keyword_count'] += temp_count
            df['has_keyword'] &= (temp_count > 0)
        
        # AND条件を満たさない行はカウントを0にする
        df.loc[~df['has_keyword'], 'keyword_count'] = 0
    
    elif search_type == 'OR':
        # OR検索：いずれかのキーワードを含む場合にカウント
        df['keyword_count'] = 0
        df['has_keyword'] = False
        
        for match_type, kw in keywords:
            if match_type == 'exact':
                pattern = r'\b' + re.escape(kw) + r'\b'
                temp_count = df['content_text_safe'].str.count(pattern)
            else:
                temp_count = df['content_text_safe'].str.count(re.escape(kw))
            
            df['keyword_count'] += temp_count
            df['has_keyword'] |= (temp_count > 0)
    
    # 一時的な列を削除
    df = df.drop('content_text_safe', axis=1)
    
    return df

# サイドバー
with st.sidebar:
    st.header("設定")
    
    # CSVファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルを選択", type=['csv'])
    
    # AI設定（共起ネットワーク用）
    if OPENAI_AVAILABLE:
        st.header("AI設定（共起ネットワーク用）")
        use_ai_extraction = st.checkbox("AIによるキーワード抽出を使用", value=False, 
                                      help="OpenAI APIを使用して特徴的なキーワードを抽出します")
        if use_ai_extraction:
            openai_api_key = st.text_input("OpenAI API Key", type="password",
                                          help="OpenAI APIキーを入力してください")
        else:
            openai_api_key = None
            
        # 高速化オプション
        st.header("高速化設定")
        enable_cache = st.checkbox("キャッシュを有効化", value=True,
                                 help="一度処理したデータをキャッシュして再利用します")
        if not enable_cache:
            st.session_state.ai_keywords_cache = {}
    else:
        use_ai_extraction = False
        openai_api_key = None
        enable_cache = True
    
    if uploaded_file:
        # データ読み込みボタン
        if st.button("データを読み込む", type="primary"):
            with st.spinner("データを読み込み中..."):
                # CSVを読み込む
                try:
                    df = pd.read_csv(uploaded_file)
                except:
                    df = pd.read_csv(uploaded_file, encoding='shift_jis')
                
                # 位置情報を追加
                df = add_location_columns(df)
                st.session_state.df_with_locations = df
                st.session_state.df_loaded = True
                st.success("データの読み込みが完了しました！")

# メインコンテンツ
if st.session_state.df_loaded and st.session_state.df_with_locations is not None:
    df = st.session_state.df_with_locations
    
    # カスタムキーワード入力セクション
    st.header("🔍 キーワード設定")
    
    # 検索方法の説明
    with st.expander("検索方法の説明"):
        st.markdown("""
        **検索方法：**
        - **部分一致検索**: `食育` → 「食育」を含むすべての文字列にマッチ
        - **完全一致検索**: `"食育"` → 単語として独立した「食育」のみにマッチ
        - **AND検索**: `"食育" AND "教育"` → 両方のキーワードを含むページのみ
        - **OR検索**: `"食育" OR "給食"` → いずれかのキーワードを含むページ
        
        **例：**
        - `デジタル` → 「デジタル化」「デジタルトランスフォーメーション」なども含む
        - `"デジタル"` → 単語として独立した「デジタル」のみ
        - `"食育" AND "給食"` → 「食育」と「給食」の両方を含むページ
        - `防災 OR 減災` → 「防災」または「減災」を含むページ
        """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_keyword_input = st.text_input(
            "分析したいキーワードを入力",
            placeholder='例: デジタル化, "食育", "食育" AND "教育", 防災 OR 減災'
        )
    with col2:
        if st.button("キーワード追加", type="secondary"):
            if custom_keyword_input and custom_keyword_input not in st.session_state.custom_keywords:
                st.session_state.custom_keywords.append(custom_keyword_input)
                st.success(f"「{custom_keyword_input}」を追加しました")
            elif custom_keyword_input in st.session_state.custom_keywords:
                st.warning("このキーワードは既に追加されています")
    
    # カスタムキーワードリスト表示
    if st.session_state.custom_keywords:
        st.write("**登録されたキーワード:**")
        cols = st.columns(5)
        for i, kw in enumerate(st.session_state.custom_keywords):
            with cols[i % 5]:
                if st.button(f"❌ {kw}", key=f"del_{i}"):
                    st.session_state.custom_keywords.remove(kw)
                    st.rerun()
    
    st.markdown("---")
    
    # タブ
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 基本統計", "🔍 キーワード分析", "📈 時系列分析", "🔗 共起ネットワーク", "📑 データ詳細"])
    
    with tab1:
        st.header("基本統計")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("総文書数", f"{len(df):,}")
        with col2:
            st.metric("都道府県数", df['prefecture_name'].nunique())
        with col3:
            st.metric("市区町村数", df['municipality_name'].nunique())
        with col4:
            fiscal_year_min = df['fiscal_year'].min()
            fiscal_year_max = df['fiscal_year'].max()
            if pd.notna(fiscal_year_min) and pd.notna(fiscal_year_max):
                st.metric("年度範囲", f"{fiscal_year_min}～{fiscal_year_max}")
            else:
                st.metric("年度範囲", "データなし")
        
        # 都道府県別のfile_id数
        st.subheader("都道府県別の文書数")
        pref_counts = df.groupby('prefecture_name')['file_id'].nunique().sort_values(ascending=False)
        
        # DataFrameに変換
        pref_df = pd.DataFrame({
            '都道府県': pref_counts.index,
            '文書数': pref_counts.values
        })
        
        fig1 = px.bar(
            pref_df,
            x='文書数',
            y='都道府県',
            orientation='h',
            title="都道府県別ユニーク文書数"
        )
        fig1.update_layout(height=600)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 市区町村別のfile_id数
        st.subheader("市区町村別の文書数")
        
        # 表示件数の選択
        display_option = st.radio(
            "表示件数",
            options=["上位20件", "すべて表示"],
            horizontal=True,
            key="muni_display_option"
        )
        
        muni_counts = df.groupby('municipality_name')['file_id'].nunique().sort_values(ascending=False)
        
        if display_option == "上位20件":
            muni_counts = muni_counts.head(20)
            title_suffix = "（上位20）"
        else:
            title_suffix = "（全市区町村）"
        
        # DataFrameに変換
        muni_df = pd.DataFrame({
            '市区町村': muni_counts.index,
            '文書数': muni_counts.values
        })
        
        fig2 = px.bar(
            muni_df,
            x='文書数',
            y='市区町村',
            orientation='h',
            title=f"市区町村別ユニーク文書数{title_suffix}"
        )
        if display_option == "すべて表示":
            fig2.update_layout(height=max(600, len(muni_counts) * 20))
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("キーワード分析")
        
        if not st.session_state.custom_keywords:
            st.info("👆 上部でキーワードを追加してください")
        else:
            # 都道府県フィルタとキーワード選択
            col1, col2 = st.columns([1, 2])
            with col1:
                pref_filter = st.selectbox(
                    "都道府県でフィルタ",
                    options=['全国'] + sorted(df['prefecture_name'].dropna().unique().tolist()),
                    key="pref_filter_keyword"
                )
            
            with col2:
                selected_keyword = st.selectbox(
                    "分析するキーワードを選択",
                    options=st.session_state.custom_keywords,
                    index=0 if st.session_state.custom_keywords else None
                )
            
            if selected_keyword:
                # データをフィルタリング
                df_filtered_kw = df.copy()
                if pref_filter != '全国':
                    df_filtered_kw = df_filtered_kw[df_filtered_kw['prefecture_name'] == pref_filter]
                
                # キーワードの出現回数を計算
                df_with_keyword = count_keyword_occurrences(df_filtered_kw.copy(), selected_keyword)
                
                # 基本情報（検索タイプに応じて表示を変更）
                search_type, _ = parse_search_query(selected_keyword)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_occurrences = df_with_keyword['keyword_count'].sum()
                    st.metric("総出現回数", f"{total_occurrences:,}")
                with col2:
                    docs_with_keyword = (df_with_keyword['has_keyword']).sum()
                    st.metric("該当ページ数", f"{docs_with_keyword:,}")
                with col3:
                    st.metric("検索タイプ", search_type)
                
                if pref_filter == '全国':
                    # 都道府県別の出現回数
                    st.subheader(f"都道府県別「{selected_keyword}」出現回数")
                    pref_keyword_counts = df_with_keyword.groupby('prefecture_name')['keyword_count'].sum()
                    # 0より大きい値のみフィルタリング
                    pref_keyword_counts = pref_keyword_counts[pref_keyword_counts > 0].sort_values(ascending=False)
                    
                    if not pref_keyword_counts.empty:
                        # DataFrameに変換
                        pref_keyword_df = pd.DataFrame({
                            '都道府県': pref_keyword_counts.index,
                            '出現回数': pref_keyword_counts.values
                        })
                        
                        fig3 = px.bar(
                            pref_keyword_df,
                            x='出現回数',
                            y='都道府県',
                            orientation='h',
                            title=f"都道府県別「{selected_keyword}」出現回数"
                        )
                        fig3.update_layout(height=600)
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.info("該当するデータがありません。")
                
                # 市区町村別の出現回数とページ割合
                st.subheader(f"市区町村別「{selected_keyword}」出現状況")
                
                # 表示件数の選択
                display_option_kw = st.radio(
                    "表示件数",
                    options=["上位20件", "すべて表示"],
                    horizontal=True,
                    key="muni_display_option_kw"
                )
                
                # 市区町村別の集計データを作成
                muni_stats = df_with_keyword.groupby('municipality_name').agg({
                    'keyword_count': 'sum',  # キーワード出現回数の合計
                    'file_id': 'count'  # 総ページ数（レコード数）
                }).reset_index()
                
                # キーワードが出現するページ数を計算（has_keywordフラグを使用）
                muni_keyword_pages = df_with_keyword[df_with_keyword['has_keyword']].groupby('municipality_name').size().reset_index(name='pages_with_keyword')
                
                # データを結合
                muni_stats = muni_stats.merge(muni_keyword_pages, on='municipality_name', how='left')
                muni_stats['pages_with_keyword'] = muni_stats['pages_with_keyword'].fillna(0).astype(int)
                
                # ページ割合を計算
                muni_stats['page_ratio'] = (muni_stats['pages_with_keyword'] / muni_stats['file_id'] * 100).round(1)
                
                # 列名を変更
                muni_stats.columns = ['市区町村', 'キーワード出現回数', '総ページ数', 'キーワード含有ページ数', 'ページ割合(%)']
                
                # キーワード出現回数でソート
                muni_stats = muni_stats[muni_stats['キーワード出現回数'] > 0].sort_values('キーワード出現回数', ascending=False)
                
                if not muni_stats.empty:
                    if display_option_kw == "上位20件":
                        muni_stats_display = muni_stats.head(20)
                        title_suffix = "（上位20）"
                    else:
                        muni_stats_display = muni_stats
                        title_suffix = "（全市区町村）"
                    
                    # グラフ表示
                    fig4 = px.bar(
                        muni_stats_display,
                        x='キーワード出現回数',
                        y='市区町村',
                        orientation='h',
                        title=f"{'[' + pref_filter + '] ' if pref_filter != '全国' else ''}市区町村別「{selected_keyword}」出現回数{title_suffix}",
                        hover_data=['総ページ数', 'キーワード含有ページ数', 'ページ割合(%)']
                    )
                    if display_option_kw == "すべて表示":
                        fig4.update_layout(height=max(600, len(muni_stats_display) * 20))
                    st.plotly_chart(fig4, use_container_width=True)
                    
                    # 表形式でも表示
                    st.subheader("詳細データ")
                    st.dataframe(
                        muni_stats_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "キーワード出現回数": st.column_config.NumberColumn(format="%d"),
                            "総ページ数": st.column_config.NumberColumn(format="%d"),
                            "キーワード含有ページ数": st.column_config.NumberColumn(format="%d"),
                            "ページ割合(%)": st.column_config.NumberColumn(format="%.1f%%")
                        }
                    )
                else:
                    st.info("該当するデータがありません。")
    
    with tab3:
        st.header("時系列分析")
        
        if not st.session_state.custom_keywords:
            st.info("👆 上部でキーワードを追加してください")
        else:
            selected_keyword_ts = st.selectbox(
                "時系列分析するキーワードを選択",
                options=st.session_state.custom_keywords,
                index=0 if st.session_state.custom_keywords else None,
                key="keyword_ts"
            )
            
            if selected_keyword_ts:
                # データの再計算
                df_ts = count_keyword_occurrences(df.copy(), selected_keyword_ts)
                
                # デバッグ情報（検索タイプを表示）
                search_type_ts, _ = parse_search_query(selected_keyword_ts)
                total_count = df_ts['keyword_count'].sum()
                docs_with_keyword = df_ts['has_keyword'].sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("総出現回数", f"{total_count:,}")
                with col2:
                    st.metric("該当ページ数", f"{docs_with_keyword:,}")
                with col3:
                    st.metric("検索タイプ", search_type_ts)
                
                # 年度別の出現回数
                if total_count > 0:
                    # NaNを除外してグループ化
                    df_ts_valid = df_ts[df_ts['fiscal_year'].notna() & (df_ts['keyword_count'] > 0)]
                    
                    if len(df_ts_valid) > 0:
                        yearly_data = df_ts_valid.groupby('fiscal_year')['keyword_count'].sum()
                        
                        if len(yearly_data) > 0:
                            # DataFrameに変換
                            yearly_df = pd.DataFrame({
                                'fiscal_year': yearly_data.index,
                                'keyword_count': yearly_data.values
                            })
                            
                            # fiscal_yearが数値であることを確認
                            yearly_df['fiscal_year'] = pd.to_numeric(yearly_df['fiscal_year'], errors='coerce')
                            yearly_df = yearly_df.dropna(subset=['fiscal_year'])
                            
                            if len(yearly_df) > 0:
                                yearly_df['fiscal_year'] = yearly_df['fiscal_year'].astype(int)
                                yearly_df = yearly_df.sort_values('fiscal_year')
                                
                                # グラフ作成
                                fig5 = px.line(
                                    yearly_df,
                                    x='fiscal_year',
                                    y='keyword_count',
                                    labels={'fiscal_year': '年度', 'keyword_count': '出現回数'},
                                    title=f"「{selected_keyword_ts}」の年度別出現回数推移",
                                    markers=True
                                )
                                
                                # x軸の設定
                                fig5.update_xaxes(
                                    tickformat='d',
                                    dtick=1,
                                    tickmode='linear'
                                )
                                
                                st.plotly_chart(fig5, use_container_width=True)
                            else:
                                st.warning("有効な年度データがありません。")
                        else:
                            st.warning("年度別の集計データがありません。")
                    else:
                        st.warning("有効な年度データを持つレコードがありません。")
                else:
                    st.warning(f"「{selected_keyword_ts}」の出現データがありません。")
                
                st.markdown("---")
                
                # 都道府県選択
                selected_pref = st.selectbox(
                    "都道府県を選択（時系列詳細）",
                    options=['全国'] + sorted(df['prefecture_name'].dropna().unique().tolist())
                )
                
                if selected_pref != '全国':
                    # 選択した都道府県のデータをフィルタリング
                    df_pref = df_ts[df_ts['prefecture_name'] == selected_pref]
                    
                    # データがある場合のみ処理を続行
                    if len(df_pref) > 0:
                        pref_total = df_pref['keyword_count'].sum()
                        
                        if pref_total > 0:
                            # キーワードが出現している行のみ抽出
                            df_pref_with_keyword = df_pref[df_pref['keyword_count'] > 0]
                            
                            # 年度別・市区町村別の集計
                            muni_yearly = df_pref_with_keyword.groupby(['fiscal_year', 'municipality_name'])['keyword_count'].sum().reset_index()
                            
                            if len(muni_yearly) > 0:
                                # ピボットテーブルの作成
                                pivot_data = muni_yearly.pivot(
                                    index='fiscal_year',
                                    columns='municipality_name',
                                    values='keyword_count'
                                ).fillna(0)
                                
                                # 市区町村のリストを取得（出現回数の多い順）
                                muni_totals = pivot_data.sum().sort_values(ascending=False)
                                available_munis = muni_totals.index.tolist()
                                
                                if len(available_munis) > 0:
                                    # 市区町村選択（複数選択可能）
                                    st.subheader("表示する市区町村を選択")
                                    
                                    # デフォルトで上位5市区町村を選択
                                    default_munis = available_munis[:5]
                                    
                                    # 表示オプション
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        selected_munis = st.multiselect(
                                            "市区町村を選択（複数選択可）",
                                            options=available_munis,
                                            default=default_munis,
                                            help="グラフに表示する市区町村を選択してください"
                                        )
                                    with col2:
                                        if st.button("上位5件を選択", key="select_top5"):
                                            selected_munis = available_munis[:5]
                                            st.rerun()
                                        if st.button("すべて選択", key="select_all"):
                                            selected_munis = available_munis
                                            st.rerun()
                                    
                                    if selected_munis:
                                        # グラフ作成
                                        fig6 = go.Figure()
                                        
                                        for muni in selected_munis:
                                            if muni in pivot_data.columns:
                                                fig6.add_trace(go.Scatter(
                                                    x=pivot_data.index,
                                                    y=pivot_data[muni],
                                                    mode='lines+markers',
                                                    name=muni
                                                ))
                                        
                                        # タイトルを動的に変更
                                        if len(selected_munis) == len(available_munis):
                                            title_suffix = "（全市区町村）"
                                        elif len(selected_munis) <= 5:
                                            title_suffix = f"（{len(selected_munis)}市区町村）"
                                        else:
                                            title_suffix = f"（{len(selected_munis)}市区町村）"
                                        
                                        fig6.update_layout(
                                            title=f"{selected_pref}の市区町村別「{selected_keyword_ts}」出現回数推移{title_suffix}",
                                            xaxis_title="年度",
                                            yaxis_title="出現回数",
                                            hovermode='x unified',
                                            height=500
                                        )
                                        
                                        fig6.update_xaxes(
                                            tickformat='d',
                                            dtick=1,
                                            tickmode='linear'
                                        )
                                        
                                        st.plotly_chart(fig6, use_container_width=True)
                                        
                                        # 選択した市区町村の統計情報
                                        with st.expander("選択した市区町村の統計情報"):
                                            # 各市区町村の詳細データを計算
                                            stats_data = []
                                            
                                            # 選択した都道府県のデータで再計算
                                            df_pref_for_stats = df_ts[df_ts['prefecture_name'] == selected_pref]
                                            
                                            for muni in selected_munis:
                                                muni_data = df_pref_for_stats[df_pref_for_stats['municipality_name'] == muni]
                                                
                                                if len(muni_data) > 0:
                                                    # 総ページ数
                                                    total_pages = len(muni_data)
                                                    
                                                    # キーワード含有ページ数（has_keywordフラグを使用）
                                                    pages_with_kw = muni_data['has_keyword'].sum()
                                                    
                                                    # ページ割合
                                                    page_ratio = (pages_with_kw / total_pages * 100) if total_pages > 0 else 0
                                                    
                                                    # キーワード総出現回数
                                                    total_kw_count = muni_data['keyword_count'].sum()
                                                    
                                                    # 年度範囲
                                                    valid_years = pivot_data[pivot_data[muni] > 0].index if muni in pivot_data.columns else []
                                                    first_year = int(valid_years.min()) if len(valid_years) > 0 else '-'
                                                    last_year = int(valid_years.max()) if len(valid_years) > 0 else '-'
                                                    
                                                    stats_data.append({
                                                        '市区町村': muni,
                                                        'キーワード出現回数': int(total_kw_count),
                                                        '総ページ数': int(total_pages),
                                                        'キーワード含有ページ数': int(pages_with_kw),
                                                        'ページ割合(%)': round(page_ratio, 1),
                                                        '最初の年度': first_year,
                                                        '最後の年度': last_year
                                                    })
                                            
                                            if stats_data:
                                                stats_df = pd.DataFrame(stats_data)
                                                
                                                # キーワード出現回数でソート
                                                stats_df = stats_df.sort_values('キーワード出現回数', ascending=False)
                                                
                                                st.dataframe(
                                                    stats_df,
                                                    use_container_width=True,
                                                    hide_index=True,
                                                    column_config={
                                                        "キーワード出現回数": st.column_config.NumberColumn(format="%d"),
                                                        "総ページ数": st.column_config.NumberColumn(format="%d"),
                                                        "キーワード含有ページ数": st.column_config.NumberColumn(format="%d"),
                                                        "ページ割合(%)": st.column_config.NumberColumn(format="%.1f%%")
                                                    }
                                                )
                                    else:
                                        st.warning("表示する市区町村を選択してください。")
                                else:
                                    st.info(f"{selected_pref}の市区町村別データがありません。")
                            else:
                                st.info(f"{selected_pref}には「{selected_keyword_ts}」の年度別データがありません。")
                        else:
                            st.info(f"{selected_pref}には「{selected_keyword_ts}」の出現データがありません。")
                    else:
                        st.info(f"{selected_pref}のデータがありません。")
    
    with tab4:
        st.header("共起ネットワーク分析")
        
        st.markdown("""
        テキストを形態素解析して単語に分解し、同じページに出現する単語の共起関係をネットワークで可視化します。
        """)
        
        # AI抽出の説明
        if OPENAI_AVAILABLE:
            with st.expander("AI抽出機能について"):
                st.markdown("""
                **AIによるキーワード抽出を使用すると：**
                - OpenAI APIを使用して、各文書から特徴的で重要なキーワードを自動抽出します
                - 一般的な単語や行政用語を除外し、より特徴的な単語を抽出できます
                - 複合語（例：「地域活性化」「デジタル田園都市」）も適切に抽出されます
                - 処理時間は長くなりますが、より質の高い共起ネットワークが作成できます
                
                **通常の形態素解析では：**
                - MeCabまたは簡易的な形態素解析を使用します
                - 高速に処理できますが、一般的な単語も含まれる可能性があります
                """)
        
        # フィルタリング設定
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_pref_net = st.selectbox(
                "都道府県でフィルタ",
                options=['全国'] + sorted(df['prefecture_name'].dropna().unique().tolist()),
                key="filter_pref_net"
            )
        with col2:
            filter_year_net = st.selectbox(
                "年度でフィルタ",
                options=['すべて'] + sorted(df['fiscal_year'].dropna().unique().tolist()),
                key="filter_year_net"
            )
        with col3:
            filter_muni_net = st.selectbox(
                "市区町村でフィルタ",
                options=['すべて'] + (
                    sorted(df[df['prefecture_name'] == filter_pref_net]['municipality_name'].dropna().unique().tolist())
                    if filter_pref_net != '全国' else []
                ),
                key="filter_muni_net",
                disabled=(filter_pref_net == '全国')
            )
        
        # パラメータ設定
        st.subheader("ネットワークパラメータ")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_count = st.slider("最小出現回数", min_value=1, max_value=50, value=5, 
                                help="この回数以上出現する単語のみを表示")
        with col2:
            top_n_words = st.slider("表示する単語数", min_value=10, max_value=200, value=50, step=10,
                                  help="出現頻度の高い単語から上位N個を表示")
        with col3:
            layout_type = st.selectbox("レイアウト", 
                                      options=['spring', 'circular', 'kamada_kawai'],
                                      help="ネットワークの配置方法")
        
        # コミュニティ検出パラメータ
        st.subheader("コミュニティ検出設定")
        col1, col2 = st.columns(2)
        with col1:
            community_resolution = st.slider(
                "コミュニティ解像度",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="値が大きいほど小さなコミュニティに分割されます"
            )
        with col2:
            edge_threshold = st.slider(
                "エッジ表示閾値",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="弱い共起関係のエッジを非表示にします"
            )
        
        # AI使用時の追加設定
        if use_ai_extraction and openai_api_key:
            col1, col2 = st.columns(2)
            with col1:
                sample_size = st.number_input(
                    "サンプリング数（0=全データ）",
                    min_value=0,
                    max_value=len(df),
                    value=min(300, len(df)),
                    step=50,
                    help="AI処理のコスト削減のため、データをサンプリングします"
                )
            with col2:
                estimated_cost = (sample_size if sample_size > 0 else len(df)) * 0.004
                st.info(f"推定API料金: 約${estimated_cost:.2f}")
        else:
            sample_size = 0
        
        if st.button("共起ネットワークを生成", type="primary"):
            # データをフィルタリング
            df_filtered_net = df.copy()
            if filter_pref_net != '全国':
                df_filtered_net = df_filtered_net[df_filtered_net['prefecture_name'] == filter_pref_net]
            if filter_year_net != 'すべて':
                df_filtered_net = df_filtered_net[df_filtered_net['fiscal_year'] == filter_year_net]
            if filter_muni_net != 'すべて':
                df_filtered_net = df_filtered_net[df_filtered_net['municipality_name'] == filter_muni_net]
            
            if len(df_filtered_net) == 0:
                st.warning("フィルタ条件に該当するデータがありません。")
            else:
                # AI使用時のAPIキーチェック
                if use_ai_extraction and not openai_api_key:
                    st.error("OpenAI APIキーを入力してください。")
                else:
                    with st.spinner("共起ネットワークを生成中..."):
                        # 共起頻度を計算
                        word_counts, cooccurrence_data, top_words = calculate_cooccurrence(
                            df_filtered_net, 
                            min_count=min_count, 
                            top_n_words=top_n_words,
                            use_ai=use_ai_extraction,
                            api_key=openai_api_key,
                            sample_size=sample_size if use_ai_extraction else None
                        )
                        
                        if not cooccurrence_data:
                            st.warning("共起関係が見つかりませんでした。最小出現回数を下げてみてください。")
                        else:
                            # ネットワークを作成
                            traces, stats, G, partition = create_cooccurrence_network(
                                word_counts, 
                                cooccurrence_data, 
                                top_words, 
                                layout_type,
                                community_resolution,
                                edge_threshold
                            )
                            
                            # Plotlyでの可視化
                            fig_net = go.Figure(data=traces)
                            
                            fig_net.update_layout(
                                title=f"共起ネットワーク（{stats['num_nodes']}単語、{stats['num_edges']}共起関係、{stats['num_communities']}コミュニティ）",
                                showlegend=True,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=800,
                                plot_bgcolor='rgba(240,240,240,0.1)'
                            )
                            
                            st.plotly_chart(fig_net, use_container_width=True)
                            
                            # 統計情報
                            st.subheader("ネットワーク統計")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("ノード数", stats['num_nodes'])
                            with col2:
                                st.metric("エッジ数", stats['num_edges'])
                            with col3:
                                st.metric("コミュニティ数", stats['num_communities'])
                            with col4:
                                st.metric("平均次数", f"{stats['avg_degree']:.2f}")
                            with col5:
                                st.metric("密度", f"{stats['density']:.3f}")
                            
                            # コミュニティ別の単語リスト
                            with st.expander("コミュニティ別単語リスト"):
                                communities_dict = defaultdict(list)
                                for node, comm_id in partition.items():
                                    communities_dict[comm_id].append((node, word_counts[node]))
                                
                                for comm_id in sorted(communities_dict.keys()):
                                    members = sorted(communities_dict[comm_id], key=lambda x: x[1], reverse=True)
                                    st.write(f"**コミュニティ {comm_id}** ({len(members)}単語)")
                                    member_text = ", ".join([f"{word}({count})" for word, count in members[:10]])
                                    if len(members) > 10:
                                        member_text += f"... 他{len(members)-10}単語"
                                    st.write(member_text)
                                    st.write("")
                            
                            # 頻出単語ランキング
                            with st.expander("頻出単語ランキング（上位20）"):
                                # 重要度スコアを計算してソート
                                word_score_list = []
                                for word, count in word_counts.items():
                                    if word in top_words:
                                        score = calculate_word_importance_score(word, count, len(df_filtered_net))
                                        word_score_list.append((word, count, score))
                                
                                # スコアでソート
                                word_score_list.sort(key=lambda x: x[2], reverse=True)
                                
                                top_words_df = pd.DataFrame([
                                    {'単語': word, '出現回数': count, '重要度スコア': f"{score:.2f}"}
                                    for word, count, score in word_score_list[:20]
                                ])
                                st.dataframe(top_words_df, use_container_width=True, hide_index=True)
                            
                            # 共起頻度ランキング
                            with st.expander("共起頻度ランキング（上位20）"):
                                cooccur_df = pd.DataFrame([
                                    {'単語1': word1, '単語2': word2, '共起回数': count}
                                    for (word1, word2), count in sorted(cooccurrence_data.items(), key=lambda x: x[1], reverse=True)[:20]
                                ])
                                st.dataframe(cooccur_df, use_container_width=True, hide_index=True)
                            
                            # 使用した抽出方法を表示
                            extraction_method = "AI抽出" if use_ai_extraction else ("MeCab" if MECAB_AVAILABLE else "簡易形態素解析")
                            st.info(f"使用した抽出方法: {extraction_method}")
                            
                            # データを保存
                            st.session_state.cooccurrence_data = {
                                'word_counts': word_counts,
                                'cooccurrence': cooccurrence_data,
                                'top_words': top_words,
                                'graph': G
                            }
    
    with tab5:
        st.header("データ詳細")
        
        # フィルタリング
        col1, col2, col3 = st.columns(3)
        with col1:
            detail_pref = st.selectbox(
                "都道府県でフィルタ",
                options=['すべて'] + sorted(df['prefecture_name'].dropna().unique().tolist()),
                key="detail_pref"
            )
        with col2:
            detail_muni = st.selectbox(
                "市区町村でフィルタ",
                options=['すべて'] + (
                    sorted(df[df['prefecture_name'] == detail_pref]['municipality_name'].dropna().unique().tolist())
                    if detail_pref != 'すべて' else sorted(df['municipality_name'].dropna().unique().tolist())
                ),
                key="detail_muni"
            )
        with col3:
            detail_year = st.selectbox(
                "年度でフィルタ",
                options=['すべて'] + sorted(df['fiscal_year'].dropna().unique().tolist()),
                key="detail_year"
            )
        
        # データをフィルタリング
        df_detail = df.copy()
        if detail_pref != 'すべて':
            df_detail = df_detail[df_detail['prefecture_name'] == detail_pref]
        if detail_muni != 'すべて':
            df_detail = df_detail[df_detail['municipality_name'] == detail_muni]
        if detail_year != 'すべて':
            df_detail = df_detail[df_detail['fiscal_year'] == detail_year]
        
        # データ件数を表示
        st.info(f"フィルタ後のデータ件数: {len(df_detail):,}件")
        
        # 表示する列を選択
        all_columns = df_detail.columns.tolist()
        default_columns = ['prefecture_name', 'municipality_name', 'fiscal_year', 'title', 'content_text']
        display_columns = [col for col in default_columns if col in all_columns]
        
        selected_columns = st.multiselect(
            "表示する列を選択",
            options=all_columns,
            default=display_columns
        )
        
        if selected_columns:
            # データ表示
            st.dataframe(
                df_detail[selected_columns],
                use_container_width=True,
                height=600
            )
            
            # CSVダウンロード
            csv = df_detail[selected_columns].to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="フィルタ済みデータをCSVでダウンロード",
                data=csv,
                file_name=f"filtered_data_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("表示する列を選択してください。")

else:
    st.info("👆 サイドバーからCSVファイルをアップロードしてください。")
    
    # サンプルデータの説明
    with st.expander("必要なCSVファイルの形式"):
        st.markdown("""
        以下の列を含むCSVファイルが必要です：
        
        - **code**: 自治体コード（6桁）
        - **fiscal_year_start**: 年度開始日または年度
        - **file_id**: ファイルID（文書を識別）
        - **title**: タイトル（オプション）
        - **content_text**: 分析対象のテキスト内容
        
        その他の列があっても問題ありません。
        """)
        
        # サンプルデータ
        sample_data = pd.DataFrame({
            'code': ['131016', '141003', '271004'],
            'fiscal_year_start': ['2023-04-01', '2023-04-01', '2023-04-01'],
            'file_id': ['doc001', 'doc002', 'doc003'],
            'title': ['デジタル化推進計画', '食育推進計画', '防災計画'],
            'content_text': [
                'デジタル化により市民サービスの向上を図る。AIやIoTを活用した新しい行政サービスを展開。',
                '学校給食における食育の推進。地産地消を通じた食育教育の実施。',
                '防災・減災対策の強化。避難所の整備と防災訓練の実施。'
            ]
        })
        
        st.dataframe(sample_data)
        
        # サンプルCSVダウンロード
        sample_csv = sample_data.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="サンプルCSVをダウンロード",
            data=sample_csv,
            file_name="sample_data.csv",
            mime="text/csv"
        )
