import json
import os
import re
from pathlib import Path
from typing import AsyncGenerator, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from supabase_client import supabase

# 既存モジュールのインポート（環境に合わせてパスを微調整してください）
from utils.LLM import LLM
from utils.Embedding import Embeddings
from utils.replace_id_to_stream import replace_id_to_title_stream
from components.generate_question.QuestionManager import QuestionManager
from components.generate_question.QuestionContext import QuestionContext
from components.mismatch_reasoning.MismatchReasoning import MismatchReasoning
from components.mismatch_reasoning.UnMismatchReasoning import UnMismatchReasoning
from components.mismatch_reasoning.CombineMovieFeature import CommonFeatureExtractor
from components.mismatch_reasoning.ExtractUnSimCluster import ExtractUnSimCluster
from components.mismatch_reasoning.MovieFeatureExtractor import MovieFeatureExtractor
from components.recommend.recommend import RecommendManager
from components.recommend.content_base import ContentBasePredictor
from components.recommend.llm_recommend import LLMReranker
from components.recommend.ease_recommend_main import EASEPredictor
from components.SelectCurrentTurnMove import SelectCurrentTurnMove
from components.llm_extract_user_preferences import ExtractUserPreferences

# .env の読み込み
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Movie Recommendation API")

# CORS設定（Next.jsからのリクエストを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://dialogue-based-movie-recommend-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_all_movies_from_supabase():
  """Supabaseのmoviesテーブルからページネーションを使用して全件データを取得する"""
  print("Supabaseからmoviesテーブルのデータを取得中...")
  all_rows = []
  batch_size = 1000  # 1回に取得する上限件数
  start = 0

  while True:
    end = start + batch_size - 1
    response = (
        supabase.table("movies").select("*").range(start, end).execute()
    )
    rows = response.data

    if not rows:
      break

    all_rows.extend(rows)
    print(
        f"  - {start} 〜 {start + len(rows) - 1}"
        " 件目のデータを取得しました"
    )

    # 取得件数がバッチサイズ未満なら最後のページとみなして終了
    if len(rows) < batch_size:
      break

    start += batch_size

  print(f"合計 {len(all_rows)} 件のデータを取得しました。")
  return all_rows


def fetch_movies_to_dict():
  """全件データを取得し、タイトルをキーにした辞書に変換する"""
  # 切り出した関数を呼び出して全行データを取得
  all_rows = fetch_all_movies_from_supabase()

  if not all_rows:
    print("データが見つかりませんでした。")
    return {}

  # タイトルをキーにした辞書に変換
  movies_dict = {}
  for row in all_rows:
    title = row.get("title")
    movie_id = str(row.get("id"))
    if not (title or movie_id):
      continue

    movies_dict[movie_id] = {
        "movie_id": movie_id,  # DB側のカラム名が id の場合
        "title": title,
        "genre": row.get("genre"),  # jsonbや文字列
        "year": row.get("year"),
        "length": row.get("length"),
        "director": row.get("director"),
        "script": row.get("script"),
        "actor": row.get("actor"),
        "summary": row.get("summary"),
        "features": row.get("features", []),
        "style": row.get("style"),
    }

  return movies_dict

# AIモデルおよびマネージャーの初期化（サーバー起動時に一括ロード）
llm_client = LLM(api_key)
embedding_client = Embeddings(api_key)
question_manager = QuestionManager(llm_client)

ease_model_path = "./components/recommend/model/ease_model_v1.pkl"
parquet_path = "./parquet/movie_vectors_one_element.parquet"
movie_metadata = fetch_movies_to_dict()
content_base_predictor = ContentBasePredictor(parquet_path, embedding_client)
ease_predictor = EASEPredictor(ease_model_path)
current_move_selector = SelectCurrentTurnMove(llm_client)
pref_extractor = ExtractUserPreferences(llm_client)
llm_reranker = LLMReranker(movie_metadata, llm_client)
recommend_manager = RecommendManager(ease_predictor, content_base_predictor, llm_reranker)
extractor = MovieFeatureExtractor(llm_client)
movie_feature_combiner = CommonFeatureExtractor(embedding_client)
mismatch_predictor = MismatchReasoning(llm_client)
un_sim_extractor = ExtractUnSimCluster(embedding_client)
un_mismatch_extractor = UnMismatchReasoning(llm_client)




# 補助関数
def extract_recommend_title(recommend_response: str) -> set:
    """推薦出力テキストから『映画タイトル』を取り出して set で返す"""
    return set(re.findall(r'『(.*?)』', recommend_response))


def extract_recent_recommend_response(dialogue: list) -> Optional[str]:
    """対話履歴から直近の推薦応答（category='recommend'）を取得"""
    for dialogue_ele in reversed(dialogue):
        if dialogue_ele.get("role") == "system" and dialogue_ele.get("category") == "recommend":
            return dialogue_ele.get("content")
    return None


# --- リクエスト/レスポンススキーマ ---
class StartSessionRequest(BaseModel):
    user_id: str  # users.id (varchar)

class InitQuestionRequest(BaseModel):
    session_id: str


class ChatMessageRequest(BaseModel):
    session_id: str  # chat_sessions.id (uuid)
    user_input: str


# --- 1. セッション開始 API（高速・DB処理のみ） ---
@app.post("/api/session/start")
def start_session(req: StartSessionRequest):
    user_res = supabase.table("users").select("*").eq("id", req.user_id).execute()
    if not user_res.data:
        raise HTTPException(status_code=404, detail="該当のuser_idは存在しません")
    
    # 新規チャットセッションを作成（LLM生成は待たずに即返却）
    session_res = supabase.table("chat_sessions").insert({
        "user_id": req.user_id,
        "short_term_like": [],
        "short_term_dislike": [],
        "already_recommend": [],
        "move_history": []
    }).execute()
    
    if not session_res.data:
        raise HTTPException(status_code=500, detail="セッションの作成に失敗しました")
        
    session_id = session_res.data[0]["id"]

    return {
        "session_id": session_id,
        "user_id": req.user_id
    }

# --- 2. 初期質問ストリーミング API（LLM生成） ---
@app.post("/api/session/init-question")
def stream_init_question(req: InitQuestionRequest):
    # セッション情報の取得
    session_res = supabase.table("chat_sessions").select("*").eq("id", req.session_id).execute()
    if not session_res.data:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    
    session_row = session_res.data[0]
    user_id = session_row.get("user_id")

    # ユーザー長期嗜好・履歴の取得
    user_row = supabase.table("users").select("*").eq("id", user_id).execute().data[0]
    ratings_res = supabase.table("ratings").select("movie_id, title, rating").eq("user_id", user_id).execute()
    watch_record = ratings_res.data if ratings_res.data else []

    user_data = {
        "id": user_row.get("id"),
        "like_long_preference": user_row.get("long_term_like") or [],
        "dislike_long_preference": user_row.get("long_term_dislike") or [],
        "watch_record": watch_record
    }

    # 初期質問の生成ストリーム
    context = QuestionContext(user_profile=user_data)
    first_stream = question_manager.generate_question("first", context)

    async def event_generator():
        full_res_text = ""
        for chunk in first_stream:
            full_res_text += chunk
            yield f"data: {json.dumps({'content': chunk})}\n\n"

        # 生成完了後にDB保存
        supabase.table("chat_messages").insert({
            "session_id": req.session_id,
            "role": "system",
            "category": "first_question",
            "content": full_res_text
        }).execute()

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# チャット
@app.post("/api/chat")
async def chat_endpoint(req: ChatMessageRequest):
    session_id = req.session_id
    user_input = req.user_input

    # 1. セッションデータの取得
    sess_res = supabase.table("chat_sessions").select("*").eq("id", session_id).execute()
    if not sess_res.data:
        raise HTTPException(status_code=404, detail="セッションが見つかりませんでした")
    
    session_data = sess_res.data[0]
    user_id = session_data["user_id"]

    # 2. ユーザーの長期嗜好と視聴履歴を取得
    user_res = supabase.table("users").select("*").eq("id", user_id).execute()
    if not user_res.data:
        raise HTTPException(status_code=404, detail="ユーザー情報が見つかりませんでした")
    user_row = user_res.data[0]

    ratings_res = supabase.table("ratings").select("movie_id, title, rating").eq("user_id", user_id).execute()
    watch_record = ratings_res.data if ratings_res.data else []


    # 3. chat_messages から過去の対話履歴を取得
    msg_res = supabase.table("chat_messages").select("*").eq("session_id", session_id).order("created_at").execute()
    dialogue_list = [
        {
            "role": m["role"],
            "category": m.get("category"),
            "content": m["content"]
        }
        for m in msg_res.data
    ]

    session_record = supabase.table("chat_sessions").select("*").eq("id", session_id).execute()
    session_row = session_record.data[0]

    user_data = {
        "id": user_id,
        "like_long_preference": user_row.get("long_term_like") or [],
        "dislike_long_preference": user_row.get("long_term_dislike") or [],
        "watch_record": watch_record,
        "short_term" : { "like" : session_row.get("short_term_like", []), 
                        "dislike" : session_row.get("short_term_dislike", [])
                        }
    }

    long_term_like = user_data["like_long_preference"]
    long_term_dislike = user_data["dislike_long_preference"]



    # 4. 今回のユーザー発言を dialogue_list に追加 ＆ chat_messages へ保存
    dialogue_list.append({"role": "user", "content": user_input})
    supabase.table("chat_messages").insert({
        "session_id": session_id,
        "role": "user",
        "category": None,
        "content": user_input
    }).execute()

    # 5. セッション状態の復元
    already_recommend = set(session_data.get("already_recommend") or [])
    move_history = session_data.get("move_history") or []

    context = QuestionContext(user_profile=user_data)
    # context.short_term_like = session_data.get("short_term_like") or []
    # context.short_term_dislike = session_data.get("short_term_dislike") or []

    # 6. 今ターンのアクション (Current Turn Move) を選定
    current_turn_move = current_move_selector.select_move(dialogue_list, move_history)
    move_history.append(current_turn_move)

    # 7. Moveに応じた処理分岐とジェネレータの取得
    stream_generator = None

    if current_turn_move == "recommend":
        # 嗜好抽出
        result_preferences = pref_extractor.extract_user_preferences(
            user_input,
            dialogue_list,
            context.short_term_like,
            context.short_term_dislike
        )
        context.update_short_term_preference(result_preferences)

        raw_stream_generator = recommend_manager.recommend(
                    watch_record,
                    already_recommend,
                    context.short_term_like,
                    context.short_term_dislike,
                    long_term_like,
                    long_term_dislike
                )
        # 【追加】recommend の場合のみ『ID』を『タイトル』に置換するジェネレータでラップ
        stream_generator = replace_id_to_title_stream(raw_stream_generator, movie_metadata)

    elif current_turn_move == "mismatch_question":
        recent_rec = extract_recent_recommend_response(dialogue_list)
        features_dict = extractor.extract(recent_rec)
        common_features = movie_feature_combiner.extract_common_features(features_dict)

        mismatch_reasoning_result = mismatch_predictor.infer_mismatch(
            common_features,
            context.short_term_like,
            context.short_term_dislike,
            long_term_like,
            long_term_dislike,
            recent_recommend_res=recent_rec
        )

        is_mismatch = mismatch_reasoning_result.get("status") == "mismatch"
        if is_mismatch:
            mismatch_content = mismatch_reasoning_result.get("content")
            context.mismatch_results = mismatch_content
            stream_generator = question_manager.generate_question("mismatch", context)
        else:
            un_sim_cluster = un_sim_extractor.extract(common_features, context.short_term_like)
            un_mismatch_result = un_mismatch_extractor.reasoning(
                un_sim_cluster,
                context.short_term_like,
                context.short_term_dislike,
                long_term_like,
                long_term_dislike
            )
            is_unmismatch = un_mismatch_result.get("status") == "unmismatch"
            if is_unmismatch:
                context.un_mismatch_results = un_mismatch_result
                stream_generator = question_manager.generate_question("not_mismatch", context)
            else:
                stream_generator = question_manager.generate_question("fallback", context)
    else:
        # fallback等
        stream_generator = question_manager.generate_question("fallback", context)

    # 8. SSEイベントストリームの生成関数
    async def event_generator() -> AsyncGenerator[str, None]:
        full_system_res = ""
        
        # LLMの出力を1チャンクずつフロントエンドへストリーミング送信
        for chunk in stream_generator:
            full_system_res += chunk
            yield f"data: {json.dumps({'content': chunk})}\n\n"

        # ストリーミング完了後：推薦映画があれば推薦済みセットを更新
        if current_turn_move == "recommend":
            rec_movie_set = extract_recommend_title(full_system_res)
            already_recommend.update(rec_movie_set)

        # Supabaseの chat_sessions テーブル状態を更新
        supabase.table("chat_sessions").update({
            "short_term_like": context.short_term_like,
            "short_term_dislike": context.short_term_dislike,
            "already_recommend": list(already_recommend),
            "move_history": move_history
        }).eq("id", session_id).execute()

        # システム応答を chat_messages テーブルへ保存
        supabase.table("chat_messages").insert({
            "session_id": session_id,
            "role": "system",
            "category": current_turn_move,
            "content": full_system_res
        }).execute()

        # ストリーム終了合図
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")