import asyncio
import json
import os
import re
import httpx
from pathlib import Path
from typing import AsyncGenerator, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from supabase_client import supabase

# 既存モジュールのインポート
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
from components.recommend.base_line import BaseLineRecommend
from components.SelectCurrentTurnMove import SelectCurrentTurnMove
from components.llm_extract_user_preferences import ExtractUserPreferences
from components.extract_feature_hint import FeatureHintExtractor
from utils.load_embedding import load_feature_embedding_dict

# .env の読み込み
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Movie Recommendation API")

# CORS設定（Next.jsからのリクエストを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 【追加】重い計算・LLM処理が同時に何件も走ってサーバーがパンクするのを防ぐためのセマフォ（例: 最大3件まで同時実行）
heavy_task_semaphore = asyncio.Semaphore(3)


# Supabaseクエリをスレッドプールで非同期実行するためのヘルパー関数
# async def async_supabase_execute(query):
    # return await asyncio.to_thread(query.execute)

async def async_supabase_execute(query, retries=3, delay=0.5):
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(query.execute)
        except (httpx.ReadError, httpx.ConnectError) as e:
            if attempt == retries - 1:
                raise e
            await asyncio.sleep(delay * (2 ** attempt)) # 指数バックオフ


def fetch_all_movies_from_supabase():
    """Supabaseのmoviesテーブルからページネーションを使用して全件データを取得する（起動時同期処理）"""
    # print("Supabaseからmoviesテーブルのデータを取得中...")
    all_rows = []
    batch_size = 1000
    start = 0

    while True:
        end = start + batch_size - 1
        response = (
            supabase.table("movies").select("id, title").range(start, end).execute()
        )
        rows = response.data

        if not rows:
            break

        all_rows.extend(rows)
        print(f"  - {start} 〜 {start + len(rows) - 1} 件目のデータを取得しました")

        if len(rows) < batch_size:
            break

        start += batch_size

    print(f"合計 {len(all_rows)} 件のデータを取得しました。")
    return all_rows


def fetch_movies_to_dict():
    """全件データを取得し、IDをキーにした辞書に変換する"""
    all_rows = fetch_all_movies_from_supabase()

    if not all_rows:
        print("データが見つかりませんでした。")
        return {}

    movies_dict = {}
    for row in all_rows:
        title = row.get("title")
        movie_id = str(row.get("id"))
        if not (title or movie_id):
            continue

        movies_dict[movie_id] = {
            "movie_id": movie_id,
            "title": title,
        }

    return movies_dict


# AIモデルおよびマネージャーの初期化
llm_client = LLM(api_key)
embedding_client = Embeddings(api_key)
question_manager = QuestionManager(llm_client)

ease_model_path = "./components/recommend/model/ease_model_series_v5_four_1000.pkl"
parquet_path = "./parquet/movie_vectors_one_element.parquet"
movie_metadata = fetch_movies_to_dict()
content_base_predictor = ContentBasePredictor(parquet_path, embedding_client)
ease_predictor = EASEPredictor(ease_model_path)
current_move_selector = SelectCurrentTurnMove(llm_client)
pref_extractor = ExtractUserPreferences(llm_client)
llm_reranker = LLMReranker(llm_client)
recommend_manager = RecommendManager(ease_predictor, content_base_predictor, llm_reranker)
extractor = MovieFeatureExtractor(llm_client)
movie_feature_combiner = CommonFeatureExtractor(embedding_client)
mismatch_predictor = MismatchReasoning(llm_client, embedding_client, content_base_predictor.feature_embedding_dict)
un_sim_extractor = ExtractUnSimCluster(embedding_client)
un_mismatch_extractor = UnMismatchReasoning(llm_client)
base_line_recommender = BaseLineRecommend(llm_client)

hint_extractor = FeatureHintExtractor(content_base_predictor.feature_embedding_dict)


# 補助関数
def extract_recommend_title(recommend_response: str) -> set:
    """推薦出力テキストから『映画タイトル』を取り出して set で返す"""
    return set(re.findall(r"『(.*?)』", recommend_response))


def extract_recent_recommend_response(dialogue: list) -> Optional[str]:
    """対話履歴から直近の推薦応答（category='recommend'）を取得"""
    for dialogue_ele in reversed(dialogue):
        if dialogue_ele.get("role") == "system" and dialogue_ele.get("category") == "recommend":
            return dialogue_ele.get("content")
    return None


# --- リクエスト/レスポンススキーマ ---
class StartSessionRequest(BaseModel):
    user_id: str
    system_type: str = "A"


class InitQuestionRequest(BaseModel):
    session_id: str


class ChatMessageRequest(BaseModel):
    session_id: str
    user_input: str


# --- 1. セッション開始 API ---
@app.post("/api/session/start")
async def start_session(req: StartSessionRequest):
    user_res = await async_supabase_execute(
        supabase.table("users").select("*").eq("id", req.user_id)
    )
    if not user_res.data:
        raise HTTPException(status_code=404, detail="該当のuser_idは存在しません")

    system_type = req.system_type.upper()

    session_res = await async_supabase_execute(
        supabase.table("chat_sessions").insert({
            "user_id": req.user_id,
            "system_type": system_type,
            "short_term_like": [],
            "short_term_dislike": [],
            "already_recommend": [],
            "move_history": [],
        })
    )

    if not session_res.data:
        raise HTTPException(status_code=500, detail="セッションの作成に失敗しました")

    session_id = session_res.data[0]["id"]

    return {
        "session_id": session_id,
        "user_id": req.user_id,
        "system_type": system_type,
    }


# --- 2. 初期質問ストリーミング API ---
@app.post("/api/session/init-question")
async def stream_init_question(req: InitQuestionRequest):
    session_res = await async_supabase_execute(
        supabase.table("chat_sessions").select("*").eq("id", req.session_id)
    )
    if not session_res.data:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")

    session_row = session_res.data[0]
    user_id = session_row.get("user_id")
    system_type = str(session_row.get("system_type", "A")).upper()

    # システムB, Dの場合：定型挨拶テキストを出力
    if system_type in ["B", "D"]:
        greeting_text = "どんな映画が観たいか自由に入力してください！"

        async def event_generator():
            yield f"data: {json.dumps({'content': greeting_text})}\n\n"

            await async_supabase_execute(
                supabase.table("chat_messages").insert({
                    "session_id": req.session_id,
                    "role": "system",
                    "category": "first_question",
                    "content": greeting_text,
                })
            )

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # システムA, Cの場合：LLMにより初期質問を生成
    user_res = await async_supabase_execute(
        supabase.table("users").select("*").eq("id", user_id)
    )
    user_row = user_res.data[0]

    ratings_res = await async_supabase_execute(
        supabase.table("ratings").select("movie_id, title, rating").eq("user_id", user_id)
    )
    watch_record = ratings_res.data if ratings_res.data else []

    user_data = {
        "id": user_row.get("id"),
        "like_long_preference": user_row.get("long_term_like") or [],
        "dislike_long_preference": user_row.get("long_term_dislike") or [],
        "watch_record": watch_record,
    }

    context = QuestionContext(user_profile=user_data)

    # 初期質問生成の同期処理を別スレッドで取得（ここにもセマフォを適用したい場合は `async with heavy_task_semaphore:` を使えます）
    async with heavy_task_semaphore:
        first_stream = await asyncio.to_thread(
            question_manager.generate_question, "first", context
        )

    async def event_generator():
        full_res_text = ""
        for chunk in first_stream:
            full_res_text += chunk
            yield f"data: {json.dumps({'content': chunk})}\n\n"
            await asyncio.sleep(0)

        await async_supabase_execute(
            supabase.table("chat_messages").insert({
                "session_id": req.session_id,
                "role": "system",
                "category": "first_question",
                "content": full_res_text,
            })
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- 3. チャット API ---
@app.post("/api/chat")
async def chat_endpoint(req: ChatMessageRequest):
    session_id = req.session_id
    user_input = req.user_input

    # 1. セッションデータの取得
    sess_res = await async_supabase_execute(
        supabase.table("chat_sessions").select("*").eq("id", session_id)
    )
    if not sess_res.data:
        raise HTTPException(status_code=404, detail="セッションが見つかりませんでした")

    session_data = sess_res.data[0]
    user_id = session_data["user_id"]
    system_type = str(session_data.get("system_type", "A")).upper()

    # 2. ユーザー情報・視聴履歴の非同期取得
    user_res_task = async_supabase_execute(supabase.table("users").select("*").eq("id", user_id))
    ratings_res_task = async_supabase_execute(
        supabase.table("ratings").select("movie_id, title, rating").eq("user_id", user_id)
    )
    msg_res_task = async_supabase_execute(
        supabase.table("chat_messages").select("*").eq("session_id", session_id).order("created_at")
    )

    user_res, ratings_res, msg_res = await asyncio.gather(
        user_res_task, ratings_res_task, msg_res_task
    )

    if not user_res.data:
        raise HTTPException(status_code=404, detail="ユーザー情報が見つかりませんでした")
    user_row = user_res.data[0]

    watch_record = ratings_res.data if ratings_res.data else []

    dialogue_list = [
        {
            "role": m["role"],
            "category": m.get("category"),
            "content": m["content"],
        }
        for m in msg_res.data
    ]

    if system_type in ["B", "D"]:
        long_term_like = []
        long_term_dislike = []
    else:
        long_term_like = user_row.get("long_term_like") or []
        long_term_dislike = user_row.get("long_term_dislike") or []

    user_data = {
        "id": user_id,
        "like_long_preference": long_term_like,
        "dislike_long_preference": long_term_dislike,
        "watch_record": watch_record,
        "short_term": {
            "like": session_data.get("short_term_like", []),
            "dislike": session_data.get("short_term_dislike", []),
        },
    }

    already_recommend = set(session_data.get("already_recommend") or [])
    move_history = session_data.get("move_history") or []
    mismatch_hints = session_data.get("mismatch_hint", [])

    context = QuestionContext(user_profile=user_data)

    stream_generator = None
    current_turn_move = None
    stage1_candidate = []
    short_term_like_vec = None

    # --- 【セマフォ適用】重い処理のブロック全体をセマフォで保護 ---
    async with heavy_task_semaphore:
        # システムD（ベースライン）
        if system_type == "D":
            stream_generator = await asyncio.to_thread(
                base_line_recommender.generate_response, user_input, dialogue_list
            )

        # システムA, B, C
        else:
            current_turn_move = await asyncio.to_thread(
                current_move_selector.select_move, user_input, dialogue_list, move_history
            )
            move_history.append(current_turn_move)

            if current_turn_move == "recommend":
                result_preferences = await asyncio.to_thread(
                    pref_extractor.extract_user_preferences,
                    user_input,
                    dialogue_list,
                    context.short_term_like,
                    context.short_term_dislike,
                )
                context.update_short_term_preference(result_preferences)

                raw_stream_generator, stage1_candidate, short_term_like_vec = await asyncio.to_thread(
                    recommend_manager.recommend,
                    watch_record,
                    already_recommend,
                    context.short_term_like,
                    context.short_term_dislike,
                    long_term_like,
                    long_term_dislike,
                )
                stream_generator = replace_id_to_title_stream(raw_stream_generator, movie_metadata)

            elif current_turn_move == "mismatch_question":
                if system_type == "C":
                    stream_generator = await asyncio.to_thread(
                        question_manager.generate_question, "fallback", context
                    )
                else:
                    recent_rec = extract_recent_recommend_response(dialogue_list)

                    def run_mismatch_pipeline():
                        features_dict = extractor.extract(recent_rec)
                        common_features = movie_feature_combiner.extract_common_features(features_dict)
                        mismatch_reasoning_result = mismatch_predictor.mismatch_reasoning_matched_watched_record_feature(
                            watch_record,
                            features_dict,
                            context.short_term_like,
                            context.short_term_dislike,
                            long_term_like,
                            long_term_dislike,
                            mismatch_hints,
                        )
                        # mismatch_reasoning_result = mismatch_predictor.mismatch_reasoning_matched_watched_record_feature_no_hint(
                        #     watch_record,
                        #     features_dict,
                        #     context.short_term_like,
                        #     context.short_term_dislike,
                        #     long_term_like,
                        #     long_term_dislike,
                        # )
                        return features_dict, common_features, mismatch_reasoning_result

                    features_dict, common_features, mismatch_reasoning_result = await asyncio.to_thread(
                        run_mismatch_pipeline
                    )
                    # print(mismatch_reasoning_result)
                    is_mismatch = mismatch_reasoning_result.get("status") == "mismatch"
                    if is_mismatch:
                        mismatch_content = mismatch_reasoning_result.get("content")
                        context.mismatch_results = mismatch_content
                        stream_generator = await asyncio.to_thread(
                            question_manager.generate_question, "mismatch", context
                        )
                    else:
                        def run_unmismatch_pipeline():
                            un_sim_cluster = un_sim_extractor.extract(common_features, context.short_term_like)
                            return un_mismatch_extractor.reasoning(
                                un_sim_cluster,
                                context.short_term_like,
                                context.short_term_dislike,
                                long_term_like,
                                long_term_dislike,
                            )

                        un_mismatch_result = await asyncio.to_thread(run_unmismatch_pipeline)
                        is_unmismatch = un_mismatch_result.get("status") == "unmismatch"

                        if is_unmismatch:
                            context.un_mismatch_results = un_mismatch_result
                            stream_generator = await asyncio.to_thread(
                                question_manager.generate_question, "not_mismatch", context
                            )
                        else:
                            stream_generator = await asyncio.to_thread(
                                question_manager.generate_question, "fallback", context
                            )
            else:
                stream_generator = await asyncio.to_thread(
                    question_manager.generate_question, "fallback", context
                )

    # 4. SSEイベントストリームの非同期生成
    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal mismatch_hints
        full_system_res = ""

        for chunk in stream_generator:
            full_system_res += chunk
            yield f"data: {json.dumps({'content': chunk})}\n\n"
            await asyncio.sleep(0)

        if current_turn_move == "recommend":
            rec_movie_set = extract_recommend_title(full_system_res)
            already_recommend.update(rec_movie_set)

            movie_ids = [
                movie["movie_id"]
                for movie in stage1_candidate
                if movie["title"] not in rec_movie_set
            ]

            mismatch_hints = await asyncio.to_thread(
                hint_extractor.extract_hints,
                movie_ids,
                short_term_like_vec,
                context.short_term_like,
                20,
                0.40,
                0.70,
                0.60,
            )
            # print(mismatch_hints)

        await async_supabase_execute(
            supabase.table("chat_sessions").update({
                "short_term_like": context.short_term_like,
                "short_term_dislike": context.short_term_dislike,
                "already_recommend": list(already_recommend),
                "move_history": move_history,
                "mismatch_hint": mismatch_hints,
            }).eq("id", session_id)
        )

        await async_supabase_execute(
            supabase.table("chat_messages").insert({
                "session_id": session_id,
                "role": "user",
                "category": None,
                "content": user_input,
            })
        )

        await async_supabase_execute(
            supabase.table("chat_messages").insert({
                "session_id": session_id,
                "role": "system",
                "category": current_turn_move,
                "content": full_system_res,
            })
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")