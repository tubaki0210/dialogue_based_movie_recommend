import re
def replace_id_to_title_stream(stream_generator, movie_data):
    """
    ストリーミングテキスト内の『ID』を判定し、movie_data からタイトルを取得して置換するジェネレータ
    """
    buffer = ""
    in_bracket = False

    for chunk in stream_generator:
        # chunkが文字列でないオブジェクトの場合（chunk.contentなど）にも対応
        text = chunk if isinstance(chunk, str) else getattr(chunk, 'content', str(chunk))
        if not text:
            continue

        for char in text:
            if char == '『':
                in_bracket = True
                buffer = '『'
            elif char == '』' and in_bracket:
                buffer += '』'
                in_bracket = False
                
                # 『数字ID』の形式からIDを抽出
                match = re.match(r'『(\d+)』', buffer)
                if match:
                    movie_id = match.group(1)
                    
                    # movie_data[movie_id].get("title") でタイトルを取得
                    # (IDがint型で保持されている可能性も考慮して str/int 両方で判定)
                    movie_info = movie_data.get(movie_id) or movie_data.get(int(movie_id) if movie_id.isdigit() else None)
                    
                    title = movie_info.get("title") if isinstance(movie_info, dict) else None
                    
                    if title:
                        yield f"『{title}』"
                    else:
                        yield buffer  # タイトルが見つからない場合は元の『ID』をそのまま出す
                else:
                    yield buffer  # ID数字以外が入っていた場合はそのまま出す
                buffer = ""
            elif in_bracket:
                buffer += char
                # 『 の後に数字以外が入ってきた場合や長すぎる場合の安全策（バッファ溢れ防止）
                if len(buffer) > 20 or (len(buffer) > 1 and not buffer[1:].isdigit()):
                    yield buffer
                    buffer = ""
                    in_bracket = False
            else:
                yield char

    # ストリーミング終了時にバッファに文字が残っていれば出力
    if buffer:
        yield buffer