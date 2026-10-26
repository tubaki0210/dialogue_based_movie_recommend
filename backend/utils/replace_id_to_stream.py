import re
def replace_id_to_title_stream(stream_generator, movie_data):
    """
    1文字ずつのループを排出し、正規表現で一括置換する超高速ジェネレータ
    """
    # ID抽出用正規表現
    pattern = re.compile(r'『(\d+)』')

    def get_title(match):
        movie_id = match.group(1)
        movie_info = movie_data.get(movie_id) or movie_data.get(int(movie_id) if movie_id.isdigit() else None)
        if isinstance(movie_info, dict) and "title" in movie_info:
            return f"『{movie_info['title']}』"
        return match.group(0)

    buffer = ""
    for chunk in stream_generator:
        text = chunk if isinstance(chunk, str) else getattr(chunk, 'content', str(chunk))
        if not text:
            continue
            
        buffer += text
        
        # 『 が閉じられている部分までを一括置換して yield
        if '』' in buffer:
            last_close = buffer.rfind('』')
            processable = buffer[:last_close + 1]
            buffer = buffer[last_close + 1:]
            
            # 正規表現で一括置換して即座に出力
            yield pattern.sub(get_title, processable)
        elif '『' not in buffer:
            # 『 が含まれない安全なテキストはそのまま即座に出力
            yield buffer
            buffer = ""

    if buffer:
        yield pattern.sub(get_title, buffer)
