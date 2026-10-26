import json, re
def parse_json_response(response):
    try:
        # そのままパースできるか試す
        return json.loads(response)
    except json.JSONDecodeError:
        # ```json から ``` の中身を抽出を試みる
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 波括弧 {} で囲まれた部分だけでも抽出を試みる
        match = re.search(r'(\{.*?\})', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
                
        # どうしてもパースできない場合のエラーハンドリング
        print(f"Failed to parse JSON response: {response}")
        raise