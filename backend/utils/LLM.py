from openai import OpenAI

class LLM:
    def __init__(self, key):
        self.key = key
        self.client = OpenAI(api_key=key)

    def generate_response(self, role, prompt, model="gpt-4.1-mini-2025-04-14", temperature=0.5):
        response = self.client.responses.create(
            model = model,
            input = [
                {"role" : "system", "content" : role},
                {"role" : "user", "content"  : prompt}
            ],
            # temperature=temperature
        )
        return response.output_text.replace('　','').replace(' ','')

    def generate_response_stream(self,role, prompt, model="gpt-4.1-mini-2025-04-14" ,temperature=0.5):
        response = self.client.responses.create(
            model = model,
            input = [
                {"role" : "system", "content" : role},
                {"role" : "user", "content"  : prompt}
            ],
            stream=True,
            # temperature=temperature
        )
        for chunk in response:
            # chat.completions の応答からテキストコンテンツを取得
            if chunk.type == 'response.output_text.delta' and chunk.delta:
                yield chunk.delta