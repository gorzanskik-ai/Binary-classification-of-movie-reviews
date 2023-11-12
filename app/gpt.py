from openai import OpenAI


client = OpenAI(api_key=r'API-KEY')


def generate_review(prompt='Write me a professional movie review'):
    response = client.completions.create(
        model='gpt-3.5-turbo-instruct',
        prompt=prompt,
        max_tokens=150,
        temperature=0.71)

    return response.choices[0].text
