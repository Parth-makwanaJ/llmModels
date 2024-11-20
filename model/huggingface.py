from langchain_huggingface import HuggingFaceEndpoint

def keyword_fetch_(endpoint_url, secretkey, Keywords):
    llm = HuggingFaceEndpoint(endpoint_url=endpoint_url, huggingfacehub_api_token=secretkey)
    return llm.invoke(f"suggest keywords based on this sentence : {Keywords}")

# repo_id="Qwen/Qwen2.5-Coder-32B-Instruct"
# sec_key="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"
# artical = """ article """
# print(keyword_fetch_(repo_id, sec_key, artical))
