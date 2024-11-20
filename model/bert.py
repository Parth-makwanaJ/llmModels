from keybert import KeyBERT

def extract_keywords_bert_(article):
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    keywords = kw_model.extract_keywords(article, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    return keywords

# print(extract_keywords_bert("Owning these bottles can be a great experience..."))
