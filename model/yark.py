from yake import KeywordExtractor

def extract_keywords_yake_(article):
    kw_extractor = KeywordExtractor(lan="en", n=2, dedupLim=0.9)
    keywords = kw_extractor.extract_keywords(article)
    return keywords
# print(keywords)
