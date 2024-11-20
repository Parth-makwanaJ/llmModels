import spacy
nlp = spacy.load("en_core_web_sm")
article = """Owning these bottles can be a great experience..."""
doc = nlp(article)
keywords = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
print(keywords)
