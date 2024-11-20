from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from keybert import KeyBERT
from rake_nltk import Rake
import yake
import spacy
from collections import Counter
from string import punctuation
import nltk
from nltk.corpus import stopwords
import indicnlp.loader
from indicnlp.tokenize import sentence_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import re

class MultilingualKeywordGenerator:
    def __init__(self):
        # Initialize NLTK
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('universal_tagset')
        
        # Initialize Indic NLP Library
        indicnlp.loader.load()
        
        # Initialize models for different languages
        self.keybert_model = KeyBERT('ai4bharat/indic-bert')
        self.rake = Rake()
        
        # Load spaCy models
        self.nlp_en = spacy.load('en_core_web_sm')
        
        # Initialize normalizers for Indian languages
        self.normalizer_factory = IndicNormalizerFactory()
        self.normalizers = {
            'hi': self.normalizer_factory.get_normalizer('hi'),
            'gu': self.normalizer_factory.get_normalizer('gu'),
            'mr': self.normalizer_factory.get_normalizer('mr')
        }
        
        # Stopwords for Indian languages
        self.stopwords = {
            'en': set(stopwords.words('english')),
            'hi': set([
                'का', 'के', 'की', 'है', 'में', 'से', 'को', 'पर', 'इस', 'और',
                'यह', 'हैं', 'था', 'थे', 'थी', 'जो', 'कि', 'वह', 'बहुत', 'कर'
            ]),
            'gu': set([
                'છે', 'અને', 'તે', 'ના', 'માં', 'થી', 'ને', 'નું', 'નો', 'ની',
                'હતું', 'હતા', 'હતી', 'પણ', 'એક', 'શું', 'કે', 'જે', 'આ', 'તો'
            ]),
            'mr': set([
                'आहे', 'आणि', 'ते', 'च्या', 'मध्ये', 'ला', 'ची', 'चे', 'तो', 'ती',
                'होते', 'होता', 'होती', 'एक', 'काय', 'की', 'जे', 'हे', 'या', 'तर'
            ])
        }

    def detect_language(self, text):
        """Detect the language of the text"""
        # Simple script-based detection
        devanagari = len(re.findall(r'[\u0900-\u097F]', text))
        gujarati = len(re.findall(r'[\u0A80-\u0AFF]', text))
        marathi = len(re.findall(r'[\u0900-\u097F]', text))  # Uses Devanagari
        
        if gujarati > devanagari:
            return 'gu'
        elif marathi > 0 and devanagari > gujarati:
            # Additional check for Marathi-specific characters
            marathi_specific = len(re.findall(r'[ळऴ]', text))
            return 'mr' if marathi_specific > 0 else 'hi'
        elif devanagari > 0:
            return 'hi'
        return 'en'

    def normalize_text(self, text, language):
        """Normalize text based on language"""
        if language in self.normalizers:
            return self.normalizers[language].normalize(text)
        return text

    def extract_keywords_indic(self, text, language):
        """Extract keywords for Indian languages"""
        # Normalize text
        normalized_text = self.normalize_text(text, language)
        
        # Tokenize into sentences
        sentences = sentence_tokenize.sentence_split(normalized_text, lang=language)
        
        # Extract words and remove stopwords
        words = []
        for sentence in sentences:
            # Simple word tokenization based on spaces
            sentence_words = sentence.split()
            # Remove stopwords and punctuation
            words.extend([
                word for word in sentence_words 
                if word not in self.stopwords.get(language, set()) 
                and word not in punctuation
            ])
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Get top words
        return [word for word, _ in word_freq.most_common(10)]

    def calculate_keyword_score(self, keyword, text, language):
        """Calculate a score for each keyword based on multiple factors"""
        score = 0
        
        # Frequency score (0-5)
        frequency = text.lower().count(keyword.lower())
        score += min(frequency / 2, 5)
        
        # Position score (0-3)
        first_occurrence = text.lower().find(keyword.lower())
        if first_occurrence < len(text) / 3:
            score += 3
        elif first_occurrence < len(text) / 2:
            score += 2
        else:
            score += 1
        
        # Length score (0-2)
        if language == 'en':
            word_count = len(keyword.split())
            if 2 <= word_count <= 3:
                score += 2
            elif word_count == 1:
                score += 1
        else:
            # For Indian languages, use character length
            char_length = len(keyword)
            if 4 <= char_length <= 15:
                score += 2
            elif char_length > 15:
                score += 1
        
        return score

    def get_multilingual_keywords(self, text):
        """Extract keywords with language detection and ranking"""
        # Detect language
        language = self.detect_language(text)
        
        # Get keywords based on language
        if language == 'en':
            keywords = self.extract_keywords_keybert(text)
            keywords.extend(self.extract_keywords_rake(text))
            keywords.extend(self.extract_keywords_spacy(text))
        else:
            keywords = self.extract_keywords_indic(text, language)
        
        # Remove duplicates while preserving order
        unique_keywords = list(dict.fromkeys(keywords))
        
        # Score and rank keywords
        scored_keywords = [
            (keyword, self.calculate_keyword_score(keyword, text, language))
            for keyword in unique_keywords
        ]
        
        # Sort by score in descending order
        ranked_keywords = sorted(scored_keywords, key=lambda x: x[1], reverse=True)
        
        return {
            'language': language,
            'keywords': [kw[0] for kw in ranked_keywords[:10]],
            'scored_keywords': dict(ranked_keywords[:10])
        }


generator = MultilingualKeywordGenerator()

# Get keywords for any text
text = "Your text here..."
results = generator.get_keywords(text)

# Access results
print(f"Language: {results['language']}")
print("Keywords:", results['keywords'])
print("Scores:", results['scores'])


# Example usage
def main():
    # English example
    english_text = """Owning these bottles can be a great experience. But what is a thermosteel water bottle? 
    It is an insulated container that is designed to maintain the temperature of the liquid inside for 
    a prolonged period."""
    
    # Hindi example
    hindi_text = """जल संरक्षण का अर्थ है पानी की बचत करना और इसे प्रदूषण से बचाना। 
    पानी हमारे जीवन का एक महत्वपूर्ण हिस्सा है और इसके बिना जीवन संभव नहीं है।"""
    
    # Gujarati example
    gujarati_text = """પાણી બચાવો એટલે પાણીનો કરકસરયुક્ત ઉપયોગ કરવો અને તેને પ્રદૂषણથી 
    બચાવવું. પાણી આપણા જીવનનો એક મહત્વપૂર્ણ ભાગ છે અને તેના વગર જીવન શક્ય નથી."""
    
    # Initialize the multilingual keyword generator
    keyword_gen = MultilingualKeywordGenerator()
    
    # Process each text
    for text, language in [
        (english_text, "English"), 
        (hindi_text, "Hindi"), 
        (gujarati_text, "Gujarati")
    ]:
        print(f"\n{language} Text Analysis:")
        print("-" * 40)
        
        results = keyword_gen.get_multilingual_keywords(text)
        
        print(f"Detected Language: {results['language']}")
        print("\nTop Keywords with Scores:")
        for keyword, score in results['scored_keywords'].items():
            print(f"- {keyword}: {score:.2f}")

if __name__ == "__main__":
    main()