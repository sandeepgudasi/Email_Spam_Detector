#!/usr/bin/env python3
"""
NLTK Setup Script - Fixes NLTK data issues
"""

import nltk
import os

def fix_nltk():
    """Download all required NLTK data"""
    print("🔧 Fixing NLTK data issues...")
    
    try:
        # Download all required NLTK data
        nltk_data = [
            'punkt',
            'punkt_tab',  # Newer NLTK versions
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words'
        ]
        
        for data in nltk_data:
            try:
                print(f"📦 Downloading {data}...")
                nltk.download(data, quiet=True)
                print(f"✅ {data} downloaded successfully")
            except Exception as e:
                print(f"⚠️ Could not download {data}: {e}")
        
        print("🎉 NLTK setup completed!")
        
        # Test NLTK functionality
        print("\n🧪 Testing NLTK functionality...")
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        
        # Test tokenization
        test_text = "This is a test message for spam detection."
        tokens = word_tokenize(test_text)
        print(f"✅ Tokenization works: {tokens}")
        
        # Test stopwords
        stop_words = set(stopwords.words('english'))
        print(f"✅ Stopwords loaded: {len(stop_words)} words")
        
        # Test stemming
        stemmer = PorterStemmer()
        stemmed = stemmer.stem("running")
        print(f"✅ Stemming works: running -> {stemmed}")
        
        print("\n🎯 NLTK is now fully functional!")
        return True
        
    except Exception as e:
        print(f"❌ NLTK setup failed: {e}")
        print("⚠️ The system will use fallback text processing")
        return False

if __name__ == '__main__':
    fix_nltk()
