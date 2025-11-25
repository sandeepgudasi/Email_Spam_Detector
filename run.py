#!/usr/bin/env python3
"""
Mail Spam Detector - Startup Script
This script helps ensure the application starts correctly with proper error handling.
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'flask', 'scikit-learn', 'pandas', 'numpy', 'nltk', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def fix_nltk_if_needed():
    """Fix NLTK data issues if needed"""
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        # Test if NLTK works
        word_tokenize("test")
        return True
    except Exception as e:
        print(f"⚠️ NLTK issue detected: {e}")
        print("🔧 Attempting to fix NLTK...")
        try:
            from fix_nltk import fix_nltk
            return fix_nltk()
        except Exception as fix_error:
            print(f"⚠️ Could not fix NLTK: {fix_error}")
            print("📝 System will use fallback text processing")
            return True  # Continue anyway

def test_model_accuracy():
    """Test model accuracy before starting the server"""
    print("🧪 Testing model accuracy...")
    try:
        from test_accuracy import test_accuracy
        accuracy = test_accuracy()
        if accuracy < 70:
            print("⚠️  Warning: Model accuracy is below 70%. Consider retraining.")
        
        print("\n🧪 Testing challenging cases...")
        try:
            from test_challenging_cases import test_challenging_cases
            challenging_accuracy = test_challenging_cases()
            if challenging_accuracy >= 95:
                print("🎉 Model ready for production with excellent accuracy!")
            elif challenging_accuracy >= 90:
                print("👍 Model performs well on challenging cases!")
            else:
                print("⚠️ Model may need further tuning for optimal performance")
        except Exception as e:
            print(f"⚠️ Challenging cases test failed: {e}")
        
        return True
    except Exception as e:
        print(f"⚠️  Could not test accuracy: {e}")
        return True  # Continue anyway

def main():
    """Main startup function"""
    print("🚀 Mail Spam Detector - Starting up...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    print("✅ Project structure looks good")
    
    # Fix NLTK issues if needed
    if not fix_nltk_if_needed():
        print("⚠️ NLTK fix failed, but continuing with fallback processing")
    
    # Test model accuracy
    if not test_model_accuracy():
        print("❌ Model accuracy test failed")
        sys.exit(1)
    
    print("🔄 Starting Flask application...")
    print("=" * 50)
    
    try:
        # Import and run the app
        from app import app
        print("🌐 Server starting at: http://localhost:5000")
        print("🧪 Test endpoint: http://localhost:5000/test")
        print("📱 Scan page: http://localhost:5000/scan")
        print("📊 Accuracy test: python test_accuracy.py")
        print("=" * 50)
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
