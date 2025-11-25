# Mail Spam Detector

Advanced email/SMS spam detection web app with ensemble ML algorithms and neon UI.

## 🚀 Quick Start

### Option 1: Using the startup script (Recommended)
```powershell
python run.py
```

### Option 2: Manual setup
1. Create virtual environment (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies
```powershell
pip install -r requirements.txt
```

3. Run the app
```powershell
python app.py
```

## 🌐 Access Points
- **Main App:** http://localhost:5000
- **Scan Page:** http://localhost:5000/scan
- **Test Endpoint:** http://localhost:5000/test
- **How to Use:** http://localhost:5000/how-to-use
- **Support:** http://localhost:5000/support

## 🧠 ML Features
- **Ensemble Learning:** Naive Bayes + SVM + Logistic Regression
- **Advanced NLP:** NLTK preprocessing with stemming and stopword removal
- **Multiple Datasets:** SMS Spam + Enron Email datasets
- **Enhanced Analysis:** Detailed reasoning and confidence scores
- **Cross-validation:** 5-fold validation for robust performance

## 📱 Responsive Design
- **Mobile-First:** Optimized for phones, tablets, and desktops
- **Touch-Friendly:** Large tap targets and touch optimizations
- **Cross-Platform:** Works on iOS, Android, Windows, Mac, Linux
- **Adaptive Layout:** Automatically adjusts to any screen size
- **High DPI Support:** Crisp graphics on retina displays

## 📊 Expected Performance
- **Accuracy:** 92-96% with ensemble learning
- **Features:** 10,000+ TF-IDF features with n-grams
- **Processing:** Real-time analysis with detailed explanations

## 🔧 Troubleshooting

### NLTK Issues
If you see NLTK errors, run the fix script:
```powershell
python fix_nltk.py
```

### General Issues
- **First run:** Downloads datasets and trains model automatically
- **Test endpoint:** Visit `/test` to verify model is working
- **Logs:** Results saved to `logs/results.csv`
- **Debug mode:** Enabled for detailed error messages
- **Fallback processing:** Works even without NLTK

## 📝 Notes
- Model trains automatically on first run
- Results are logged with timestamps and detailed analysis
- Professional-grade spam detection with ensemble learning
