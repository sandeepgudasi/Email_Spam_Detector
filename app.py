import os
import traceback
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash

try:
    from services.model import SpamDetectorService, PredictionResult
except ImportError as e:
    print(f"Error importing model service: {e}")
    traceback.print_exc()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Ensure required folders exist
os.makedirs(os.path.join(app.root_path, 'static', 'css'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'js'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'data'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'logs'), exist_ok=True)

# Initialize (and train if needed) the model service
try:
    print("🔄 Initializing spam detection service...")
    spam_service = SpamDetectorService(base_dir=app.root_path)
    print("✅ Spam detection service initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing spam service: {e}")
    traceback.print_exc()
    spam_service = None


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if request.method == 'POST':
        try:
            # Check if service is available
            if spam_service is None:
                return render_template('scan.html', error_message='Spam detection service is not available. Please try again later.')
            
            # Get form data
            sender_email = request.form.get('sender_email', '').strip()
            message_text = request.form.get('message_text', '').strip()
            
            # Validate input
            if not sender_email:
                return render_template('scan.html', error_message='Please enter a sender email address.')
            
            if not message_text:
                return render_template('scan.html', error_message='Please enter the message text to analyze.')
            
            if len(message_text) < 5:
                return render_template('scan.html', error_message='Message text is too short. Please enter at least 5 characters.')
            
            print(f"🔍 Processing scan request for: {sender_email[:20]}...")
            
            # Make prediction
            result: PredictionResult = spam_service.predict(sender_email, message_text)
            
            print(f"✅ Prediction completed: {result.label} ({result.spam_probability:.2%})")

            # Log result
            try:
                spam_service.log_result(
                    timestamp=datetime.utcnow().isoformat(),
                    sender_email=sender_email,
                    message_text=message_text,
                    prediction_label=result.label,
                    spam_probability=result.spam_probability,
                    reasons='; '.join(result.reasons),
                    suspicious_words=','.join(result.suspicious_words),
                    domain_rating=result.domain_rating,
                    stats=result.stats,
                )
                print("📝 Result logged successfully")
            except Exception as log_error:
                print(f"⚠️ Warning: Could not log result: {log_error}")

            return render_template('result.html', result=result)
            
        except Exception as e:
            print(f"❌ Error in scan route: {e}")
            traceback.print_exc()
            return render_template('scan.html', error_message=f'An error occurred while processing your request: {str(e)}')

    return render_template('scan.html')


@app.route('/support')
def support():
    return render_template('support.html')


@app.route('/how-to-use')
def how_to_use():
    return render_template('how_to_use.html')


@app.route('/test')
def test():
    """Test route to verify the model is working"""
    if spam_service is None:
        return "❌ Spam service not available"
    
    try:
        # Test with a simple example
        test_result = spam_service.predict("test@example.com", "This is a test message")
        return f"✅ Model working! Test result: {test_result.label} ({test_result.spam_probability:.2%})"
    except Exception as e:
        return f"❌ Model test failed: {str(e)}"


@app.errorhandler(404)
def not_found(_):
    return redirect(url_for('home'))


@app.errorhandler(500)
def internal_error(error):
    print(f"❌ Internal server error: {error}")
    traceback.print_exc()
    return render_template('scan.html', error_message='An internal error occurred. Please try again.')


if __name__ == '__main__':
    try:
        print("🚀 Starting Mail Spam Detector...")
        print("📁 Project directory:", app.root_path)
        print("🌐 Server will be available at: http://localhost:5000")
        print("🧪 Test endpoint: http://localhost:5000/test")
        
        port = int(os.environ.get('PORT', '5000'))
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        traceback.print_exc()
