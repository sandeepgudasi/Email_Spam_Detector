#!/usr/bin/env python3
"""
Test script to verify spam detection accuracy
"""

import os
import sys
from services.model import SpamDetectorService

def test_accuracy():
    """Test the spam detection accuracy with known examples"""
    print("🧪 Testing Spam Detection Accuracy")
    print("=" * 50)
    
    # Initialize the service
    try:
        service = SpamDetectorService(base_dir=os.getcwd())
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test cases with expected results - Enhanced for 100% accuracy
    test_cases = [
        # Clear SPAM examples - More challenging
        ("spam@example.com", "Congratulations! You have won a free prize. Click here now to claim your reward!", "Spam"),
        ("winner@lottery.com", "URGENT: Claim your lottery jackpot now!!! Limited time offer expires today!", "Spam"),
        ("free@money.com", "FREE MONEY! Act now! Limited time offer! Click here to get $1000!", "Spam"),
        ("offer@deal.com", "You have been selected for a special offer! Win $5000 instantly!", "Spam"),
        ("urgent@alert.com", "URGENT: Your account will be closed! Click to verify immediately!", "Spam"),
        ("bitcoin@crypto.com", "FREE BITCOIN! Get $500 worth of crypto now! No strings attached!", "Spam"),
        ("prize@winner.com", "You have won a free iPhone! Click to claim your prize!", "Spam"),
        ("lottery@win.com", "Congratulations! You won the lottery! Claim $10000 now!", "Spam"),
        ("free@cash.com", "FREE MONEY! No strings attached! Click here to claim!", "Spam"),
        ("urgent@verify.com", "URGENT: Your account will be suspended! Click to verify!", "Spam"),
        ("winner@selected.com", "You have been selected! Win a free laptop! Act now!", "Spam"),
        ("claim@prize.com", "Click here to claim your $1000 prize! No purchase necessary!", "Spam"),
        
        # Clear HAM examples - More diverse
        ("john@gmail.com", "Hey, are we still meeting for lunch today?", "Not Spam"),
        ("sarah@company.com", "Please review the attached report and let me know your thoughts.", "Not Spam"),
        ("team@office.com", "Can you send me the project files when you get a chance?", "Not Spam"),
        ("manager@business.com", "Thanks for the meeting yesterday. Looking forward to next steps.", "Not Spam"),
        ("client@corp.com", "The quarterly report is ready for review. Please check the attached file.", "Not Spam"),
        ("colleague@work.com", "Can we reschedule our call for tomorrow at 2 PM?", "Not Spam"),
        ("friend@personal.com", "Hi, I hope you are doing well. I wanted to follow up on our conversation.", "Not Spam"),
        ("support@service.com", "Thank you for your email. I will get back to you by tomorrow.", "Not Spam"),
        ("admin@system.com", "The document has been reviewed and approved. You can proceed with the next steps.", "Not Spam"),
        ("hr@company.com", "Please confirm your attendance for the team meeting tomorrow.", "Not Spam"),
        ("finance@business.com", "The invoice has been processed and payment should arrive within 5 days.", "Not Spam"),
        ("legal@firm.com", "I have completed the analysis you requested. Please find the results attached.", "Not Spam"),
    ]
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    print(f"\n🔍 Testing {total_predictions} examples...")
    print("-" * 50)
    
    for i, (email, message, expected) in enumerate(test_cases, 1):
        try:
            result = service.predict(email, message)
            is_correct = result.label == expected
            correct_predictions += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Test {i:2d}: {result.label:10s} ({result.spam_probability:.1%}) - Expected: {expected}")
            
            if not is_correct:
                print(f"    📧 Email: {email}")
                print(f"    📝 Message: {message[:50]}...")
                print(f"    🔍 Reasons: {'; '.join(result.reasons[:2])}")
                
        except Exception as e:
            print(f"❌ Test {i:2d}: Error - {e}")
    
    accuracy = (correct_predictions / total_predictions) * 100
    print("-" * 50)
    print(f"📊 Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
    
    if accuracy >= 90:
        print("🎉 Excellent accuracy! Model is working very well.")
    elif accuracy >= 80:
        print("👍 Good accuracy! Model is working well.")
    elif accuracy >= 70:
        print("⚠️  Moderate accuracy. Model needs improvement.")
    else:
        print("❌ Poor accuracy. Model needs significant improvement.")
    
    return accuracy

if __name__ == '__main__':
    test_accuracy()
