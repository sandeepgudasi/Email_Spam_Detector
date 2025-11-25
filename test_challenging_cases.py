#!/usr/bin/env python3
"""
Test script for challenging spam detection cases
This script tests the model with various challenging examples to ensure 100% accuracy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.model import SpamDetectorService

def test_challenging_cases():
    """Test the model with challenging spam detection cases"""
    print("🚀 Testing Challenging Spam Detection Cases")
    print("=" * 50)
    
    try:
        spam_service = SpamDetectorService(base_dir='.')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Challenging test cases that should be detected as SPAM
    challenging_spam_cases = [
        # Subtle spam examples
        ("noreply@bank.com", "Your account needs verification. Please click here to update your information immediately."),
        ("support@paypal.com", "We noticed unusual activity on your account. Click here to secure your account now."),
        ("noreply@amazon.com", "Your order has been delayed. Click here to update your payment information."),
        ("security@apple.com", "Your Apple ID has been locked. Click here to unlock your account."),
        ("billing@netflix.com", "Your subscription will be cancelled. Click here to update your payment method."),
        
        # Prize and lottery spam
        ("winner@lottery.org", "Congratulations! You have been selected as a winner in our annual lottery draw."),
        ("prize@contest.com", "You have won a free vacation! Click here to claim your prize now."),
        ("sweepstakes@win.com", "You are the lucky winner! Claim your $5000 prize immediately."),
        ("jackpot@lottery.net", "You have won the jackpot! Click here to claim your $10000 prize."),
        ("winner@prize.org", "Congratulations! You have been chosen as our grand prize winner."),
        
        # Financial spam
        ("investment@crypto.com", "Get rich quick with our guaranteed investment program. Click here to start earning."),
        ("loan@money.com", "Get approved for a $5000 loan instantly! No credit check required. Click here."),
        ("credit@repair.com", "Fix your credit score in 30 days! Guaranteed results. Click here to start."),
        ("debt@consolidation.com", "Consolidate your debt and save money! Click here for a free quote."),
        ("bitcoin@mining.com", "Earn $1000 per day with our bitcoin mining program. Click here to join."),
        
        # Urgency-based spam
        ("urgent@security.com", "URGENT: Your account will be closed in 24 hours. Click here to prevent this."),
        ("emergency@update.com", "EMERGENCY: Your information needs to be updated immediately. Click here."),
        ("critical@action.com", "CRITICAL: Action required on your account. Click here to resolve."),
        ("immediate@response.com", "IMMEDIATE ACTION REQUIRED: Click here to verify your account."),
        ("expires@soon.com", "This offer expires in 2 hours! Click here to claim your free gift."),
        
        # Free offers and deals
        ("free@offer.com", "Get a free iPhone! No purchase necessary. Click here to claim."),
        ("deal@limited.com", "Limited time offer: 90% off everything! Click here to save now."),
        ("discount@special.com", "Special discount just for you! Click here to get 50% off."),
        ("free@trial.com", "Free trial for 30 days! No obligation. Click here to start."),
        ("gift@free.com", "Free gift waiting for you! Click here to claim your reward."),
    ]
    
    # Legitimate emails that should NOT be detected as spam
    legitimate_cases = [
        # Business emails
        ("john@company.com", "Hi, I wanted to follow up on our meeting yesterday. Can we schedule a call for next week?"),
        ("sarah@business.com", "Please find the quarterly report attached. Let me know if you have any questions."),
        ("team@office.com", "The project deadline has been extended to next Friday. Please update your schedules."),
        ("manager@corp.com", "I need your approval on the budget proposal. Can we discuss this tomorrow?"),
        ("hr@company.com", "Please confirm your attendance for the team meeting on Monday at 2 PM."),
        
        # Personal emails
        ("friend@personal.com", "Hey, are we still on for dinner tonight? Let me know if you need directions."),
        ("family@home.com", "Thanks for the birthday wishes! I had a great time at the party."),
        ("colleague@work.com", "I have completed the analysis you requested. The results look promising."),
        ("client@business.com", "Thank you for your patience. The issue has been resolved and everything is working."),
        ("support@service.com", "I have forwarded your request to the appropriate department. You should hear back soon."),
        
        # Professional communications
        ("legal@firm.com", "The contract has been reviewed and approved. You can proceed with the next steps."),
        ("finance@company.com", "The invoice has been processed and payment should arrive within 5 business days."),
        ("admin@system.com", "The system maintenance has been completed. All services are now available."),
        ("it@support.com", "Your password has been reset successfully. Please check your email for the new password."),
        ("marketing@team.com", "The campaign launch has been scheduled for next Monday. Please prepare your materials."),
    ]
    
    print(f"\n🔍 Testing {len(challenging_spam_cases)} challenging SPAM cases...")
    spam_correct = 0
    spam_total = len(challenging_spam_cases)
    
    for i, (sender, message) in enumerate(challenging_spam_cases, 1):
        try:
            result = spam_service.predict(sender, message)
            is_spam = result.label == "Spam"
            spam_correct += is_spam
            
            status = "✅" if is_spam else "❌"
            print(f"{status} Test {i:2d}: {result.label} ({result.spam_probability:.1%}) - {message[:50]}...")
            
            if not is_spam:
                print(f"   ⚠️  Expected SPAM but got {result.label}")
                print(f"   📊 Confidence: {result.spam_probability:.1%}")
                print(f"   🔍 Reasons: {', '.join(result.reasons[:3])}")
                
        except Exception as e:
            print(f"❌ Test {i:2d}: Error - {e}")
    
    print(f"\n🔍 Testing {len(legitimate_cases)} legitimate email cases...")
    ham_correct = 0
    ham_total = len(legitimate_cases)
    
    for i, (sender, message) in enumerate(legitimate_cases, 1):
        try:
            result = spam_service.predict(sender, message)
            is_ham = result.label == "Not Spam"
            ham_correct += is_ham
            
            status = "✅" if is_ham else "❌"
            print(f"{status} Test {i:2d}: {result.label} ({result.spam_probability:.1%}) - {message[:50]}...")
            
            if not is_ham:
                print(f"   ⚠️  Expected NOT SPAM but got {result.label}")
                print(f"   📊 Confidence: {result.spam_probability:.1%}")
                print(f"   🔍 Reasons: {', '.join(result.reasons[:3])}")
                
        except Exception as e:
            print(f"❌ Test {i:2d}: Error - {e}")
    
    # Calculate overall accuracy
    total_correct = spam_correct + ham_correct
    total_tests = spam_total + ham_total
    overall_accuracy = (total_correct / total_tests) * 100
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"=" * 30)
    print(f"🎯 SPAM Detection: {spam_correct}/{spam_total} ({spam_correct/spam_total*100:.1f}%)")
    print(f"✅ HAM Detection:  {ham_correct}/{ham_total} ({ham_correct/ham_total*100:.1f}%)")
    print(f"🏆 Overall Accuracy: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
    
    if overall_accuracy >= 95:
        print(f"🎉 EXCELLENT! Model achieves {overall_accuracy:.1f}% accuracy!")
    elif overall_accuracy >= 90:
        print(f"👍 GOOD! Model achieves {overall_accuracy:.1f}% accuracy!")
    else:
        print(f"⚠️  Model needs improvement. Current accuracy: {overall_accuracy:.1f}%")
    
    return overall_accuracy

if __name__ == "__main__":
    test_challenging_cases()
