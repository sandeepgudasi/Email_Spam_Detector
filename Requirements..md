Name: Email fraud detection.
Work: Predicting the Normal safe email and fraud email by senders mail and message content.
    -create a mail fraud detecting site with related and advanced theme, features and animations.
    -here are few suggestions for building the site.

#Working:
    Input:
        -it should take the input from the user i.e, Senders mail id and mail message.
    Output:
        -By reviewing thoroughly the senders mail and message it should give is it spam mail or normal safe mail.

##Site:
    -this site should have two pages 
        -1.Home Page
                -in the header at the right side i need home button and support button. which are clickable and when i click on home it should redirect to the home page and when i click on the support button it should redirect to the support page and at the left side logo as Mail Scanner.
                -at the beginning add a glowing and attractive button named click here to begin.
                -in the home page i need some detaild about the page like why its needed and how it will help to users and all and also add something about recent mail scams i need this in full page of 10 cards means 2-3 pages of details about the recent digital scams and mail scams
                -in the footer add copyright as Mail Scanner and add all rights are reserved.
                -this entire page should be very attractive and glowing neon effects for this use the theme of virus scanning pages like color and theme 
        -2.Scanning Page.
                -use the online datasets and based on that give the result 
        -3.result page
        -4.support page
        -5.
        -6.



    
1. Functional Requirements

        Spam/Ham Detection

        The system must classify emails/messages as Spam or Ham (legitimate).

        Input: sender email + message text.

        Output: prediction with label.

        Spam Probability (Confidence Score)

        The system must provide a probability score showing how confident the model is about its prediction.

        Suspicious Word Detection

        The system should highlight words that commonly indicate spam (e.g., win, free, money, prize).

        Sender Reputation Analysis

        Extract sender domain (e.g., gmail.com, lottery.com).

        Mark common/trusted domains as “safe” and unusual domains as “suspicious.”

        Message Statistics

        Extract basic features such as:

        Word count

        Number of capitalized words

        Number of links (http://, https://)

        Result Logging (Optional)

        Save predictions to a file (CSV) for future reference.

2. Non-Functional Requirements

        The system should be lightweight and run on a personal computer.

        The model should achieve at least 85–90% accuracy on test data.

        The interface should allow easy testing (command line or simple UI).

3. Data Requirements

        Dataset Source:

        SMS Spam Collection Dataset (Kaggle: link)

        Additional sample email datasets: SpamAssassin Public Corpus

        Extracted Features from Data:

        Sender Features:

        Email domain (gmail.com, yahoo.com, suspicious domains)

        Text Features:

        Spam-related keywords (e.g., win, money, free)

        Word frequency (using TF-IDF vectorizer)

        Count of links, numbers, capitalized words Label

        Spam or Ham (from dataset ground truth)

4. Software & Tools Required

        Libraries:

        pandas → data handling

        numpy → numerical operations

        scikit-learn → machine learning models (Naive Bayes, Logistic Regression, etc.)

        matplotlib / seaborn → visualization (confusion matrix)

        nltk (optional) → text preprocessing (stopwords, stemming)
5. Outputs

        Prediction: Spam / Ham

        Confidence score (e.g., Spam 92%)

        Highlighted suspicious words in the message

        Sender trust rating

        Message statistics (words, caps, links)

        Result history log

        give the reason why its spam and whats the percentage of being spam

        if its not spam then give reason why its not spam 

    


