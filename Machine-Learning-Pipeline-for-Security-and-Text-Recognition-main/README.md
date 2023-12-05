# Machine Learning Pipeline for Security and Text Recognition

+ Predicting XOR-PUF Responses: Trained an SVM, Used Khatri-Rao product to perform feature engineering on the challenge-response pairs.
  + SVM with hinge loss and stochastic gradient descent performed the best on the evaluation dataset.
+ Multiclass Classification to Correct Errors in Code: Trained multiple models to detect the error class associated with a Bag of Words representation of a line of code.
  + Evaluation measures were precision at k and macro-precision at k along with model size and inference speed.
  +  Logistic Regreesion performed best on the evaluation metrics against kNN, SVM, Decision Tree and XGBoost.
+ DeCAPTCHA, Developed an end-to-end pipeline to preprocess CAPTCHA images and perform optical character recognition.
  +  Used image processing techniques namely, erosion, dilation and thresholding to remove color and obfuscation.
  +  Segmented the characters from cleaned image by scanning the pixel values in binary image.
  + The evaluation metric was accuracy and Logistic Regression performed best against kNN, SVC and Random Forest.
