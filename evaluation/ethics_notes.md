1. Dataset Bias

The Amazon review dataset may contain biased language, uneven representation of product categories, and subjective labelling. Positive and negative sentiment is highly context-dependent and reviewers may use sarcasm or culturally specific expressions that are misinterpreted by the model. This introduces potential bias into predictions.

2. Label Accuracy

The dataset uses binary labels (“positive” or “negative”), which oversimplify human sentiment. Many reviews contain neutral or mixed opinions. Training the model on imperfect labels can lead to unfair or inaccurate classifications.

3. Fairness and Representation

Certain writing styles or cultural expressions may be incorrectly classified. For example, users who write short sentences or non-native English writers may be judged more negatively due to limited context or grammar differences.

4. Privacy Considerations

The dataset contains user-generated reviews. Although the dataset is publicly available and anonymised, it is important that the system does not store, track, or log any new user inputs during testing. Our Flask app does not retain or transmit user text, ensuring privacy protection.

5. Transparency & Explainability

Sentiment models like LSTMs behave as “black-box” predictors. Users should be informed that predictions are automated and based on patterns learned from historical data. The displayed confidence score helps increase transparency but does not fully explain how predictions were made.

6. Responsible Use

The model should not be used to make high-impact decisions such as hiring, lending, or personal assessment. It is only suitable for simple product review sentiment tasks. Misuse of the model outside this context may cause harm or unfair outcomes.

7. Model Bias due to Lexical Over-Reliance 

Our model struggles with short negated expressions such as ‘not bad’. Although humans interpret this as mildly positive, the model heavily relies on the presence of the word ‘bad’, which is strongly associated with negative sentiment in the training data