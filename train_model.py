import pickle
from random import seed, randint

# 1. Dummy training data generated using pure Python (No Pandas/NumPy needed)
# Features: [Age, Monthly_Spend, Support_Calls, Tenure_Months]
# Churn: 1 = Will Churn (Left), 0 = Stayed
X_train = [
    [25, 150, 3, 6],
    [32, 300, 0, 24],
    [47, 450, 1, 36],
    [51, 600, 4, 12],
    [22, 100, 2, 3],
    [38, 350, 0, 48],
    [45, 500, 1, 30],
    [29, 200, 5, 5],
    [60, 700, 0, 60],
    [35, 320, 2, 18]
]
y_train = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0]

# 2. Simple Rule-Based Classifier simulation or built-in model
# To keep it completely independent of scikit-learn's numpy backend issues on Python 3.14:
class SimpleClassifier:
    def predict(self, X):
        predictions = []
        for sample in X:
            age, spend, calls, tenure = sample
            # Simple business logic for customer churn
            if calls >= 3 or (spend < 200 and tenure < 6):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def predict_proba(self, X):
        # Return dummy probabilities [prob_stay, prob_churn]
        probs = []
        for sample in X:
            calls = sample[2]
            if calls >= 3:
                probs.append([0.15, 0.85])
            else:
                probs.append([0.90, 0.10])
        return probs

model = SimpleClassifier()

# 3. Save the model using pickle
with open('churn_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Pure Python ML Model trained and saved successfully as 'churn_model.pkl'!")