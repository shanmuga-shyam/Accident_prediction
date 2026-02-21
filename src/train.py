import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from preprocessor import preprocess_data

# Resolve project root (parent of this `src` directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = str(PROJECT_ROOT / "dataset" / "accident.xlsx")

def train_models():

    # Load data
    df, encoders = preprocess_data(DATASET)

    target = "Accident Type"

    X = df.drop(target, axis=1)
    y = df[target]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SCALE DATA (Important for SVM & Logistic)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "SVM": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }

    results = {}

    # Train and evaluate
    for name, model in models.items():

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")

    # Best model
    best_name = max(results, key=results.get)
    best_model = models[best_name]

    print("\nBest Model:", best_name)

    # Save everything to project-level `models/` directory
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, str(models_dir / "best_model.pkl"))
    joblib.dump(encoders, str(models_dir / "encoders.pkl"))
    joblib.dump(scaler, str(models_dir / "scaler.pkl"))

    return results

if __name__ == "__main__":
    train_models()