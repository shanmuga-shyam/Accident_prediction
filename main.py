from src.train import train_models

print("Training Multiple Machine Learning Models...\n")

results = train_models()

print("\nFinal Accuracy Comparison")
for k,v in results.items():
    print(f"{k} : {v:.4f}")

print("\nProject Completed Successfully!")