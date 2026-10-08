import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# RANDOM FOREST VOTING
# ==============================

# Predictions made by three decision trees
predictions = ["Pass", "Pass", "Fail"]

# Display predictions
print("===== DECISION TREE PREDICTIONS =====")

for i, prediction in enumerate(predictions, start=1):
    print(f"Tree {i}: {prediction}")


# ==============================
# COUNT VOTES
# ==============================

votes = pd.Series(predictions).value_counts()

pass_votes = votes.get("Pass", 0)
fail_votes = votes.get("Fail", 0)

print("\n===== VOTING RESULTS =====")
print("Pass Votes:", pass_votes)
print("Fail Votes:", fail_votes)


# ==============================
# MAJORITY VOTING
# ==============================

final_prediction = votes.idxmax()

print("\nFinal Random Forest Prediction:", final_prediction)


# ==============================
# VOTING PERCENTAGE
# ==============================

total_votes = len(predictions)

final_vote_percentage = (
    votes[final_prediction] / total_votes
) * 100

print(
    "Voting Percentage:",
    round(final_vote_percentage, 2),
    "%"
)


# ==============================
# GRAPH
# ==============================

plt.figure(figsize=(7, 5))

plt.bar(
    ["Pass", "Fail"],
    [pass_votes, fail_votes]
)

plt.title("Random Forest Majority Voting")
plt.xlabel("Prediction Class")
plt.ylabel("Number of Votes")

plt.tight_layout()
plt.show()