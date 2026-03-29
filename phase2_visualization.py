import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("outputs/phase2_clustered.csv")

sns.set_style("whitegrid")

plt.figure()
sns.countplot(x="Vulnerability_Label", data=df)
plt.title("Distribution of Financial Vulnerability Groups")
plt.savefig("plots/vulnerability_distribution.png")
plt.close()

plt.figure()
sns.scatterplot(
    x=df["Income"],
    y=df["Expense_Ratio"],
    hue=df["Vulnerability_Label"]
)
plt.title("Income vs Expense Ratio by Vulnerability Group")
plt.savefig("plots/income_expense_clusters.png")
plt.close()

plt.figure()
sns.boxplot(
    x="Vulnerability_Label",
    y="Expense_Ratio",
    data=df
)
plt.title("Expense Ratio Distribution by Vulnerability Group")
plt.savefig("plots/expense_ratio_by_group.png")
plt.close()

plt.figure()
sns.boxplot(
    x="Vulnerability_Label",
    y="Savings_Gap",
    data=df
)
plt.title("Savings Gap Distribution by Vulnerability Group")
plt.savefig("plots/savings_gap_by_group.png")
plt.close()

plt.figure()
for label in df["Vulnerability_Label"].unique():
    subset = df[df["Vulnerability_Label"] == label]
    sns.kdeplot(subset["Income"], label=label)

plt.legend()
plt.title("Income Distribution by Vulnerability Group")
plt.savefig("plots/income_distribution_groups.png")
plt.close()

cluster_counts = df["Vulnerability_Label"].value_counts()

plt.figure()
plt.pie(cluster_counts, labels=cluster_counts.index, autopct="%1.1f%%")
plt.title("Financial Vulnerability Population Share")
plt.savefig("plots/vulnerability_pie_chart.png")
plt.close()


print("All Visualization Charts Generated Successfully")
