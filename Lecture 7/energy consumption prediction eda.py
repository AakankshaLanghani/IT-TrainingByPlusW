import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "yearly_full_release_long_format.csv"
df = pd.read_csv(file_path)

# Display basic dataset info
df.info()
print(df.head())

# Checking for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values[missing_values > 0])

# Filter dataset for total energy consumption (Demand)
df_energy = df[df["Variable"] == "Demand"].copy()
df_energy = df_energy[['Area', 'Year', 'Value']].dropna()
df_energy.rename(columns={'Area': 'Country', 'Value': 'Energy Consumption (TWh)'}, inplace=True)

# Global energy consumption trend over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_energy.groupby("Year")["Energy Consumption (TWh)"].sum().reset_index(),
             x="Year", y="Energy Consumption (TWh)", marker="o", color="b")
plt.title("Global Energy Consumption Trend Over Time")
plt.xlabel("Year")
plt.ylabel("Total Energy Consumption (TWh)")
plt.grid(True)
plt.show()

# Top 10 energy-consuming countries (latest year available)
latest_year = df_energy["Year"].max()
top_countries = df_energy[df_energy["Year"] == latest_year].nlargest(10, "Energy Consumption (TWh)")

plt.figure(figsize=(12, 6))
sns.barplot(data=top_countries, x="Energy Consumption (TWh)", y="Country", palette="viridis")
plt.title(f"Top 10 Energy-Consuming Countries in {latest_year}")
plt.xlabel("Energy Consumption (TWh)")
plt.ylabel("Country")
plt.grid(axis="x")
plt.show()

# Energy consumption trends for top 5 countries
top_5_countries = top_countries["Country"].head(5).tolist()
df_top_countries = df_energy[df_energy["Country"].isin(top_5_countries)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_top_countries, x="Year", y="Energy Consumption (TWh)", hue="Country", marker="o")
plt.title("Energy Consumption Trends for Top 5 Countries")
plt.xlabel("Year")
plt.ylabel("Energy Consumption (TWh)")
plt.legend(title="Country")
plt.grid(True)
plt.show()

# Correlation analysis
correlation = df_energy[["Year", "Energy Consumption (TWh)"]].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Year and Energy Consumption")
plt.show()
