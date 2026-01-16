# ==============================
# Import required libraries
# ==============================
import pandas as pd                  # Data handling and analysis
import matplotlib.pyplot as plt      # Graphs and plots
import numpy as np
import geopandas as gpd
import seaborn as sns                # Advanced visualizations
import warnings

warnings.filterwarnings('ignore')


# ==============================
# Load & Inspect Refugee Dataset
# ==============================
df = pd.read_csv("Refugee_Dataset.csv", skiprows=4)

df.info()
df.shape
df.columns


# ==============================
# Rename Columns
# ==============================
df.rename(
    columns={
        df.columns[0]: "Country",
        df.columns[1]: "Code",
        df.columns[2]: "Indicator_name",
        df.columns[3]: "Indicator_code"
    },
    inplace=True
)

# Drop unnecessary column
df = df.drop(columns=["Unnamed: 68"])


# ==============================
# Clean Country Names
# ==============================
df = df[df["Country"] != "World"]

df.drop(
    df[df["Country"].str.contains("demographic dividend")].index,
    inplace=True
)

remove_words = ["&", "Africa", "Asia", "America", "Europe"]
df = df[~df["Country"].str.contains("|".join(remove_words))]

df = df[~df["Country"].str.contains("income", na=False)]

remove_words = ["OECD", "small states", "developed countries"]
df = df[~df["Country"].str.contains("|".join(remove_words), na=False)]

df = df[~df["Country"].str.contains("IDA|IBRD|Fragile")]


# ==============================
# Handle Missing Values
# ==============================
df.isnull().sum()
df.info()

# Drop years 1960–1989
df = df.drop(columns=[str(year) for year in range(1960, 1990)])

# Fill missing values using column means (from year columns onward)
df.iloc[:, 4:] = df.iloc[:, 4:].T.fillna(df.iloc[:, 4:].T.mean()).T

df.fillna(0, inplace=True)

# Keep rows with valid data in 1999 and 2019
df = df[df["1999"].notna() & df["2019"].notna()]

x = df.copy()


# ==============================
# Load World Map & Merge
# ==============================
world = gpd.read_file("ne_110m_admin_0_countries.shp")
world = world[["ADMIN", "ADM0_A3", "geometry"]]
world.columns = ["name", "code", "geometry"]

merged_df = pd.merge(
    df,
    world,
    left_on="Code",
    right_on="code",
    how="left"
)

merged_df = merged_df.dropna(subset=["name", "code"])
merged_df = gpd.GeoDataFrame(merged_df)


# ==============================
# Choropleth Map – 1999
# ==============================
fig, ax = plt.subplots(figsize=(12, 8))

merged_df.plot(
    column="1999",
    ax=ax,
    legend=True,
    cmap="viridis",
    missing_kwds={"color": "lightgrey"},
    edgecolor="white"
)

ax.set_title("Choropleth Map 1999", fontsize=16)
ax.set_axis_off()
plt.show()


# ==============================
# Choropleth Map – 2019
# ==============================
fig, ax = plt.subplots(figsize=(12, 8))

merged_df.plot(
    column="2019",
    ax=ax,
    legend=True,
    cmap="plasma",
    missing_kwds={"color": "lightgrey"},
    edgecolor="white"
)

ax.set_title("Choropleth Map 2019", fontsize=16)
ax.set_axis_off()
plt.show()


# ==============================
# Load Conflict Dataset
# ==============================
df = pd.read_csv("GEDEvent_v24_1.csv")

df["year"] = df["year"].astype(int)
df["country"] = df["country"].str.strip()

df = df.dropna(subset=["country", "year", "latitude", "longitude"])

df[["latitude", "longitude"]] = df[["latitude", "longitude"]].apply(
    pd.to_numeric, errors="coerce"
)

death_columns = [
    "deaths_a",
    "deaths_b",
    "deaths_civilians",
    "deaths_unknown",
    "best",
    "high",
    "low"
]

for col in death_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["total_deaths"] = df["best"]

df = df[(df["year"] >= 1990) & (df["year"] <= 2023)]


# ==============================
# Conflict Aggregation
# ==============================
annual = (
    df.groupby(["country", "year"])
      .agg(
          conflict_count=("id", "count"),
          total_deaths=("total_deaths", "sum")
      )
      .reset_index()
)


# ==============================
# Country Name Standardization
# ==============================
standardization = {
    "United States of America": "United States",
    "Russia (Soviet Union)": "Russia",
    "Yemen (North Yemen)": "Yemen",
    "Myanmar (Burma)": "Myanmar",
    "Zimbabwe (Rhodesia)": "Zimbabwe",
    "Iran, Islamic Rep.": "Iran",
    "Korea, Rep.": "South Korea",
    "Venezuela, RB": "Venezuela",
    "Egypt, Arab Rep.": "Egypt",
    "Syrian Arab Republic": "Syria"
}

df["clean"] = df["country"].replace(standardization)


# ==============================
# High Conflict Countries (Since 1999)
# ==============================
conflict_since_1999 = (
    df[df["year"] >= 1999]
    .groupby("country")
    .size()
)

conflict_countries = conflict_since_1999[
    conflict_since_1999 > 5000
].index.tolist()


# ==============================
# Refugee Data for 2019
# ==============================
ref2019 = x[x["Country"].isin(conflict_countries)][
    ["Country", "Code", "2019"]
]

ref2019 = ref2019.rename(
    columns={"2019": "refugee_population"}
)

ref2019["refugee_population"] = pd.to_numeric(
    ref2019["refugee_population"],
    errors="coerce"
)

ref2019 = ref2019.dropna(subset=["refugee_population"])
ref2019

# ==============================
# Import required libraries
# ==============================
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==============================
# Inspect country codes
# ==============================
print(ref2019["Code"].unique())


# ==============================
# Prepare data
# ==============================
# Reset index for a clean DataFrame
ref2019_clean = ref2019.reset_index(drop=True)


# ==============================
# Interactive Choropleth Map (2019)
# ==============================
fig = px.choropleth(
    ref2019_clean,
    locations="Code",                          # Country codes for mapping
    color="refugee_population",                # Color intensity
    hover_name="Country",                      # Hover label
    color_continuous_scale="turbo",             # Vibrant color scale
    title=(
        "Refugee Population (2019) – "
        "Countries with >5000 Conflicts Since 1999"
    )
)

# Display the interactive map
fig.show()

#Selects all unique countries in the Middle East from the DataFrame df.
# Define Middle East countries
middle_east = df[df['region'] == 'Middle East']['country'].unique()
middle_east

#Filters ref2019_clean to keep Middle East countries with more than 50,000 refugees in 2019.

# Filter refugee data for Middle East countries with >50,000 refugees in 2019
me_ref2019 = ref2019_clean[
    (ref2019_clean['Country'].isin(middle_east)) & 
    (ref2019_clean['refugee_population'] > 50000)
]

me_ref2019

import plotly.express as px

# Create an interactive choropleth map for Middle East countries with >50,000 refugees
fig = px.choropleth(
    me_ref2019,
    locations='Code',                     # Country codes for mapping
    color='refugee_population',           # Column to determine color intensity
    hover_name='Country',                 # Country name shown on hover
    color_continuous_scale='plasma',      # Attractive and vibrant color scale
    title='Middle East Countries with >50,000 Refugees in 2019'
)

# Display the interactive map
fig.show()

# Calculate total refugees 1990-2020
year1990_2020 = [str(year) for year in range(1990, 2021)]
year = [year for year in year1990_2020 if year in x.columns]
year

x   

x['refugees_1990_2020'] = x[year].sum(axis=1, skipna=True)
x['refugees_1990_2020']

# Top 10 countries
top_10 = x.nlargest(10, 'refugees_1990_2020')[['Country', 'refugees_1990_2020']]
print("Top 10 countries by total refugees (1990-2020):")
print(top_10)

top_10 = x[~x['Country'].isin(['Heavily indebted poor countries (HIPC)', 'Arab World'])].nlargest(10, 'refugees_1990_2020')[['Country', 'refugees_1990_2020']]
print("Top 10 countries by total refugees (1990-2020):")
print(top_10)

conflict = df.groupby(['country', 'year']).agg({
    'id': 'count',
    'best': 'sum'
}).reset_index()
conflict.columns = ['country', 'year', 'conflict_count', 'total_deaths']
