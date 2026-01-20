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
# Define a mapping to standardize country names
mapping = {
    'Congo, Dem. Rep.': 'DR Congo (Zaire)',
    'Myanmar': 'Myanmar (Burma)', 
    'Viet Nam': 'Vietnam'
}

# Apply the mapping to the top 10 refugee countries
top_10_mapped = [mapping.get(country, country) for country in top_10['Country'].tolist()]

# Filter conflict data for the top 10 countries between 1990 and 2020
con = conflict[
    (conflict['country'].isin(top_10_mapped)) & 
    (conflict['year'] >= 1990) & 
    (conflict['year'] <= 2020)
]

# Calculate total conflicts per country and sort in descending order
total_conflicts = con.groupby('country')['conflict_count'].sum().sort_values(ascending=False)

# Display the results
print("Total conflicts for top 10 refugee countries (1990-2020):")
print(total_conflicts)
import matplotlib.pyplot as plt

# Align data - only countries that appear in both datasets
common_countries = set(top_10['Country']) & set(total_conflicts.index)
refugee_values = [top_10[top_10['Country'] == country]['refugees_1990_2020'].iloc[0] for country in common_countries]
conflict_values = [total_conflicts[country] for country in common_countries]

plt.figure(figsize=(10, 6))

# Scatter plot with color and alpha
plt.scatter(
    conflict_values, 
    refugee_values, 
    s=150,                 # Bubble size
    c=refugee_values,      # Color based on refugee population
    cmap='viridis',        # Attractive color map
    alpha=0.8,             # Transparency
    edgecolor='black'      # Border color for better contrast
)

# Add country labels
for i, country in enumerate(common_countries):
    plt.annotate(
        country, 
        (conflict_values[i], refugee_values[i]), 
        xytext=(5, 5), 
        textcoords='offset points', 
        fontsize=9
    )

plt.xlabel('Total Conflicts (1990-2020)')
plt.ylabel('Total Refugees (1990-2020)')
plt.title('Refugee Population vs Conflicts for Top 10 Refugee Countries')
plt.grid(True, alpha=0.3)
plt.colorbar(label='Refugee Population')  # Show color scale
plt.show()

# Filter conflict data for years 2010 and onwards
conflicts_since_2010 = conflict[conflict['year'] >= 2010]

# Calculate total conflicts per country since 2010
total_conflicts_2010 = conflicts_since_2010.groupby('country')['conflict_count'].sum()

# Identify the country with the highest number of conflicts
highest_conflict = total_conflicts_2010.idxmax()
highest_conflict_count = total_conflicts_2010.max()

# Display results
print(f"Country with highest conflicts since 2010: {highest_conflict}")
print(f"Total conflicts: {highest_conflict_count}")
# Syria has the highest conflicts since 2010
# Select all years from 2000 to 2020
years = [str(y) for y in range(2000, 2021)]

# Filter refugee data for Syria
syria_data = x[x['Country'] == 'Syria']

# Extract yearly refugee values where data exists and is greater than 0
trend = [
    {'year': int(year), 'refugees': syria_data[year].iloc[0]}
    for year in years
    if year in syria_data.columns and pd.notna(syria_data[year].iloc[0]) and syria_data[year].iloc[0] > 0
]

# Convert to DataFrame
trend_df = pd.DataFrame(trend)

trend_df
trend_df['pct_change'] = trend_df['refugees'].pct_change() * 100
import matplotlib.pyplot as plt

# Plot Syria refugee trend (2000-2020)
plt.figure(figsize=(14, 7))  # Slightly larger figure for better readability
plt.plot(
    trend_df['year'], 
    trend_df['refugees'], 
    marker='o', 
    linewidth=2, 
    color='mediumseagreen'  # Attractive and professional color
)
plt.title('Syria Refugee Population Trend (2000-2020)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Refugee Population', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(trend_df['year'], rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Display the data
print("Syria refugee trend (2000-2020):")
print(trend_df)
# Check if 'Syrian Arab Republic' exists in the dataset
syria_data = x[x['Country'] == 'Syrian Arab Republic']
print(f"Syria data found: {len(syria_data)} rows")

if len(syria_data) == 0:
    # If not found, list all countries containing 'Syria' in their name
    print("Available countries with 'Syria':")
    print(x[x['Country'].str.contains('Syria', case=False, na=False)]['Country'].values)
else:
    # Extract refugee trend for years 2000-2020
    years_2000_2020 = [str(year) for year in range(2000, 2021)]
    
    trend = []
    for year in years_2000_2020:
        if year in syria_data.columns:
            value = syria_data[year].iloc[0]
            # Include only valid refugee numbers (>0)
            if pd.notna(value) and value > 0:
                trend.append({'year': int(year), 'refugees': value})
    
    # Convert to DataFrame
    trend_df = pd.DataFrame(trend)
    print(trend_df)
# Calculate mean refugees per country (1990-2020)
years= [str(year) for year in range(1990, 2021)]
available_year = [year for year in years if year in x.columns]

means = x[['Country'] + available_year].copy()
means['mean_refugees'] = means[available_year].mean(axis=1, skipna=True)
means = means[['Country', 'mean_refugees']].dropna()

# Calculate mean conflicts per country (1990-2020)
conflict_1990_2020 = conflict[
    (conflict['year'] >= 1990) & 
    (conflict['year'] <= 2020)
]
conflict_mean = conflict_1990_2020.groupby('country')['conflict_count'].mean().reset_index()
conflict_mean.columns = ['Country', 'mean_conflicts']

# Merge datasets
corr_data = means.merge(conflict_mean, on='Country', how='inner')
corr_data = corr_data[(corr_data['mean_refugees'] > 0) & (corr_data['mean_conflicts'] > 0)]
# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(corr_data['mean_conflicts'], corr_data['mean_refugees'], alpha=0.6, s=60)

import matplotlib.pyplot as plt
import seaborn as sns

# Set figure size for better readability
plt.figure(figsize=(10, 8))

# Calculate correlation between mean conflicts and mean refugees
corr = corr_data[['mean_conflicts', 'mean_refugees']].corr()

# Create heatmap
sns.heatmap(
    corr, 
    annot=True,               # Show correlation values
    fmt='.3f',                # 3 decimal places
    cmap='YlGnBu',            # Attractive gradient color
    square=True,              # Square cells
    cbar_kws={'shrink': 0.8} # Shrink colorbar slightly for aesthetics
)

# Add title
plt.title('Mean Refugee Population vs Mean Conflicts (1990-2020)\nCorrelation', fontsize=14)
plt.tight_layout()            # Adjust layout to prevent clipping
plt.show()

import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Twitter dataset
df = pd.read_csv('tweets.csv') 


# Filter for verified users with at least 100 followers and available locations
tweet_df = df[
    (df['user_verified'] == True) & 
    (df['user_followers'] >= 100) & 
    (df['user_location'].notna()) & 
    (df['user_location'] != '')
].copy()
# Randomly sample 500 tweets
if len(tweet_df) >= 500:
    tweet_df = tweet_df.sample(n=500, random_state=42)
else:
    new_df = tweet_df.copy()
    print(f"Only {len(filtered)} tweets available with the criteria")

import re
import pandas as pd

# Function to preprocess and clean text (tweets)
def clean(text):
    if pd.isna(text):  # Return empty string if input is NaN
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions (@username) and hashtags (#tag)
    text = re.sub(r'[@#]\w+', '', text)
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

# Apply text cleaning
tweet_df['cleaned_text'] = tweet_df['tweets'].apply(clean)

# Remove empty tweets after cleaning
tweet_df = tweet_df[tweet_df['cleaned_text'] != ''].reset_index(drop=True)
tweet_df
cleaned_df = tweet_df[['user_location', 'cleaned_text', 'user_verified', 'user_followers']].copy()
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
# Initialize geocoder
geolocator = Nominatim(user_agent="sentimental_analysis")
from tqdm import tqdm
import time

# Function to geocode a user location safely
def geocode_location(user_location):
    """
    Returns the latitude and longitude for a given location string.
    If geocoding fails or location not found, returns (None, None).
    """
    try:
        time.sleep(1)  # Prevent rate-limiting
        location = geolocator.geocode(user_location, timeout=10)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception:
        return None, None

# Get all unique user locations
unique_locations = cleaned_df['user_location'].dropna().unique()
print(f"Geocoding {len(unique_locations)} unique locations...")

# Initialize cache dictionary
location_cache = {}

# Loop through locations with a progress bar
for loc in tqdm(unique_locations, desc="Geocoding"):
    location_cache[loc] = geocode_location(loc)

print("Geocoding complete!")
# Apply cached results
cleaned_df['coords'] = cleaned_df['user_location'].map(location_cache)
cleaned_df['latitude'] = cleaned_df['coords'].apply(lambda x: x[0] if x else None)
cleaned_df['longitude'] = cleaned_df['coords'].apply(lambda x: x[1] if x else None)
geocoded_df = cleaned_df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
import folium
from folium import plugins
# Calculate polarity for each tweet
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

geocoded_df['polarity'] = geocoded_df['cleaned_text'].apply(get_polarity)
# Group by location and calculate average polarity
polarity = geocoded_df.groupby(['latitude', 'longitude']).agg({
    'polarity': 'mean',
    'user_location': 'first',
    'cleaned_text': 'count'
}).rename(columns={'cleaned_text': 'tweet_count'}).reset_index()
# Create polarity visualization map
def polarity_map(data):
    # Center map on mean coordinates
    lat = data['latitude'].mean()
    lon = data['longitude'].mean()
    
    m = folium.Map(user_location=[lat, lon], zoom_start=2)
    
    # Add markers with color based on polarity
    for idx, row in data.iterrows():
        # Color based on polarity: red (negative), yellow (neutral), green (positive)
        if row['polarity'] > 0.1:
            color = 'green'
        elif row['polarity'] < -0.1:
            color = 'red'
        else:
            color = 'yellow'
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=f"Location: {row['user_location']}<br>Polarity: {row['polarity']:.3f}<br>Tweets: {row['tweet_count']}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

import matplotlib.pyplot as plt

# Create scatter plot of geocoded tweets with polarity
plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    geocoded_df['longitude'], 
    geocoded_df['latitude'], 
    c=geocoded_df['polarity'],    # Color based on polarity
    cmap='coolwarm',              # Attractive color map
    s=120,                        # Bubble size
    alpha=0.8,                    # Transparency
    edgecolors='black'            # Bubble borders
)

# Add colorbar
plt.colorbar(scatter, label='Polarity')

# Add labels and title
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Geospatial Distribution of Tweet Polarity', fontsize=16)

# Add grid
plt.grid(True, alpha=0.3)

# Display plot
plt.show()

# Display polarity statistics
print("Polarity statistics:")
print(geocoded_df['polarity'].describe())
# Calculate subjectivity for each tweet
def subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

geocoded_df['subjectivity'] = geocoded_df['cleaned_text'].apply(subjectivity)

# Group by location and calculate average subjectivity
loc_subjectivity = geocoded_df.groupby(['latitude', 'longitude']).agg({
    'subjectivity': 'mean',
    'user_location': 'first',
    'cleaned_text': 'count'
}).rename(columns={'cleaned_text': 'tweet_count'}).reset_index()
# Create subjectivity visualization map
def subjectivity_map(data):
    lat = data['latitude'].mean()
    lon = data['longitude'].mean()
    
    m = folium.Map(user_location=[lat, lon], zoom_start=2)
    
    # Add markers with color based on subjectivity
    for idx, row in data.iterrows():
        # Color based on subjectivity: blue (objective), red (subjective)
        if row['subjectivity'] > 0.6:
            color = 'red'
        elif row['subjectivity'] > 0.3:
            color = 'orange'
        else:
            color = 'blue'
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=f"Location: {row['user_location']}<br>Subjectivity: {row['subjectivity']:.3f}<br>Tweets: {row['tweet_count']}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    return m
import matplotlib.pyplot as plt

# Scatter plot of geocoded tweets by subjectivity
plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    geocoded_df['longitude'], 
    geocoded_df['latitude'], 
    c=geocoded_df['subjectivity'],  # Color represents subjectivity
    cmap='viridis',                 # Attractive color map
    s=120,                          # Slightly larger bubble size
    alpha=0.8,                      # Transparency
    edgecolors='black'              # Bubble borders for contrast
)

# Add colorbar
plt.colorbar(scatter, label='Subjectivity')

# Add labels and title
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Geospatial Distribution of Tweet Subjectivity', fontsize=16)

# Add grid
plt.grid(True, alpha=0.3)  

# Display plot
plt.show()

# Print statistics
print("Subjectivity statistics:")
print(geocoded_df['subjectivity'].describe())
