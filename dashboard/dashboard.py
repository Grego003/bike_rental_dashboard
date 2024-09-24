import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import regex as re
from os import path


def replace_outliers(df, filter=[]):
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        if col in filter:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        min_range = q1 - (iqr * 1.5)
        max_range = q3 + (iqr * 1.5)

        df.loc[df[col] < min_range, col] = min_range
        df.loc[df[col] > max_range, col] = max_range

    return df


@st.cache_data
def get_cleaned_data():
    data_directory = "data"
    day_csv_filename = "day.csv"
    hour_csv_filename = "hour.csv"

    day_csv_file_path = path.join(data_directory, day_csv_filename)
    hour_csv_file_path = path.join(data_directory, hour_csv_filename)

    day_df = pd.read_csv(day_csv_file_path)
    hour_df = pd.read_csv(hour_csv_file_path)

    filtered_col = ["holiday", "weathersit", "season", "yr", "weekday", "workingday"]

    replace_outliers(day_df, filtered_col)
    replace_outliers(hour_df, filtered_col)

    day_df["cnt"] = day_df["casual"] + day_df["registered"]
    hour_df["cnt"] = hour_df["casual"] + hour_df["registered"]

    return day_df, hour_df


@st.cache_data
def get_weather_and_season_data():
    day_df, _ = get_cleaned_data()

    seasons = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}

    season_stats_df = (
        day_df.groupby("season")
        .agg(
            {
                "temp": "mean",
                "atemp": "mean",
                "hum": "mean",
                "windspeed": "mean",
                "casual": "sum",
                "registered": "sum",
                "cnt": "sum",
            }
        )
        .reset_index()
    )

    season_stats_df["temp"] = season_stats_df["temp"].apply(lambda x: x * 42)
    season_stats_df["atemp"] = season_stats_df["atemp"].apply(lambda x: x * 50)
    season_stats_df["hum"] = season_stats_df["hum"].apply(lambda x: x * 100)
    season_stats_df["windspeed"] = season_stats_df["windspeed"].apply(lambda x: x * 67)
    season_stats_df["season"] = season_stats_df["season"].apply(lambda x: seasons[x])

    season_stats_df.rename(
        columns={
            "cnt": "total",
        },
        inplace=True,
    )

    return season_stats_df


def plot_weather_and_season(user_type, season):

    user_type_color = {"casual": "#4169E1", "registered": "#FF7F0E", "total": "#28A745"}
    highlighted_color = ["#FF5733", "#28A745", "#4169E1"]
    user_color = user_type_color[user_type]
    user_types = list(user_type_color.keys())

    season_stats_df = get_weather_and_season_data()
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = sns.barplot(
        data=season_stats_df, x="season", y=user_type, ax=ax, color=user_color
    )

    if season:
        bars = sns.barplot(
            data=season_stats_df, x="season", y=user_type, ax=ax, color=user_color
        )

    for bar in bars.patches:
        season_name = season_stats_df.iloc[
            int(bar.get_x() + bar.get_width() / 2)
        ].season
        if season_name == season:
            bar.set_color(highlighted_color[user_types.index(user_type)])

    ax.set_title(f"{user_type.capitalize()} Users by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel(f"{user_type.capitalize()} Users")

    return fig


def extracted_weather_factors(season):
    season_stats_df = get_weather_and_season_data()
    weather_factors = season_stats_df.loc[
        season_stats_df["season"] == season, ["temp", "atemp", "hum", "windspeed"]
    ]

    temp = weather_factors["temp"].apply(lambda val: f"{val:.1f}").values
    atemp = weather_factors["atemp"].apply(lambda val: f"{val:.1f}").values
    hum = weather_factors["hum"].apply(lambda val: f"{val:.1f}").values
    windspeed = weather_factors["windspeed"].apply(lambda val: f"{val:.1f}").values

    return temp[0], atemp[0], hum[0], windspeed[0]


def plot_weather_matrix(weather_col, user_col):

    column = weather_col + user_col

    weather_effect_corr = day_df[column].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        weather_effect_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
    )

    ax.set_title("Impact of Weather Factors on Casual and Registered Users")
    plt.tight_layout()
    plt.show()

    return fig


@st.cache_data
def get_rental_trends_date_data():
    month = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    year = {0: "2011", 1: "2012"}

    bikes_trend_date_df = (
        day_df[["mnth", "yr", "casual", "registered", "cnt"]]
        .groupby(["mnth", "yr"])
        .agg({"casual": "sum", "registered": "sum", "cnt": "sum"})
        .sort_values(by=["yr", "mnth"])
        .reset_index()
    )

    bikes_trend_date_df["mnth"] = bikes_trend_date_df["mnth"].apply(lambda x: month[x])
    bikes_trend_date_df["yr"] = bikes_trend_date_df["yr"].apply(lambda x: year[x])
    bikes_trend_date_df["date"] = (
        bikes_trend_date_df["mnth"] + " " + bikes_trend_date_df["yr"].astype(str)
    )

    return bikes_trend_date_df


def plot_rental_date_trends(user_type):

    user_type_color = {"casual": "#4169E1", "registered": "#FF7F0E"}

    bikes_trend_date_df = get_rental_trends_date_data()

    fig, ax = plt.subplots(1, figsize=(10, 6))

    sns.lineplot(
        data=bikes_trend_date_df,
        x="date",
        y=user_type,
        marker="o",
        ax=ax,
        color=user_type_color[user_type],
    )
    ax.set_xlabel("Month-Year")
    ax.set_ylabel(f"{user_type.capitalize()} Rentals")
    ax.set_title(f"{user_type.capitalize()} Rentals Trend by Month and Year")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="center")

    plt.tight_layout()
    plt.show()

    return fig


@st.cache_data
def get_rental_trends_time_data():

    hour_df["time_segment"] = hour_df["hr"].apply(categorize_time)

    time_trend_df = (
        hour_df.groupby(["time_segment"])
        .agg({"casual": "sum", "registered": "sum", "cnt": "sum"})
        .reset_index()
    )

    return time_trend_df


def categorize_time(hr):

    parts_of_the_day = {
        "Morning": (5, 12),  # 5 AM to 11:59 AM
        "Afternoon": (12, 17),  # 12 PM to 4:59 PM
        "Evening": (17, 21),  # 5 PM to 8:59 PM
        "Night": (21, 5),  # 9 PM to 4:59 AM
    }

    if hr >= 21 or hr < 5:
        return "Night"

    for parts, (start, end) in parts_of_the_day.items():
        if start <= hr < end:
            return parts
    return "Unknown"


def plot_rental_time_trends(user_type):
    time_trend_df = get_rental_trends_time_data()

    fig, ax = plt.subplots(1, figsize=(10, 6))

    cmap_casual = sns.color_palette("Blues", as_cmap=True)
    cmap_registered = sns.color_palette("Oranges", as_cmap=True)

    casual_norm = plt.Normalize(
        time_trend_df["casual"].min(), time_trend_df["casual"].max()
    )
    registered_norm = plt.Normalize(
        time_trend_df["registered"].min(), time_trend_df["registered"].max()
    )

    colors_casual = [
        cmap_casual(0.2 + (casual_norm(value) * 0.7))
        for value in time_trend_df[user_type]
    ]
    colors_registered = [
        cmap_registered(0.2 + (registered_norm(value) * 0.7))
        for value in time_trend_df[user_type]
    ]

    if user_type == "casual":
        ax.pie(
            time_trend_df["casual"],
            labels=time_trend_df["time_segment"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors_casual,
        )
        ax.set_title("Distribution of Casual Bike Rentals by Hour of the Day")
        ax.axis("equal")
    else:
        ax.pie(
            time_trend_df["registered"],
            labels=time_trend_df["time_segment"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors_registered,
        )
        ax.set_title("Distribution of registered Bike Rentals by Hour of the Day")
        ax.axis("equal")

    return fig


day_df, hour_df = get_cleaned_data()


st.title("Bike :red[Rental] Dashboard 🚴")

st.header(
    "The Impact of Weather Conditions and Seasonal Trends",
    anchor=False,
)

seasons = ["Spring 🌱", "Summer 🌻", "Fall 🍂", "Winter ☃️"]

season_picker = re.sub(
    r"\s+\S+",
    "",
    st.radio(
        label="Seasons",
        options=seasons,
        horizontal=True,
    ),
)

temp, atemp, hum, windspeed = extracted_weather_factors(season_picker)

user_types = [
    "casual",
    "registered",
    "total",
]

user_type = st.selectbox(label="Choose The User Types", options=user_types)

col1, col2 = st.columns([0.78, 0.22], gap="medium", vertical_alignment="center")

with col2:
    st.metric(label="Average Temp", value=f"{temp}°C")
    st.metric(label="Average Apparent Temp", value=f"{atemp}°C")
    st.metric(label="Average Windspeed", value=f"{windspeed} m/s")
    st.metric(label="Average Humidity", value=f"{hum}°%")

with col1:
    st.pyplot(plot_weather_and_season(user_type=user_type, season=season_picker))

st.subheader(
    "Correlation Matrix: Weather Factors and User Types",
    anchor=False,
)

col3, col4 = st.columns(2)

with col3:
    option_weather = st.multiselect(
        "Weather Factors",
        ["temp", "atemp", "hum", "windspeed"],
        ["temp", "atemp"],
        key="weather_type",
    )

with col4:
    option_user = st.multiselect(
        "User Types",
        ["casual", "registered"],
        ["casual"],
        key="user_type",
    )

st.pyplot(plot_weather_matrix(option_weather, option_user))

st.header("Bike Rental Trends By Month Over The Years")
tab1, tab2 = st.tabs(["casual", "registered"])
with tab1:
    st.pyplot(plot_rental_date_trends("casual"))
    data = get_rental_trends_date_data()
    st.write("Data Of Rental Trends By Month Over The Years")
    st.write(data[["date", "casual"]].T)
with tab2:
    st.pyplot(plot_rental_date_trends("registered"))
    data = get_rental_trends_date_data()
    st.write("Data Of Rental Trends By Month Over The Years")
    st.write(data[["date", "registered"]].T)

st.header("Differences In Peak Rental Hours Between Parts Of The Day")
tab3, tab4 = st.tabs(["casual", "registered"])
with tab3:
    st.pyplot(plot_rental_time_trends("casual"))
with tab4:
    st.pyplot(plot_rental_time_trends("registered"))
