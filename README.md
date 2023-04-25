# CS661_Big_Data_SpotifyFeatures Data Analysis

# Dashboard using Plotly Dash

This Python script creates a web application using the Dash library and Plotly visualizations to explore and present insights from two data sources:

- spotifyfeatures.csv: A CSV file containing data on music tracks from the Spotify API.
- archive/tracks.csv: A CSV file containing additional data on music tracks.

## Requirements

- Python 3.x
- Dash
- Plotly
- NumPy
- Pandas
- scikit-learn

## Installation

1. Clone or download this repository to your local machine.
2. Install the required packages using pip: `pip install dash plotly numpy pandas scikit-learn`

## How to run 

1. Navigate to the directory where you downloaded or cloned this repository.
2. from this link https://drive.google.com/drive/folders/1XvKKf_n9RiSD9igwsJAFl4pj8CWisNSu?usp=share_link download SpotifyFeatures.csv and archive.zip 
3. unzip archive.zip and place this two files in the directory of dashboard.py 
4. Run the Python script using the command `python dashboard.py`.
5. Open a web browser and navigate to `http://localhost:4050/` to view the web application.

## Functionality

The web application contains several pages, each with interactive visualizations that allow users to explore the data:

- Overview: A general overview of the data, including scatterplots and histograms of various features.
- Similarity: A text analysis page that uses the `CountVectorizer` and `cosine_similarity` functions from scikit-learn to show the similarity between tracks.
- Archive: A page that displays additional information on the tracks from the `archive/tracks.csv` file.

Users can interact with the visualizations and select data points to see more detailed information. The visualizations are updated in real-time based on user input using the `Input` and `Output` classes from the Dash library.

## Credits

This script was created by [Group]. Please feel free to contact me with any questions or feedback.
