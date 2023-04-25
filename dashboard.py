#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash 
import dash_core_components as dcc
import plotly.graph_objs as go
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
import base64

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# In[2]:


song_df=pd.read_csv('archive/tracks.csv')


# In[3]:


# Load the data   for song recommendation
df = pd.read_csv('SpotifyFeatures.csv')

# Randomly select a subset of 10,000 rows
subset_indices = np.random.choice(df.index, size=10000 , replace=False)
subset_data = df.loc[subset_indices]

# Create the document-term matrix
cv = CountVectorizer()
dtm = cv.fit_transform(subset_data['artist_name'])

# Calculate the cosine similarity
cosine_sim = cosine_similarity(dtm, dtm)

subset_data.reset_index(drop=True, inplace=True)

# Extract the sound-related parameters and metadata into separate dataframes
sound_params = subset_data[['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']]
metadata = subset_data.drop(['acousticness', 'danceability','energy', 'instrumentalness', 'key', 
                       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence'], axis=1)

# Scale the sound-related parameters
scaler = MinMaxScaler()
sound_features = pd.DataFrame()
for col in sound_params.select_dtypes(include=[np.number]).columns:
    sound_features[col] = scaler.fit_transform(sound_params[col].values.reshape(-1,1)).ravel()

# Combine the metadata and sound features into a single dataframe
mdf = metadata.join(sound_features)




div10=html.Div(children=[
    html.H1(children='Song Recommendation System',style={'background-color': '#1DB954',
    'color': '#FFFFFF'
}),

    dcc.Input(
        id='input-box',
        type='text',
        placeholder='Enter a song name'
    ),

    html.Div(id='output-container',style={
            'padding-top': '50px',
            'padding-bottom': '50px'
        })
])


# In[4]:


song_popularity=song_df[['name','popularity']].copy().sort_values(by=['popularity'],ascending=False, ignore_index=True)


# In[5]:


song_popularity=song_popularity[:15].sort_index(ascending=False)
# song_popularity


# In[6]:


artist_df=pd.read_csv('archive/artists.csv')


# In[7]:


artist_popularity=artist_df[['name','popularity','followers']].copy().sort_values(by=['followers','popularity'], ascending=False, ignore_index=True)


# In[8]:


artist_popularity=artist_popularity[:15].sort_index(ascending=False)


# In[9]:


genre_df = pd.read_csv('SpotifyFeatures.csv')


# In[10]:


genre_popularity = genre_df.groupby('genre')['popularity'].mean().reset_index()


# In[13]:


genre_df = pd.read_csv('SpotifyFeatures.csv')
# Get the frequency of each key in each genre.
key_freq_df = genre_df.groupby(['genre', 'key']).size().reset_index(name='count')

# Get the frequency of each time signature in each genre.
time_sig_freq_df = genre_df.groupby(['genre', 'time_signature']).size().reset_index(name='count')

genre_options=[{'label': genre, 'value': genre} for genre in genre_df['genre'].unique()]


# Define the audio features to include in the dropdown
audio_features = ['danceability', 'energy', 'instrumentalness']


# # Create scatterplot
genre_df['duration_s']=round(genre_df['duration_ms']/1000)
fig = px.scatter(genre_df, x="duration_s", y="popularity")

genres_df = genre_df.sample(int(0.009 * len(genre_df)))

corr = genre_df.corr().abs()

top_artists = genre_df.groupby("artist_name")["popularity"].sum().sort_values(ascending=False).head(20)


fig.update_layout(
    plot_bgcolor='black',  # Set background color
    paper_bgcolor='black',  # Set paper color
    font_color='white',  # Set font color
    xaxis=dict(showgrid=False),  # Hide x-axis gridlines
    yaxis=dict(showgrid=False)  # Hide y-axis gridlines
)

# Open the image file and read it in binary mode
with open("spotify.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# Get the original height and width of the image
import PIL.Image
with PIL.Image.open("spotify.jpg") as img:
    original_width, original_height = img.size


div1=html.Div([
#     html.H1('My Dashboard'),
    html.Div([
        html.Img(src="data:image/jpeg;base64,{}".format(encoded_image), 
                 style={'width': '60%', 'height': '{}px'.format(int(original_height * 8 / original_width * 100))}),
    ], style={
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
    })
])


genre_dropdown_1=dcc.Dropdown(
        id='genre-dropdown_1',
        options=[{'label': i, 'value': i} for i in genre_df['genre'].unique()],
        value='Pop',
        style={'width': '300px', 'height': '40px','color': '#000000'}
        )



div2=html.Div([
    genre_dropdown_1,
    html.Div(children=[
    dcc.Graph(id='genre-graph',
            style={'width': '50%','backgroundColor': '#000000','color': '#FFFFFF', 'display': 'inline-block'}),
    dcc.Graph(
        id='genre_popularity',
        figure={
            'data': [
                go.Bar(
                    x=genre_popularity['genre'],
                    y=genre_popularity['popularity'],
                    name='Average Popularity',
                    marker={'color': '#1DB954'}  # set the color of the bars to green
                )
            ],
            'layout': go.Layout(
                title='Genre popularity',
                xaxis={'title': 'Genre'},
                yaxis={'title': 'Popularity'},
                barmode='stack',
                plot_bgcolor='#000000',  # set the plot background color to black
                paper_bgcolor='#000000',  # set the paper background color to black
                font={'color': '#FFFFFF'}
            )
        },style={'width': '50%', 'display': 'inline-block'}
    )
     
])

])







div3= html.Div(children=[
    html.H2(children='Frequency of Keys and Time Signatures in Different Genres'),
    html.H3('Filter Options'),
    
    # Add the dropdown menu.
    dcc.Dropdown(
        id='genre-dropdown',
        options=genre_options,
        value=genre_options[0]['value'],
        searchable=False,
        clearable=False,
        style={'width': '300px', 'height': '40px','color': '#000000'}
    ),
    
    html.Div(children=[
        dcc.Graph(
            id='key-freq-chart',
            # Modify the figure to use the selected genre.
            figure=px.bar(key_freq_df[key_freq_df['genre'] == genre_options[0]['value']], x='genre',
            y='count', color='key', barmode='group').update_layout(title='Frequency of Keys in Different Genres',
                plot_bgcolor='#000000',  # set the plot background color to black
                paper_bgcolor='#000000',  # set the paper background color to black
                font={'color': '#FFFFFF'}
            ),
            style={'width': '50%'}
        ),
        dcc.Graph(
            id='time-sig-freq-chart',
            # Modify the figure to use the selected genre.
            figure=px.bar(time_sig_freq_df[time_sig_freq_df['genre'] == genre_options[0]['value']], x='genre',
            y='count', color='time_signature', barmode='group').update_layout(title='Frequency of Time Signatures in Different Genres',
                plot_bgcolor='#000000',  # set the plot background color to black
                paper_bgcolor='#000000',  # set the paper background color to black
                font={'color': '#FFFFFF'}                                                                    
            ),
            style={'width': '50%'}
        )
    ], style={'display': 'flex', 'flex-wrap': 'wrap'}),
])

           







div4=html.Div([
    html.H2("Relationship between Audio Features and Popularity"),
    
    # Add a dropdown to select the audio feature for the scatterplot
    dcc.Dropdown(
        id='audio-feature',
        options=[{'label': feature.capitalize(), 'value': feature} for feature in audio_features],
        value=audio_features[0],
        clearable=False,
        style={'width': '300px', 'height': '40px','color': '#000000'}
    ),
    
    dcc.Graph(
        id='scatterplot',
        figure={
            'data': [
                {'x': genres_df[audio_features[0]], 'y': genres_df['popularity'], 'type': 'scatter', 'mode': 'markers'}
            ],
            'layout': {
                'title': f"{audio_features[0].capitalize()} vs Popularity",
                 'plot_bgcolor': '#000000',  # set the plot background color to black
                 'paper_bgcolor': '#000000',  # set the paper background color to black
                 'font': {'color': '#FFFFFF'}
                
                
            }
        }
    )
])





div5=html.Div([html.Div(className='left-filters', children=[
            html.Label('Select a variable:'),
            dcc.RadioItems(
                id='variable-radio',
                options=[{'label': 'Duration (ms)', 'value': 'duration'},
                         {'label': 'Tempo (BPM)', 'value': 'tempo'}],
                value='duration'
            ),
            html.Br(),
            html.Label('Select genre(s):'),
            dcc.Dropdown(
                id='genre-dropdown1',
                options=[{'label': genre, 'value': genre} for genre in genre_df['genre'].unique()],
                value=[],
                multi=True,
                clearable=False,
                style={'color': '#000000'}
            )
        ],
        style={
            'width': '30%',
            'display': 'inline-block',
            'vertical-align': 'top',
            'border': '1px solid black',  # Add a border
            'padding': '10px', # Add padding for better visual appearance
        }),
        html.Div(className='right-graphs', children=[
           dcc.Graph(
                id='boxplot1',
                figure={
                'layout': {
                'plot_bgcolor': '#000000',
                'paper_bgcolor': '#000000',
                'font': {'color': '#FFFFFF'}
                    }
                }
                )
        ],
        style={
            'width': '65%',
            'display': 'inline-block',
            'vertical-align': 'top',
        }),
    ])



# div6= html.Div([
#     html.H1("Relationship between Audio Features and Key/Time Signature of Songs"),
#     html.Label("Select an Audio Feature"),
#     dcc.Dropdown(
#         id='audio_feature_dropdown',
#         options=[
#             {'label': 'Danceability', 'value': 'danceability'},
#             {'label': 'Energy', 'value': 'energy'},
#             {'label': 'Instrumentalness', 'value': 'instrumentalness'},
#             # Add more audio features here
#         ],
#         value='danceability',
#         style={'width': '300px', 'height': '40px','color': '#000000'}
#     ),
    
#     html.Div(children=[
#         dcc.Graph(id='scatter-plot'),
        
#        dcc.Graph(
#             id="scatterplot2",
#             figure=fig,
#             style={'width': '50%'},
            
#         )
#     ], style={'display': 'flex', 'flex-wrap': 'wrap'})
    
# ])

div6= html.Div([
    html.H1("Relationship between Audio Features and Key/Time Signature of Songs"),
    html.Label("Select an Audio Feature"),
    dcc.Dropdown(
        id='audio_feature_dropdown',
        options=[
            {'label': 'Danceability', 'value': 'danceability'},
            {'label': 'Energy', 'value': 'energy'},
            {'label': 'Instrumentalness', 'value': 'instrumentalness'},
            # Add more audio features here
        ],
        value='danceability',
        style={'width': '300px', 'height': '40px','color': '#000000'}
    ),
    
    html.Div(children=[
        dcc.Graph(id='scatter-plot'),
        
       dcc.Graph(
            id="scatterplot2",
            figure=fig,
            style={'width': '50%'}
            
        )
    ], style={'display': 'flex', 'flex-wrap': 'wrap'})
    
])



# Define the layout
div7 = html.Div([
    html.H1('Artist Popularity according to their followers'),
    dcc.Graph(
        id='artist-popularity-graph',
        figure={
            'data': [go.Bar(
                x=artist_popularity['followers'],
                y=artist_popularity['name'],
                orientation='h',
                marker=dict(
                    color='#1DB954',
                    line=dict(
                        color='rgba(50, 171, 96, 1.0)',
                        width=1),
                ),
                name='Artist Popularity according to their followers'
            )],
            'layout': go.Layout(
                yaxis=dict(
                    showgrid=False,
                    showline=False,
                    showticklabels=True,
                ),
                xaxis=dict(
                    zeroline=False,
                    showline=False,
                    showticklabels=True,
                    showgrid=False,
                ),
                legend=dict(x=1, y=-0.1, font_size=10),
                margin=dict(l=150, r=50, t=50, b=50),
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                font= {'color': '#FFFFFF'}
                
                
                
            )
        }
    )
])
           
    
div8=html.Div([
    html.Div(children=[
    dcc.Graph(
        id='heatmap',
        figure={
            'data': [go.Heatmap(
                z=corr.values,
                x=corr.index,
                y=corr.columns,
                colorscale='Greens',
                colorbar=dict(
                    title='Correlation',
                    titleside='right',
                    ticks='outside',
                    ticklen=3,
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=['-1', '-0.5', '0', '0.5', '1']
                ),
                showscale=True,
                reversescale=True,
                zmin=-1,
                zmax=1,
                hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>'
            )],
            'layout': go.Layout(
                title={
                        'text': 'Correlation Heatmap',
                        'font': {'color': 'white'}
                    },
                xaxis=dict(title='X attributes', color='white'),
                yaxis=dict(title='Y attributes',color='white', autorange='reversed'),
                plot_bgcolor='black',  # set plot background color
                paper_bgcolor='black',  # set paper background color,
                height=500
            )
        },style={'width': '50%','backgroundColor': '#000000','color': '#FFFFFF', 'display': 'inline-block'}),
    dcc.Graph(
        id='bar-chart',
        figure={
            'data': [go.Bar(
                x=top_artists.index,
                y=top_artists,
                marker={'color': top_artists.values,
                        'colorscale': 'Greens'}
            )],
            'layout': {
                'title': {
                            'text': 'Top Artists by Popularity',
                            'font': {'color': 'white'}
                        },
                'xaxis': {'title': 'Artists', 'tickangle': -45,'color': 'white'},
                'yaxis': {'title': 'Popularity','color': 'white'},
                'height': 500,
                'margin': {'b': 150},
                'plot_bgcolor': 'black',  # set plot background color
                'paper_bgcolor': 'black'  # set paper background color
            }
        },style={'width': '50%', 'display': 'inline-block'}) 
])

])


#creating dataset for Time_Series_Analysis
data_tsa = pd.DataFrame({'year':song_df['release_date'].str[:4],'duration_s':song_df['duration_ms']/1000, 'danceability':song_df['danceability'],'energy':song_df['energy'],'loudness':song_df['loudness'],'speechiness':song_df['speechiness'],'acousticness':song_df['acousticness'],'instrumentalness':song_df['instrumentalness'],'liveness':song_df['liveness'],'valence':song_df['valence'],'tempo':song_df['tempo']})
data_tsa['year'][478627]='2014'

options = [{'label': col, 'value': col} for col in data_tsa.columns[2:]]


div9=html.Div([
    dcc.Dropdown(
        id='column-dropdown',
        options=options,
        value=options[0]['value'],
        style={'width': '300px', 'height': '40px','color': '#000000'}
    ),
    dcc.Graph(id='tsa-graph',style={'backgroundColor': 'black'})
])



    
    
    




app = dash.Dash()


app.layout = html.Div(children=[
    div1,
    div10,
    div2,
    div3,
    div4,
    div5,
    div6,
    div7,
    div8,
    div9
],style={'backgroundColor': '#000000','color': '#FFFFFF','padding': '20px', 'marginBottom': '30px'})



# Define the callback function
@app.callback(
    Output('genre-graph', 'figure'),
    Input('genre-dropdown_1', 'value')
)
def update_genre_graph(selected_genre):
    filtered_df = genre_df[genre_df['genre'] == selected_genre].sort_values(by='popularity', ascending=False).head(10)
    
 
    
    fig = go.Figure([go.Bar(x=filtered_df['track_name'], y=filtered_df['popularity'], 
                            marker={'color': '#1DB954'},width=0.5)])  # set bar color to green
    fig.update_layout(title=f'Top 10 Most Popular Songs in {selected_genre} Genre', 
                      plot_bgcolor='#000000',  # set plot background color to black
                      paper_bgcolor='#000000',  # set paper background color to black
                      font=dict(color='#FFFFFF'),  # set label color to white
                      xaxis={
                            'tickmode': 'array',
                            'ticktext': [label[:15] for label in filtered_df['track_name']],
                            'tickvals': filtered_df['track_name'],
                            'showgrid': False
                        },# remove grid for x-axis
                      yaxis=dict(showgrid=False))  # remove grid for y-axis
    return fig


# Define the callback function to update the charts when the dropdown selection changes.
@app.callback(
    Output('key-freq-chart', 'figure'),
    Output('time-sig-freq-chart', 'figure'),
    Input('genre-dropdown', 'value')
)
def update_charts(genre):
    # Update the figures to use the selected genre.
    key_fig = px.bar(key_freq_df[key_freq_df['genre'] == genre], x='genre', y='count', color='key', barmode='group').update_layout(title='Frequency of Keys in Different Genres',
    plot_bgcolor='#000000',  # set the plot background color to black
    paper_bgcolor='#000000',  # set the paper background color to black
    font={'color': '#FFFFFF'})
    time_sig_fig = px.bar(time_sig_freq_df[time_sig_freq_df['genre'] == genre], x='genre', y='count', color='time_signature', barmode='group').update_layout(title='Frequency of Time Signatures in Different Genres',
    plot_bgcolor='#000000',  # set the plot background color to black
    paper_bgcolor='#000000',  # set the paper background color to black
    font={'color': '#FFFFFF'})
    return key_fig, time_sig_fig


# Add a callback to update the scatterplot  when the dropdown value changes
@app.callback(
    Output('scatterplot', 'figure'),
    Input('audio-feature', 'value')
)
def update_scatterplot(audio_feature):
    data = [
        {'x': genres_df[audio_feature], 'y': genres_df['popularity'], 'type': 'scatter', 'mode': 'markers'}
    ]
    layout = {
        'title': f"{audio_feature.capitalize()} vs Popularity",
         'plot_bgcolor': '#000000',  # set the plot background color to black
         'paper_bgcolor': '#000000',  # set the paper background color to black
         'font': {'color': '#FFFFFF'}
        
    }
    scatterplot = {'data': data, 'layout': layout}
    return scatterplot




# Define callback function to update boxplot based on radio and dropdown values
@app.callback(
    dash.dependencies.Output('boxplot1', 'figure'),
    [dash.dependencies.Input('variable-radio', 'value'),
     dash.dependencies.Input('genre-dropdown1', 'value')])
def update_boxplot(variable, genre_values):
    if genre_values is None or len(genre_values) == 0:
        genre_values = ['Movie']  # set a default value for genre_values
    
    if variable == 'duration':
        title = "Song Duration Distribution by Genre"
        y_axis = "duration_ms"
    else:
        title = "Tempo Distribution by Genre"
        y_axis = "tempo"
    
    filtered_df = genre_df[genre_df['genre'].isin(genre_values)]
    boxplot1= px.box(filtered_df, x="genre", y=y_axis, title=title)
    boxplot1.update_traces(marker={'color': ' #1DB954'})
    boxplot1.update_layout(
        plot_bgcolor='#000000',  # set the plot background color to black
        paper_bgcolor='#000000',  # set the paper background color to black
        font={'color': '#FFFFFF'},  # set the font color to white
       
    )
    
    
    return boxplot1




# Define the callback function for the scatterplot
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('audio_feature_dropdown', 'value')]
)
def update_scatterplot(audio_feature):
    fig = px.scatter(genre_df, x='key', y='time_signature', color=audio_feature, title=f"{audio_feature} vs Key/Time Signature")
    
    # Set the plot color to black
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black')
    
    # Set the label color to white
    fig.update_layout(font=dict(color='white'))
    
    # Remove the grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    return fig


# Define the callback function
@app.callback(
    dash.dependencies.Output('scatterplot2', 'figure'),
    [dash.dependencies.Input('genre-dropdown_1', 'value')]
)
def update_scatter_plot(genre):
    filtered_df = genre_df[genre_df['genre'] == genre]
    fig = px.scatter(filtered_df, x='duration_s', y='popularity', color='genre')
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    # Remove the grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig




# define callback to update graph based on dropdown selection
@app.callback(Output('tsa-graph', 'figure'), [Input('column-dropdown', 'value')])
def update_graph(column):
    fig = px.line(data_tsa.groupby('year')[column].mean().reset_index(),
                  x='year', y=column, title=column)
    fig.update_xaxes(dtick=10, showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_traces(line=dict(color=' #1DB954'))
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')

    return fig




@app.callback(
    Output('output-container', 'children'),
    Input('input-box', 'value')
)
def recommend_songs_same_artist(title):
    if title is not None and title != '':
        # Get the index of the song within the subset DataFrame
        idx = subset_data[subset_data["track_name"]==title].index.tolist()

        if len(idx) > 0:
            idx = idx[0]

            # Compute the similarity and organize in a sorted list by highest sim
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]

            # Get the titles of the recommended songs
            song_indices = [i[0] for i in sim_scores]
            rec_songs = subset_data.iloc[song_indices]['track_name'].values.tolist()

            return html.Ul([html.Li(rec_song) for rec_song in rec_songs])
        else:
            return "Song not found in data."
    else:
        return ""

print(subset_data['track_name'])


# In[14]:


if __name__ =='__main__':
    app.run_server(port=4050)

