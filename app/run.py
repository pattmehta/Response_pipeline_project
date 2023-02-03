import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# import parent module
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from models.train_classifier import ML

import pickle

app = Flask(__name__)

def tokenize(text):
    '''
    returns tokens from input text
    
    input:
    - text string
    output:
    - list of tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
db_path = 'data/DisasterResponse_1.db'
tbl_name = 'MessagesCategoriesTable'
model_path = 'models/classifier.pkl'
model_path = f'../{model_path}'
engine = create_engine(f'sqlite:///../{db_path}')
df = pd.read_sql_table(tbl_name, engine)
# load model
# we can use joblib, or pickle to load model e.g. joblib.load(model_path) or pickle usage seen below
model = None
with open(model_path, 'rb') as pkl_file:
    model = pickle.load(pkl_file)
if not model:
    print('could not load the model')
    exit()

# data prep for plots start
df_genre_vc = df['genre'].value_counts()
df_genre_vc_x = list(df_genre_vc.index)
df_genre_vc_y = list(df_genre_vc.values)

df_category = df.loc[:,'aid_centers':'weather_related']
# index 1 is on purpose, to remove 'related' column
df_category_topten = df_category.sum().sort_values(ascending=False)[1:11]
df_category_sort_x = list(df_category_topten.index)
df_category_sort_y = list(df_category_topten.values)

df_multicategory = df.drop(['message','original','id'],axis=1).groupby(by='genre',group_keys=True)
selected_colnames = []
def categories_agg(c):
    '''
    aggregate function for data prep
    '''
    if c.name in df_category_topten.index.values and c.name not in ['related','direct_report','other_aid','request']:
        if c.name not in selected_colnames: selected_colnames.append(c.name)
        return c.sum()
    else: return False
df_multicategory_agg = df_multicategory.aggregate(categories_agg)
df_gc_x = list(df_multicategory_agg.index)

# message labels
df_mn = df[(df['money']==1)].loc[:,'genre':'weather_related']
ignore_columns = ['related','aid_related','money','other_aid','request','offer','aid_centers','genre']
def global_mn_agg(s):
    '''
    aggregate function that returns the sum of the column
    '''
    if s.name not in ignore_columns: return s.sum()
    else: return None

df_mn_agg_cleaned = df_mn.aggregate(global_mn_agg).dropna().sort_values(ascending=False)
df_mn_message_labels = list(df_mn_agg_cleaned.index)

selected_msglabels = []
def agg_mn(s):
    '''
    aggregate function that returns the sum of the column, used after `groupby`
    '''
    if s.name not in ['related','aid_related','money','other_aid','request','offer','aid_centers','genre'] and s.name in df_mn_message_labels[:5]:
        if s.name not in selected_msglabels: selected_msglabels.append(s.name)
        return s.sum()
    else: return None

mn_agg = df_mn.groupby(by='genre',group_keys=True).aggregate(agg_mn).dropna(axis=1)
mn_agg_x = list(mn_agg.index)
# data prep for plots end


# routes start
@app.route('/viz_genre')
def viz_genre():
    '''
    route for the first visualization
    '''
    ids = [1]
    graphJSON = create_genre_plot()
    return render_template('viz_genre.html', ids=ids, graph=graphJSON)

@app.route('/viz_category')
def viz_category():
    '''
    route for the second visualization
    '''
    ids = [1]
    graphJSON = create_category_plot()
    return render_template('viz_category.html', ids=ids, graph=graphJSON)

@app.route('/viz_multicategory')
def viz_multicategory():
    '''
    route for the third visualization
    '''
    ids = [1]
    graphJSON = create_multicategory_plot()
    return render_template('viz_multicategory.html', ids=ids, graph=graphJSON)

@app.route('/viz_multicategory_mn')
def viz_multicategory_mn():
    '''
    route for the fourth visualization
    '''
    ids = [1]
    graphJSON = create_multicategory_plot_for_mn()
    return render_template('viz_multicategory_mn.html', ids=ids, graph=graphJSON)
# routes end

@app.route('/')
@app.route('/index')
def index():
    '''
    index webpage displays cool visuals and receives user input text for model
    serves master.html file with other visualizations
    '''
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


# plots start
def create_genre_plot():
    # create a graph object that represents the visualization
    graph = \
        {
            'data': [
                Bar({
                    'marker': {'color': '#1F77B4', 'line': {'width': 1.0}},
                    'opacity': 1,
                    'orientation': 'v',
                    'x': df_genre_vc_x,
                    'xaxis': 'x',
                    'y': df_genre_vc_y,
                    'yaxis': 'y',
                })
            ],
            'layout': {
                'title': 'Distribution of Message Genres by Count',
                'autosize': False,
                'bargap': 0.5,
                'height': 480,
                'hovermode': 'closest',
                'margin': {'b': 52, 'l': 80, 'pad': 0, 'r': 63, 't': 57},
                'showlegend': False,
                'template': '...',
                'width': 640,
                'xaxis': {'anchor': 'y',
                        'showgrid': True,
                        'showline': True,
                        'title': 'Genre',
                        'side': 'bottom',
                        'tickfont': {'size': 10.0},
                        'ticks': '',
                        'type': 'category',
                        'category_orders': {'x':df_genre_vc_x},
                        'zeroline': False},
                'yaxis': {'anchor': 'x',
                        'domain': [0.0, 1.0],
                        'mirror': 'ticks',
                        'showgrid': True,
                        'showline': True,
                        'title': 'Count',
                        'side': 'left',
                        'tickfont': {'size': 10.0},
                        'ticks': 'inside',
                        'type': 'linear',
                        'zeroline': False}
            }
        }
    
    # encode plotly graphs in JSON
    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def create_category_plot():
    # create a graph object that represents the visualization
    graph = \
        {
            'data': [
                Bar({
                    'marker': {'color': '#1F77B4', 'line': {'width': 1.0}},
                    'opacity': 1,
                    'orientation': 'v',
                    'x': df_category_sort_x,
                    'xaxis': 'x',
                    'y': df_category_sort_y,
                    'yaxis': 'y',
                })
            ],
            'layout': {
                'title': 'Top 10 Message Categories by Count',
                'autosize': False,
                'bargap': 0.5,
                'height': 480,
                'hovermode': 'closest',
                'margin': {'b': 52, 'l': 80, 'pad': 0, 'r': 63, 't': 57},
                'showlegend': False,
                'template': '...',
                'width': 640,
                'xaxis': {'anchor': 'y',
                        'showgrid': True,
                        'showline': True,
                        'side': 'bottom',
                        'tick0': 0,
                        'tickfont': {'size': 10.0},
                        'ticks': '',
                        'type': 'category',
                        'category_orders': {'x':df_category_sort_x},
                        'zeroline': False},
                'yaxis': {'anchor': 'x',
                        'domain': [0.0, 1.0],
                        'mirror': 'ticks',
                        'showgrid': True,
                        'showline': True,
                        'title': 'Count',
                        'side': 'left',
                        'tickfont': {'size': 10.0},
                        'ticks': 'inside',
                        'type': 'linear',
                        'zeroline': False}
            }
        }
    
    # encode plotly graphs in JSON
    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def create_multicategory_plot():
    # create a graph object that represents the visualization
    graph = \
        {
            'data': [
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [-0.20833333333333331, 0.7916666666666667, 1.7916666666666665],
                    'xaxis': 'x',
                    'y': df_multicategory_agg[selected_colnames[0]],
                    'yaxis': 'y',
                    'name': selected_colnames[0]
                }),
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [-0.125, 0.875, 1.875],
                    'xaxis': 'x',
                    'y': df_multicategory_agg[selected_colnames[1]],
                    'yaxis': 'y',
                    'name': selected_colnames[1]
                }),
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [-0.04166666666666668, 0.9583333333333333, 1.958333333333333],
                    'xaxis': 'x',
                    'y': df_multicategory_agg[selected_colnames[2]],
                    'yaxis': 'y',
                    'name': selected_colnames[2]
                }),
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [0.04166666666666662, 1.0416666666666665, 2.0416666666666665],
                    'xaxis': 'x',
                    'y': df_multicategory_agg[selected_colnames[3]],
                    'yaxis': 'y',
                    'name': selected_colnames[3]
                }),
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [0.125, 1.125, 2.125],
                    'xaxis': 'x',
                    'y': df_multicategory_agg[selected_colnames[4]],
                    'yaxis': 'y',
                    'name': selected_colnames[4]
                }),
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [0.20833333333333331, 1.208333333333333, 2.208333333333334],
                    'xaxis': 'x',
                    'y': df_multicategory_agg[selected_colnames[5]],
                    'yaxis': 'y',
                    'name': selected_colnames[5]
                })
            ],
            'layout': {
                'title': 'Distribution of Major Message Categories by Genre',
                'autosize': False,
                'bargap': 0.0,
                'height': 480,
                'hovermode': 'closest',
                'margin': {'b': 52, 'l': 80, 'pad': 0, 'r': 63, 't': 57},
                'showlegend': True,
                'template': '...',
                'width': 640,
                'xaxis': {'anchor': 'y',
                        'domain': [0.0, 1.0],
                        'dtick': 1,
                        'mirror': 'ticks',
                        'range': [-0.5, 2.5],
                        'showgrid': True,
                        'showline': True,
                        'side': 'bottom',
                        'tickfont': {'size': 10.0},
                        'ticks': '',
                        'tickmode': 'array',
                        'ticktext': df_gc_x,
                        'tickvals': list(range(0,len(df_gc_x))),
                        'title': {'font': {'color': '#000000', 'size': 10.0}, 'text': 'Genre'},
                        'type': 'linear',
                        'zeroline': False},
                'yaxis': {'anchor': 'x',
                        'domain': [0.0, 1.0],
                        'mirror': 'ticks',
                        'nticks': 8,
                        'range': [0.0, 6103.65],
                        'showgrid': True,
                        'showline': True,
                        'title': 'Count',
                        'side': 'left',
                        'tickfont': {'size': 10.0},
                        'ticks': 'inside',
                        'type': 'linear',
                        'zeroline': False}
            }
        }

    # encode plotly graphs in JSON
    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def create_multicategory_plot_for_mn():
    # create a graph object that represents the visualization
    graph = \
        {
            'data': [
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [-0.2, 0.8, 1.8],
                    'xaxis': 'x',
                    'y': mn_agg[selected_msglabels[0]],
                    'yaxis': 'y',
                    'name': selected_msglabels[0]
                }),
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [-0.09999999999999996, 0.8999999999999999, 1.9],
                    'xaxis': 'x',
                    'y': mn_agg[selected_msglabels[1]],
                    'yaxis': 'y',
                    'name': selected_msglabels[1]
                }),
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [0.0, 1.0, 2.0],
                    'xaxis': 'x',
                    'y': mn_agg[selected_msglabels[2]],
                    'yaxis': 'y',
                    'name': selected_msglabels[2]
                }),
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [0.10000000000000003, 1.1, 2.1000000000000005],
                    'xaxis': 'x',
                    'y': mn_agg[selected_msglabels[3]],
                    'yaxis': 'y',
                    'name': selected_msglabels[3]
                }),
                Bar({
                    'opacity': 1,
                    'orientation': 'v',
                    'x': [0.2, 1.2, 2.2],
                    'xaxis': 'x',
                    'y': mn_agg[selected_msglabels[4]],
                    'yaxis': 'y',
                    'name': selected_msglabels[4]
                })
            ],
            'layout': {
                'title': 'Message Request with Label Money for Various Genre',
                'autosize': False,
                'bargap': 0.0,
                'height': 480,
                'hovermode': 'closest',
                'margin': {'b': 52, 'l': 80, 'pad': 0, 'r': 63, 't': 57},
                'showlegend': True,
                'template': '...',
                'width': 640,
                'xaxis': {'anchor': 'y',
                        'domain': [0.0, 1.0],
                        'dtick': 1,
                        'mirror': 'ticks',
                        'range': [-0.5, 2.5],
                        'showgrid': True,
                        'showline': True,
                        'side': 'bottom',
                        'tickfont': {'size': 10.0},
                        'ticks': '',
                        'tickmode': 'array',
                        'ticktext': mn_agg_x,
                        'tickvals': list(range(0,len(mn_agg_x))),
                        'title': {'font': {'color': '#000000', 'size': 10.0}, 'text': 'Genre'},
                        'type': 'linear',
                        'zeroline': False},
                'yaxis': {'anchor': 'x',
                        'domain': [0.0, 1.0],
                        'mirror': 'ticks',
                        'nticks': 10,
                        'range': [0.0, 200.55],
                        'showgrid': True,
                        'showline': True,
                        'title': 'Count',
                        'side': 'left',
                        'tickfont': {'size': 10.0},
                        'ticks': 'inside',
                        'type': 'linear',
                        'zeroline': False}
            }
        }

    # encode plotly graphs in JSON
    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
# plots end

if __name__ == '__main__':
    main()