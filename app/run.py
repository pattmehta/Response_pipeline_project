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
    if c.name in df_category_topten.index.values and c.name not in ['related','direct_report','other_aid','request']:
        if c.name not in selected_colnames: selected_colnames.append(c.name)
        return c.sum()
    else: return False
df_multicategory_agg = df_multicategory.aggregate(categories_agg)
df_gc_x = list(df_multicategory_agg.index)
# data prep for plots end

# load model
# model = joblib.load(model_path)
model = None
with open(model_path, 'rb') as pkl_file:
    model = pickle.load(pkl_file)
if not model:
    print('could not load the model')
    exit()


# routes start
@app.route('/viz_genre')
def viz_genre():
    ids = [1]
    graphJSON = create_genre_plot()
    return render_template('viz_genre.html', ids=ids, graph=graphJSON)

@app.route('/viz_category')
def viz_category():
    ids = [1]
    graphJSON = create_category_plot()
    return render_template('viz_category.html', ids=ids, graph=graphJSON)

@app.route('/viz_multicategory')
def viz_multicategory():
    ids = [1]
    graphJSON = create_multicategory_plot()
    return render_template('viz_multicategory.html', ids=ids, graph=graphJSON)
# routes end

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
    
    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def create_category_plot():
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
    
    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def create_multicategory_plot():
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

    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
# plots end

if __name__ == '__main__':
    main()