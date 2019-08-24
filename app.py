import base64
import logging
import os

from urllib.parse import quote

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash_table import DataTable
from dash_table.FormatTemplate import Format
from plotly.subplots import make_subplots
import pandas as pd
import advertools as adv
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

logging.basicConfig(level=logging.INFO)

app_key = os.environ['APP_KEY']
app_secret = os.environ['APP_SECRET']
oauth_token = os.environ['OAUTH_TOKEN']
oauth_token_secret = os.environ['OAUTH_TOKEN_SECRET']

auth_params = {
    'app_key': app_key,
    'app_secret': app_secret,
    'oauth_token': oauth_token,
    'oauth_token_secret': oauth_token_secret,
}

adv.twitter.set_auth_params(**auth_params)

img_base64 = base64.b64encode(open('./logo.png', 'rb').read()).decode('ascii')

# twtr_lang_df = adv.twitter.get_supported_languages()
# twtr_lang_df.to_csv('twitter_lang_df.csv', index=False)
twtr_lang_df = pd.read_csv('twitter_lang_df.csv')
twtr_lang_df = twtr_lang_df.sort_values('name')
lang_options = [{'label': loc_name, 'value': code}
                for loc_name, code in
                zip(twtr_lang_df['name'] + ' | ' + twtr_lang_df['local_name'],
                    twtr_lang_df['code'])]

exclude_columns = ['tweet_entities', 'tweet_geo', 'user_entities',
                   'tweet_coordinates', 'tweet_metadata',
                   'tweet_extended_entities', 'tweet_contributors',
                   'tweet_display_text_range', 'tweet_user', 'tweet_place',
                   'tweet_truncated']


regex_dict = {'Emoji': adv.emoji.EMOJI_RAW,
              'Mentions': adv.regex.MENTION_RAW,
              'Hashtags': adv.regex.HASHTAG_RAW,}

phrase_len_dict = {'Words': 1,
                   '2-word Phrases': 2,
                   '3-word Phrases': 3}

def get_str_dtype(df, col):
    """Return dtype of col in df"""
    dtypes = ['datetime', 'bool', 'int', 'float',
              'object', 'category']
    for d in dtypes:
        if d in str(df.dtypes.loc[col]).lower():
            return d


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

server = app.server

app.layout = html.Div([
    dcc.Loading(dcc.Store(id='twitter_df', storage_type='memory')),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.A([
                html.Img(src='data:image/png;base64,' + img_base64,
                         width=200, style={'display': 'inline-block'}),
            ], href='https://github.com/eliasdabbas/advertools'),
            html.Br(),
            # html.A(children=['online marketing', html.Br(),
            #                  'productivity & analysis'],
            #        style={'horizontal-align': 'center'},
            #        href='https://github.com/eliasdabbas/advertools'),
        ], lg=2, xs=11, style={'textAlign': 'center'}),
        dbc.Col([
            html.Br(),
            html.H1('Twitter Search: Create & Analyze Your Own Dataset',
                    style={'textAlign': 'center'})
        ], lg=9, xs=11),
    ], style={'margin-left': '1%'}),
    html.Br(),
    dbc.Row([
        dbc.Col(lg=2, xs=10),
        dbc.Col([
           dcc.Dropdown(id='search_type',
                        placeholder='Search Type',
                        options=[{'label': c, 'value': c}
                                 for c in ['Search Tweets',
                                           'Search Users',
                                           'Get User Timeline']])
        ], lg=2, xs=10),
        dbc.Col([
            dbc.Input(id='twitter_search',
                      placeholder='Search query'),
        ], lg=2, xs=10),
        dbc.Col([
            dbc.Input(id='twitter_search_count',
                      placeholder='Number of results', type='number'),

        ], lg=2, xs=10),
        dbc.Col([
            dcc.Dropdown(id='twitter_search_lang', placeholder='Language',
                         options=lang_options,
                         style={'position': 'relative', 'zIndex': 15}
                         ),
        ], lg=2, xs=10),
        dbc.Col([
            dbc.Button(id='search_button', children='Submit', outline=True),
        ], lg=2, xs=10),
    ]),
    html.Hr(),
    dbc.Container([
        dbc.Col(lg=2, xs=10),
        dbc.Tabs([
            dbc.Tab([
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Text field:'),
                        dcc.Dropdown(id='text_columns',
                                     placeholder='Text Column',
                                     value='tweet_full_text'),
                    ], lg=3, xs=9),
                    dbc.Col([
                        dbc.Label('Weighted by:'),
                        dcc.Dropdown(id='numeric_columns',
                                     placeholder='Numeric Column',
                                     value='user_followers_count'),
                    ], lg=3, xs=9),
                    dbc.Col([
                        dbc.Label('Elements to count:'),
                        dcc.Dropdown(id='regex_options',
                                     options=[{'label': x, 'value': x}
                                              for x in ['Words', 'Emoji',
                                                        'Mentions', 'Hashtags',
                                                        '2-word Phrases',
                                                        '3-word Phrases']],
                                     value='Words'),
                    ], lg=3, xs=9),
                ]),
                html.Br(),
                html.H2(id='wtd_freq_chart_title',
                        style={'textAlign': 'center'}),
                dcc.Loading([
                    dcc.Graph(id='wtd_freq_chart',
                              config={'displayModeBar': False},
                              figure={'layout': go.Layout(plot_bgcolor='#eeeeee',
                                                          paper_bgcolor='#eeeeee')
                                      }),
                ]),
            ], label='Text Analysis', id='text_analysis_tab'),
            dbc.Tab([
                html.H3(id='user_overview', style={'textAlign': 'center'}),
                dcc.Loading([
                    dcc.Graph(id='user_analysis_chart',
                              config={'displayModeBar': False},
                              figure={'layout': go.Layout(plot_bgcolor='#eeeeee',
                                                          paper_bgcolor='#eeeeee')
                                      })
                ]),
            ], tab_id='user_analysis_tab', label='User Analysis'),
            dbc.Tab([
                html.Br(),
                html.Iframe(src="//www.slideshare.net/slideshow/embed_code/key/mgjFlVE6XvyXi4",
                            width=595, height=485,
                            style={'margin-left': '30%'})

            ], label='How to Use', tab_style={'fontWeight': 'bold'})
        ], id='tabs'),
    ]),
    html.Hr(), html.Br(),
    dbc.Row([
        dbc.Col(lg=3, xs=11),
        dbc.Col(id='container_col_select',
                children=dcc.Dropdown(id='col_select',
                                      placeholder='Filter by:'),
                lg=2, xs=11),
        dbc.Col([
            dbc.Col([
                dbc.Container(children=dcc.RangeSlider(id='num_filter',
                                                       updatemode='mouseup')),
                dbc.Container(children=html.Div(id='rng_slider_vals'),
                              style={'text-align': 'center'}),
            ], lg=0, xs=0, id='container_num_filter',
                style={'display': 'none'}),
            dbc.Col(id='container_str_filter',
                    style={'display': 'none'},
                    children=dcc.Input(id='str_filter'), lg=0, xs=0),
            dbc.Col(id='container_bool_filter',
                    style={'display': 'none'},
                    children=dcc.Dropdown(id='bool_filter',
                                          options=[{'label': 'True',
                                                    'value': 1},
                                                   {'label': 'False',
                                                    'value': 0}]),
                    lg=0, xs=0,),
            dbc.Col(id='container_cat_filter',
                    style={'display': 'none'},
                    children=dcc.Dropdown(id='cat_filter', multi=True),
                    lg=0, xs=0),
            dbc.Col([
                dcc.DatePickerRange(id='date_filter'),
            ], id='container_date_filter', style={'display': 'none'},
                lg=0, xs=0),
        ], style={'width': '20%', 'display': 'inline-block'}),
        dbc.Col(id='row_summary', lg=2, xs=11),
        dbc.Col(html.A('Download Table', id='download_link',
                       download="rawdata.csv", href="", target="_blank",
                       n_clicks=0), lg=2, xs=11),
    ], style={'position': 'relative', 'zIndex': 5}),
    dbc.Row([
        dbc.Col([
            dbc.Container(html.Label('Add/remove columns:')),
            dcc.Dropdown(id='output_table_col_select', multi=True,
                         # value=['tweet_created_at', 'user_screen_name',
                         #        'user_followers_count', 'tweet_full_text',
                         #        'tweet_retweet_count']
                         ),
        ], lg=2, xs=11, style={'margin-left': '1%'}),
        dbc.Col([
            html.Br(),
            dcc.Loading(
                DataTable(id='table', sort_action='native',
                          # n_fixed_rows=1,
                          # filtering=False,
                          virtualization=True,
                          fixed_rows={'headers': True},
                          # style_cell_conditional=[{
                          #     'if': {'row_index': 'odd'},
                          #     'backgroundColor': '#eeeeee'}],
                          style_cell={'width': '200px',
                                      'font-family': 'Source Sans Pro'}),
            ),
        ], lg=9, xs=11, style={'position': 'relative', 'zIndex': 1,
                               'margin-left': '1%'}),
    ] + [html.Br() for x in range(30)]),
], style={'backgroundColor': '#eeeeee'})




@app.callback([Output('text_columns', 'options'),
               Output('text_columns', 'value'),
               Output('output_table_col_select', 'value')],
              [Input('search_button', 'n_clicks')],
              [State('twitter_search', 'value'),
               State('search_type', 'value')])
def set_text_columns_dropdown_options(n_clicks, query, search_type):
    search_tweet_cols = ['tweet_created_at', 'user_screen_name',
                         'user_followers_count', 'tweet_full_text',
                          'tweet_retweet_count']

    if search_type == 'Search Users':
        return ([{'label': 'User Description','value': 'user_description'}],
                'user_description',
                ['user_created_at', 'user_screen_name',
                 'user_description', 'user_followers_count'])
    return ([{'label': 'Tweet Text', 'value': 'tweet_full_text'},
            {'label': 'User Description','value': 'user_description'}],
            'tweet_full_text',
            search_tweet_cols if search_type == 'Search Tweets' else
            ['tweet_created_at', 'tweet_full_text',
             'tweet_retweet_count', 'tweet_favourite_count'])


@app.callback(Output('wtd_freq_chart_title', 'children'),
              [Input('regex_options', 'value'),
               Input('twitter_df', 'data')])
def display_wtd_freq_chart_title(regex, df):
    if regex is None or df is None:
        raise PreventUpdate
    return 'Most Frequently Used ' + regex + ' (' + str(len(df)) +  ' Results)'


@app.callback(Output('user_overview', 'children'),
              [Input('twitter_df', 'data'),
               Input('search_type', 'value')])
def display_user_overview(df, search_type):
    if df is None:
        raise PreventUpdate
    df = pd.DataFrame(df)
    n_tweets = len(df)
    n_users = df['user_screen_name'].nunique()
    num_tweets = '' if search_type == 'Search Users' else \
        'Number of tweets: ' + str(n_tweets) + ' | '
    return num_tweets + 'Number of Users: ' + str(n_users)


@app.callback(Output('wtd_freq_chart', 'figure'),
              [Input('twitter_df', 'data'),
               Input('text_columns', 'value'),
               Input('numeric_columns', 'value'),
               Input('regex_options', 'value'),
               Input('search_type', 'value')])
def plot_wtd_frequency(df, text_col, num_col, regex, search_type):
    if (df is None) or (text_col is None) or (num_col is None) or \
            (search_type is None):
        raise PreventUpdate
    df = pd.DataFrame(df)
    wtd_freq_df = adv.word_frequency(df[text_col], df[num_col],
                                     regex=regex_dict.get(regex),
                                     phrase_len=phrase_len_dict.get(regex)
                                     or 1)[:20]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Weighted Frequency',
                                        'Absolute Frequency'])
    fig.append_trace(go.Bar(x=wtd_freq_df['wtd_freq'][::-1],
                            y=wtd_freq_df['word'][::-1],
                            name='Weighted Freq.',
                            orientation='h'), 1, 1)
    wtd_freq_df = wtd_freq_df.sort_values('abs_freq', ascending=False)
    fig.append_trace(go.Bar(x=wtd_freq_df ['abs_freq'][::-1],
                            y=wtd_freq_df['word'][::-1],
                            name='Abs. Freq.',
                            orientation='h'), 1, 2)

    fig['layout'].update(height=600,
                         plot_bgcolor='#eeeeee',
                         paper_bgcolor='#eeeeee',
                         showlegend=False,
                         yaxis={'title': 'Top Words: ' +
                                text_col.replace('_', ' ').title()})
    fig['layout']['annotations'] += ({'x': 0.5, 'y': -0.16,
                                      'xref': 'paper', 'showarrow': False,
                                      'font': {'size': 16},
                                      'yref': 'paper',
                                      'text': num_col.replace('_', ' ').title()
                                      },)
    fig['layout']['xaxis']['domain'] = [0.1, 0.45]
    fig['layout']['xaxis2']['domain'] = [0.65, 1.0]
    return fig


@app.callback(Output('user_analysis_chart', 'figure'),
              [Input('twitter_df', 'data'),
               Input('search_type', 'value')])
def plot_user_analysis_chart(df, search_type):
    if (df is None) or (search_type is None):
        raise PreventUpdate
    subplot_titles = ['Followers Count', 'Statuses Count',
                      'Friends Count', 'Favourites Count',
                      'Verified', 'Tweet Source',
                      'Lang', 'User Created At']
    df = pd.DataFrame(df).drop_duplicates('user_screen_name')
    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=subplot_titles)
    for i, col in enumerate(subplot_titles[:4], start=1):
        col = ('user_' + col).replace(' ', '_').lower()
        fig.append_trace(go.Histogram(x=df[col], nbinsx=30,name='Users'),
                         1, i)
    for i, col in enumerate(subplot_titles[4:7], start=5):
        if (i == 6) and (search_type == 'Search Users'):
            continue
        if col == 'Tweet Source':
            col = 'tweet_source'
        else:
            col = ('user_' + col).replace(' ', '_').lower()
        fig.append_trace(go.Bar(x=df[col].value_counts().index[:14], width=0.9,
                                y=df[col].value_counts().values[:14],
                                name='Users'), 2, i-4)
    fig.append_trace(go.Histogram(x=df['user_created_at'],name='Users',
                                  nbinsx=30, ), 2, 4)

    fig['layout'].update(height=600,
                         yaxis={'title': 'Number of Users' + (' ' * 50) + ' .'
                                },
                         width=1200,
                         plot_bgcolor='#eeeeee',
                         paper_bgcolor='#eeeeee',
                         showlegend=False)
    return fig


@app.callback(Output('numeric_columns', 'options'),
              [Input('twitter_df', 'data')])
def set_text_columns_ddown_options(df):
    if df is None:
        raise PreventUpdate
    df = pd.DataFrame(df)
    num_cols = [c for c in df.columns if 'count' in c]
    num_options = [{'label': c.replace('_', ' ').title(), 'value': c}
                   for c in num_cols]
    return num_options


@app.callback(Output('twitter_df', 'data'),
              [Input('search_button', 'n_clicks')],
              [State('search_type', 'value'),
               State('twitter_search', 'value'),
               State('twitter_search_count', 'value'),
               State('twitter_search_lang', 'value')])
def get_twitter_data_save_in_store(n_clicks, search_type, query, count, lang):
    if query is None:
        raise PreventUpdate
    if search_type == 'Search Tweets':
        df = adv.twitter.search(q=query + ' -filter:retweets',
                                count=count, lang=lang,
                                tweet_mode='extended')
    elif search_type == 'Search Users':
        resp_dfs = []
        for i, num in enumerate(adv.twitter._get_counts(count, default=20),
                                start=1):
            df = adv.twitter.search_users(q=query, count=num, page=i,
                                          include_entities=True)
            resp_dfs.append(df)
        df = pd.concat(resp_dfs)
        df.columns = ['user_' + c for c in df.columns]
    else:
        df = adv.twitter.get_user_timeline(screen_name=query,
                                           exclude_replies=False,
                                           include_rts=True,
                                           count=count,tweet_mode='extended')
    for exclude in exclude_columns:
        if exclude in df:
            del df[exclude]
    return df.to_dict('rows')


@app.callback([Output('col_select', 'options'),
               Output('output_table_col_select', 'options')],
              [Input('twitter_df', 'data')])
def set_columns_to_select(df):
    if df is None:
        raise PreventUpdate
    columns = [{'label': c.replace('_', ' ').title(), 'value': c}
               for c in df[0].keys()]
    return columns, columns


containers = ['container_num_filter', 'container_str_filter',
              'container_bool_filter', 'container_cat_filter',
              'container_date_filter']


@app.callback([Output(x, 'style')
               for x in containers],
              [Input('twitter_df', 'data'),
               Input('col_select', 'value')])
def dispaly_relevant_filter_container(df, col):
    if (col is None) or (df is None):
        raise PreventUpdate
    df = pd.DataFrame(df)
    for column in df:
        if 'created' in column:
            df[column] = pd.to_datetime(df[column])
        if ('lang' in column) or ('source' in column):
            df[column] = df[column].astype('category')

    dtypes = [['int', 'float'], ['object'], ['bool'],
              ['category'], ['datetime']]
    result = [{'display': 'none'} if get_str_dtype(df, col) not in d
              else {'display': 'inline-block'} for d in dtypes]
    return result


@app.callback([Output('num_filter', 'min'),
               Output('num_filter', 'max'),
               Output('num_filter', 'value')],
              [Input('twitter_df', 'data'),
               Input('col_select', 'value')])
def set_rng_slider_max_min_val(df, column):
    if column is None:
        raise PreventUpdate
    df = pd.DataFrame(df)
    if column and (get_str_dtype(df, column) in ['int', 'float']):
        minimum = df[column].min()
        maximum = df[column].max()
        return minimum, maximum, [minimum, maximum]
    else:
        return None, None, None


@app.callback(Output('rng_slider_vals', 'children'),
              [Input('num_filter', 'value')])
def show_rng_slider_max_min(numbers):
    if numbers is None:
        raise PreventUpdate
    return 'from: ' + ' to: '.join([str(numbers[0]), str(numbers[-1])])


@app.callback(Output('cat_filter', 'options'),
              [Input('twitter_df', 'data'),
               Input('col_select', 'value')])
def set_categorical_filter_options(df, column):
    if column is None:
        raise PreventUpdate
    df = pd.DataFrame(df)
    if any([x in column for x in ['lang', 'source']]):
        return [{'label': x, 'value': x} for x in list(df[column].unique())]
    return []


@app.callback(Output('table', 'columns'),
              [Input('output_table_col_select', 'value')])
def set_table_columns(columns):
    if columns is None:
        raise PreventUpdate
    return [{'id': c, 'name': c.replace('_', ' ').title(),
             'type': 'numeric' if 'ount' in c else 'text',
             'format': Format(group=',') if 'ount' in c else None}
            for c in columns]


@app.callback([Output('date_filter', 'start_date'),
               Output('date_filter', 'end_date'),
               Output('date_filter', 'min_date_allowed'),
               Output('date_filter', 'max_date_allowed')],
              [Input('twitter_df', 'data'),
               Input('col_select', 'value')])
def set_date_filter_params(df, col):
    if (col is None) or (df is None):
        raise PreventUpdate
    df = pd.DataFrame(df)
    if 'created' in col:
        start = df[col].min()
        end = df[col].max()
        return start, end, start, end
    return None, None, None, None


@app.callback(Output('table', 'data'),
              [Input('twitter_df', 'data'),
               Input('col_select', 'value'),
               Input('num_filter', 'value'),
               Input('cat_filter', 'value'),
               Input('str_filter', 'value'),
               Input('bool_filter', 'value'),
               Input('date_filter', 'start_date'),
               Input('date_filter', 'end_date')])
def filter_table(df, col, numbers, categories, string,
                 bool_filter, start_date, end_date):
    if df is None:
        raise PreventUpdate
    if all([param is None for param in
           [df, col, numbers, categories, string,
            bool_filter, start_date, end_date]]):
        raise PreventUpdate
    if (col is None) and any([param is not None for param in
                             [numbers, categories, string,
                              bool_filter, start_date, end_date]]):
        raise PreventUpdate
    df = pd.DataFrame(df)
    for column in df:
        if 'created' in column:
            df[column] = pd.to_datetime(df[column])
        if ('lang' in column) or ('source' in column):
            df[column] = df[column].astype('category')
    logging_dict = {k: v for k, v in locals().items()
                   if k not in ['df', 'column'] and v is not None}
    logging.info(msg=logging_dict)
    if numbers and (get_str_dtype(df, col) in ['int', 'float']):
        df = df[df[col].between(numbers[0], numbers[-1])]
        return df.to_dict('rows')
    elif categories and (get_str_dtype(df, col) == 'category'):
        df = df[df[col].isin(categories)]
        return df.to_dict('rows')
    elif string and get_str_dtype(df, col) == 'object':
        df = df[df[col].str.contains(string, case=False)]
        return df.to_dict('rows')
    elif (bool_filter is not None) and (get_str_dtype(df, col) == 'bool'):
        df = df[df[col] == bool_filter]
        return df.to_dict('rows')
    elif start_date and end_date and (get_str_dtype(df, col) == 'datetime'):
        df = df[df[col].between(start_date, end_date)]
        return df.to_dict('rows')
    else:
        return df.to_dict('rows')


@app.callback(Output('row_summary', 'children'),
              [Input('twitter_df', 'data'), Input('table', 'data')])
def show_rows_summary(orig_table, filtered_table):
    if filtered_table is None:
        raise PreventUpdate
    summary = f"{len(filtered_table)} out of {len(orig_table)} " \
              f"rows ({len(filtered_table)/len(orig_table):.1%})"
    return summary


@app.callback(Output('download_link', 'href'),
              [Input('twitter_df', 'data')])
def download_df(data_df):
    df = pd.DataFrame(data_df)
    csv_string = df.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + quote(csv_string)
    log_msg = (format(df.memory_usage().sum(), ',') +
               'bytes, shape:' + str(df.shape))
    logging.info(msg=log_msg)
    return csv_string


if __name__ == '__main__':
    app.run_server()
