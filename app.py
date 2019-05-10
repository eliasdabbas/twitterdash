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
import pandas as pd
import advertools as adv
from dash.exceptions import PreventUpdate

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
                   'tweet_display_text_range']


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
            html.Img(src='data:image/png;base64,' + img_base64,
                     width=200, style={'display': 'inline-block'}),
            html.Br(),
            html.A(children=['online marketing', html.Br(),
                             'productivity & analysis'],
                   style={'horizontal-align': 'center'},
                   href='https://github.com/eliasdabbas/advertools'),
        ], lg=2, xs=11, style={'textAlign': 'center'}),
        dbc.Col([
            html.H1('Twitter Search: Create Your Own Dataset',
                    style={'textAlign': 'center'})
        ], lg=9, xs=11),
    ], style={'margin-left': '1%'}),
    dbc.Row([
        dbc.Col(lg=3, xs=10),
        dbc.Col([
            dbc.Input(id='twitter_search',
                      placeholder='Search Twitter'),
        ], lg=2, xs=10),
        dbc.Col([
            dbc.Input(id='twitter_search_count',
                      placeholder='Number or tweets', type='number'),
        ], lg=2, xs=10),
        dbc.Col([
            dcc.Dropdown(id='twitter_search_lang', placeholder='Language',
                         options=lang_options,
                         style={'position': 'relative', 'zIndex': 15}
                         ),
        ], lg=2, xs=10),
        dbc.Col([
            html.Button(id='search_button', children='Submit'),
        ], lg=2, xs=10),
    ]),
    html.Hr(),
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
                         value=['tweet_created_at', 'user_screen_name',
                                'user_followers_count', 'tweet_full_text',
                                'tweet_retweet_count']),
        ], lg=2, xs=11, style={'margin-left': '1%'}),
        dbc.Col([
            html.Br(),
            dcc.Loading(
                DataTable(id='table', sorting=True,
                          n_fixed_rows=1,
                          virtualization=True,
                          style_cell_conditional=[{
                              'if': {'row_index': 'odd'},
                              'backgroundColor': '#eeeeee'}],
                          style_cell={'width': '200px'}),
            ),
        ], lg=9, xs=11, style={'position': 'relative', 'zIndex': 1,
                               'margin-left': '1%'}),
    ] + [html.Br() for x in range(30)]),
], style={'backgroundColor': '#eeeeee'})


@app.callback(Output('twitter_df', 'data'),
              [Input('search_button', 'n_clicks')],
              [State('twitter_search', 'value'),
               State('twitter_search_count', 'value'),
               State('twitter_search_lang', 'value')])
def get_twitter_data_save_in_store(n_clicks, query, count, lang):
    if query is None:
        raise PreventUpdate
    df = adv.twitter.search(q=query + ' -filter:retweets',
                            count=count, lang=lang,
                            tweet_mode='extended')
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
        print(list(df[column].unique()))
        return [{'label': x, 'value': x} for x in list(df[column].unique())]
    return []


@app.callback(Output('table', 'columns'),
              [Input('output_table_col_select', 'value')])
def set_table_columns(columns):
    if columns is None:
        raise PreventUpdate
    return [{'id': c, 'name': c.replace('_', ' ').title()}
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
    # if get_str_dtype(df, col) == 'datetime':
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
    loggin_dict = {k: v for k, v in locals().items()
                   if k not in ['df', 'column'] and v is not None}
    logging.info(msg=loggin_dict)
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
