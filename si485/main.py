import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
from xgboost import XGBClassifier, Booster

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Customer Churn'
server = app.server

mod = Booster({'nthread': 8})
mod.load_model('xgb_churn')

app.layout = html.Div([
    dbc.Row([html.H3(children='Customer Churn - Team 10')]),
    dbc.Row([
        dbc.Col(html.Label(children = 'Agreement Count'), width={"order": "first"}),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], '1', id = 'agreement_count', style={"width": "75%", 'display': 'inline-block'}),
        html.Br(),
        dbc.Col(html.Label(children = 'Total Labor Time'), width={"order": "first"}),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'], '1', id = 'total_labor_time', style={"width": "75%", 'display': 'inline-block'}),
        html.Br(),
        dbc.Col(html.Label(children = 'Year'), width={"order": "first"}),
        dcc.Dropdown(['2018', '2019', '2020', '2021', '2022'], '2022', id = 'year', style={"width": "75%", 'display': 'inline-block'}),
        html.Br()
    ], style={'columnCount': 3}),
        dbc.Row([
        dbc.Col(html.Label(children = 'Main Equipment Count'), width={"order": "first"}),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], '1', id = 'main_equipment_count', style={"width": "75%", 'display': 'inline-block'}),
        html.Br(),
                dbc.Col(html.Label(children = 'Previous Month Equipment Count'), width={"order": "first"}),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], '1', id = 'prev_main_equipment_count', style={"width": "75%", 'display': 'inline-block'}),
        html.Br(),
                dbc.Col(html.Label(children = 'First Month Equipment Count'), width={"order": "first"}),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], '1', id = 'first_main_equipment_count', style={"width": "75%", 'display': 'inline-block'}),
        html.Br(),
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children='Contract Revenue Discounted'), width={"order": "first"}),
        dbc.Col(dcc.Slider(min=0, max=10000, step = 500, value = 2500, id='contract_revenue_discounted')),
        html.Br(),
        dbc.Col(html.Label(children='Parts Cost'), width={"order": "first"}),
        dbc.Col(dcc.Slider(min=0, max=1000, step = 100, value = 100, id='parts_cost')),
        html.Br(),
        dbc.Col(html.Label(children='Total Service Calls'), width={"order": "first"}),
        dbc.Col(dcc.Slider(min=0, max=200, step = 10, value = 10, id='total_service_calls')),
        html.Br(),
        dbc.Col(html.Label(children='Tenure (Months)'), width={"order": "first"}),
        dbc.Col(dcc.Slider(min=0, max=150, step = 10, value = 10, id='tenure')),
        html.Br(),
        dbc.Col(html.Label(children='Previous Month Contract Revenue Discounted'), width={"order": "first"}),
        dbc.Col(dcc.Slider(min=0, max=10000, step = 500, value = 2000, id='prev_contract_revenue_discounted')),
        html.Br(),
        dbc.Col(html.Label(children='First Month Contract Revenue Discounted'), width={"order": "first"}),
        dbc.Col(dcc.Slider(min=0, max=10000, step = 500, value = 1500, id='first_contract_revenue_discounted')),
        html.Br()
    ], style={'columnCount': 2}),
], style = {'padding': '0px 0px 0px 25px', 'width': '90%'})

# app.layout = html.Div([
#     dbc.Row([html.H3(children='Chance of a Sack')]),
#     dbc.Row([
#         dbc.Col(html.Label(children='Down'), width={"order": "first"}),
#         dcc.Dropdown(['1', '2', '3', '4'], '2', id='down'),
#         dbc.Col(html.Label(children='Offensive Personnel (RB-TE)'), width={"order": "first"}),
#         dcc.Dropdown(['11', '12', '21', '13', '10', '22', '01', '20', '11*', '02', '12*'], '11', id='o_dropdown'),
#         dbc.Col(html.Label(children='Defensive Formation (DL-LB-DB)'), width={"order": "first"}),    
#         dcc.Dropdown(['4-2-5', '2-4-5', '3-3-5', '2-3-6', '4-3-4', '3-4-4', '4-1-6', '3-2-6', '1-4-6', '1-5-5'], '4-2-5', id='d-dropdown'),
#         dbc.Col(html.Label(children='Ball Spot'), width={"order": "first"}),
#         dcc.Dropdown(['Left Hash', 'Middle', 'Right Hash'], 'Middle', id='the_hash'),
#         dbc.Col(html.Label(children='Offense Formation'), width={"order": "first"}),
#         dcc.Dropdown(['Shotgun', 'I Formation', 'Jumbo', 'Pistol', 'Singleback', 'Wildcat'], 'Shotgun', id='offenseFormation')
#     ]),
#     dbc.Row([
#         dbc.Col(html.Label(children='Yards to Go'), width={"order": "first"}),
#         dbc.Col(dcc.Slider(min=1, max=15, step = 1, value = 5, id='yardsToGo')),
#         html.Br()
#     ]),
#     dbc.Row([
#         dbc.Col(html.Label(children='Yards From Endzone'), width={"order": "first"}),    
#         dbc.Col(dcc.Slider(min = 1, max = 99, step = 5, value = 20, id = "absoluteYardlineNumber")),
#         html.Br()
#     ]),
#     dbc.Row([
#         dbc.Col(html.Label(children='Defenders in the Box'), width={"order": "first"}),    
#         dbc.Col(dcc.Slider(min = 4, max = 7, step = 1, value = 6, id = "defendersInBox")),
#         html.Br()
#     ]),
#     dbc.Row([
#         dbc.Col(html.Label(children="Player 1's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
#         dcc.Input(id='rel_y', value=5.25, type='number', min = 0, max = 53.33),
#         dbc.Col(html.Label(children="Player 1's Depth From LOS (0 to 45)"), width={"order": "first"}),    
#         dcc.Input(id='rel_x', value=1.32, type='number', min = 0, max = 45),
#         dbc.Col(html.Label(children="Player 1's Official Position"), width={"order": "first"}),
#         dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DE', id='official_position'),
#         html.Br(),
#         dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")]),
#         html.Br(),
#         dbc.Row([html.Div(id='prediction output')])
#     ])
# ], style = {'padding': '0px 0px 0px 150px', 'width': '50%'})

# @app.callback(
#     Output('prediction output', 'children'),
#     Input('submit-val', 'n_clicks'),
#     State('yardsToGo', 'value'),
#     State('absoluteYardlineNumber', 'value'),
#     State('defendersInBox', 'value'),
#     State('o_dropdown', 'value'),
#     State('d_dropdown', 'value'),
#     State('rel_x', 'value'),
#     State('rel_y', 'value'),
#     State('the_hash', 'value'),
#     State('down', 'value'),
#     State('official_position', 'value'),
#     State('offenseFormation', 'value')
# )

# # make all single arrays
   
# def update_output(n_clicks, yardsToGo, absoluteYardlineNumber, defendersInBox, o_dropdown, 
#                   d_dropdown, rel_x, rel_y, the_hash, down, official_position, offenseFormation):
#     yards_to_go_array = np.array([float(yardsToGo)])
#     yardline_array = np.array([float(absoluteYardlineNumber)])
#     box_array = np.array([float(defendersInBox)])
    
#     if o_dropdown == '12*':
#         o_pers_array = np.array([1, 2, 1])
#     elif o_dropdown == '12':
#         o_pers_array = np.array([1, 2, 2])
#     elif o_dropdown == '21':
#         o_pers_array = np.array([2, 1, 2])
#     elif o_dropdown == '13':
#         o_pers_array = np.array([1, 3, 1])
#     elif o_dropdown == '10':
#         o_pers_array = np.array([1, 3, 1])
#     elif o_dropdown == '22':
#         o_pers_array = np.array([2, 2, 1])
#     elif o_dropdown == '01':
#         o_pers_array = np.array([0, 1, 4])
#     elif o_dropdown == '20':
#         o_pers_array = np.array([2, 0, 3])
#     elif o_dropdown == '11*':
#         o_pers_array = np.array([1, 1, 2])
#     elif o_dropdown == '02':
#         o_pers_array = np.array([0, 2, 3])
#     else:
#         o_pers_array = np.array([1, 1, 3])
        
#     if d_dropdown == '1-5-5':
#         d_pers_array = np.array([4, 2, 5])
#     elif d_dropdown == '2-4-5':
#         d_pers_array = np.array([2, 4, 5])
#     elif d_dropdown == '3-3-5':
#         d_pers_array = np.array([3, 3, 5])
#     elif d_dropdown == '2-3-6':
#         d_pers_array = np.array([2, 3, 6])
#     elif d_dropdown == '4-3-4':
#         d_pers_array = np.array([4, 3, 4])
#     elif d_dropdown == '3-4-4':
#         d_pers_array = np.array([3, 4, 4])
#     elif d_dropdown == '4-1-6':
#         d_pers_array = np.array([4, 1, 6])
#     elif d_dropdown == '3-2-6':
#         d_pers_array = np.array([3, 2, 6])
#     elif d_dropdown == '1-4-6':
#         d_pers_array = np.array([1, 4, 6])
#     else:
#         d_pers_array = np.array([4, 2, 5])
        
#     rel_array = np.array([float(rel_x), float(rel_y)])
#     speed_array = np.array([0.23])
#     acc_array = np.array([0.49])
#     ball_x_array = yardline_array.copy()
    
#     if the_hash == 'Right':
#         ball_y_array = np.array([29.7])
#     elif the_hash == 'Left':
#         ball_y_array = np.array([23.6])
#     else:
#         ball_y_array = np.array([27.0])
        
#     oline_array = np.array([-3.1, 3.1, 6.2])
    
    
#     if float(down) == 4:
#         down_array = np.array([0, 0, 0, 1])
#     elif float(down) == 3:
#         down_array = np.array([0, 0, 1, 0])
#     elif float(down) == 2:
#         down_array = np.array([0, 1, 0, 0])
#     else:
#         down_array = np.array([1, 0, 0, 0])
        
#     if official_position == 'CB':
#         pos_array = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
#     elif official_position == 'DE':
#         pos_array = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
#     elif official_position == 'DT':
#         pos_array = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
#     elif official_position == 'FS':
#         pos_array = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
#     elif official_position == 'ILB':
#         pos_array = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
#     elif official_position == 'MLB':
#         pos_array = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
#     elif official_position == 'NT':
#         pos_array = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
#     elif official_position == 'OLB':
#         pos_array = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
#     elif official_position == 'SS':
#         pos_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        
#     if offenseFormation == 'Shotgun':
#         formation_array = np.array([1, 0, 0, 0, 0, 0])
#         qb_array = np.array([5, -5, 0])
#         dist_from_qb = np.sqrt(np.square(rel_x + 5)  + np.square(rel_y - 0))
#         dist_from_qb_array = np.array([float(dist_from_qb)])
#     elif offenseFormation == 'I Formation':
#         formation_array = np.array([0, 1, 0, 0, 0, 0])
#         qb_array = np.array([[1.5, -1.5, 0]])
#         dist_from_qb = np.sqrt(np.square(rel_x + 1.5)  + np.square(rel_y - 0))
#         dist_from_qb_array = np.array([float(dist_from_qb)])
#     elif offenseFormation == 'Jumbo':
#         formation_array = np.array([[0, 0, 1, 0, 0, 0]])
#         qb_array = np.array([1.5, -1.5, 0])
#         dist_from_qb = np.sqrt(np.square(rel_x + 1.5)  + np.square(rel_y - 0))
#         dist_from_qb_array = np.array([float(dist_from_qb)])
#     elif offenseFormation == 'Pistol':
#         formation_array = np.array([[0, 0, 0, 1, 0, 0]])
#         qb_array = np.array([[5, -5, 0]])
#         dist_from_qb = np.sqrt(np.square(rel_x + 5)  + np.square(rel_y - 0))
#         dist_from_qb_array = np.array([float(dist_from_qb)])
#     elif offenseFormation == 'Singleback':
#         formation_array = np.array([[0, 0, 0, 0, 1, 0]])
#         qb_array = np.array([[1.5, -1.5, 0]])
#         dist_from_qb = np.sqrt(np.square(rel_x + 1.5)  + np.square(rel_y - 0))
#         dist_from_qb_array = np.array([float(dist_from_qb)])
#     else:
#         formation_array = np.array([0, 0, 0, 0, 0, 1])
#         qb_array = np.array([5, -5, 0])
#         dist_from_qb = np.sqrt(np.square(rel_x + 5)  + np.square(rel_y - 0))
#         dist_from_qb_array = np.array([float(dist_from_qb)])

#     pre_x = np.concatenate((yards_to_go_array, yardline_array, box_array, o_pers_array, d_pers_array,
#                        rel_array, speed_array, acc_array, ball_x_array, ball_y_array, oline_array,
#                        qb_array, dist_from_qb_array, down_array, pos_array, formation_array))
    
#     x = np.reshape(pre_x, (-1, len(pre_x)))
#     prediction = mod.predict_proba(x)[1]
#     return prediction
#     # return f'The chance of a sack for Player 1: {prediction}.'

if __name__ == "__main__":
    app.run_server(debug=True)

