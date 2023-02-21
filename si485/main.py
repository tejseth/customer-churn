import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Customer Churn'
server = app.server

mod = Booster({'nthread': 8})
mod.load_model('xgb_churn')

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H3(children='Customer Churn - Team 10'),
    html.Div([
        dbc.Label('Agreement Count', ),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], '1', id='agreement_count'),
        
        dbc.Label('Total Labor Time', ),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'], '1', id='total_labor_time'),
        
        dbc.Label('Main Equipment Count'),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], '1', id='main_equipment_count'),
        
        dbc.Label('Previous Month Equipment Count'),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], '1', id='prev_main_equipment_count'),
        
        dbc.Label('First Month Equipment Count'),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], '1', id='first_main_equipment_count'),
        
        dbc.Label('Contract Revenue Discounted'),
        dcc.Slider(0, 10000, 500, value=36, marks= None, id='contract_revenue_discounted', tooltip={"placement": "bottom", "always_visible": True}),
        
        dbc.Label('Parts Cost'),
        dcc.Slider(0, 1000, 100, value=0, marks=None, id='parts_cost', tooltip={"placement": "bottom", "always_visible": True}),
        
        dbc.Label('Total Service Calls'),
        dcc.Slider(0, 200, 10, value=0, marks=None, id='total_service_calls', tooltip={"placement": "bottom", "always_visible": True}),
        
        dbc.Label('Tenure (Months)'),
        dcc.Slider(0, 150, 10, value=11, marks=None, id='tenure', tooltip={"placement": "bottom", "always_visible": True}),
        
        dbc.Label('Previous Month Contract Revenue Discounted'),
        dcc.Slider(0, 10000, 500, value=0, marks=None, id='prev_contract_revenue_discounted', tooltip={"placement": "bottom", "always_visible": True}),
        
        dbc.Label('First Month Contract Revenue Discounted'),
        dcc.Slider(0, 10000, 500, value=0, marks=None, id='first_contract_revenue_discounted', tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'width': '33.6%', 'margin-top': '3.5%', 'margin-left': '1.5%', 'margin-bottom': '3%', 'text-align': 'left', 'padding':'1%','background-color': '#f5f5f5', 'border': '1px solid #e3e3e3', 'border-radius': '4px'}),

        dbc.Row([
            html.Br(),
            dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")]),
            html.Br()
        ]),
        dbc.Row([
            html.Br(),
            dbc.Row([html.Div(id='prediction output-1')]),
            html.Br()
        ])
])

@app.callback(
    # Output('bar-1', component_property= 'figure'),
    Output('prediction output-1', 'children'),
    Input('submit-val', 'n_clicks'),
    State('year', 'value'),
    State('agreement_count', 'value'),
    State('total_labor_time', 'value'),
    State('main_equipment_count', 'value'),
    State('prev_main_equipment_count', 'value'),
    State('first_main_equipment_count', 'value'),
    State('contract_revenue_discounted', 'value'),
    State('parts_cost', 'value'),
    State('total_service_calls', 'value'),
    State('tenure', 'value'),
    State('prev_contract_revenue_discounted', 'value'),
    State('first_contract_revenue_discounted', 'value'))

def update_output(n_clicks, year, agreement_count, total_labor_time, 
    main_equipment_count, prev_main_equipment_count, first_main_equipment_count,
    contract_revenue_discounted, parts_cost, total_service_calls, tenure,
    prev_contract_revenue_discounted, first_contract_revenue_discounted):

        crd_change = float(contract_revenue_discounted) - float(prev_contract_revenue_discounted)
        equipment_change = float(main_equipment_count) - float(prev_main_equipment_count)
        crd_change_first = float(contract_revenue_discounted) - float(first_contract_revenue_discounted)
        equipment_change_first = float(main_equipment_count) - float(first_main_equipment_count)


        x = np.array([[float(agreement_count), float(contract_revenue_discounted), float(main_equipment_count),
        float(parts_cost), float(total_labor_time), float(total_service_calls), float(tenure), float(crd_change),
        float(equipment_change), float(crd_change_first), float(equipment_change_first)]])

        dtrain = DMatrix(x)
        prediction = round(100*(mod.predict(dtrain)[0]), 1)

        dtrain = DMatrix(x)
        prediction = round(100*(mod.predict(dtrain)[0]), 1)

        return f'{prediction}% chance they churn'

if __name__ == '__main__':
    app.run_server(debug=True)

