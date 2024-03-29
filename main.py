import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix

external_stylesheets = ['/design.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Customer Churn'
server = app.server

mod = Booster({'nthread': 8})
mod.load_model('xgb_churn')
mod2 = Booster({'nthread': 8})
mod2.load_model('xgb_months')

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        html.H3(children='Service Express Churn Prediction Model'),
        dbc.Col(html.Div([
            
        dbc.Label('Agreement Count'),
        dcc.Dropdown(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], '1', id='agreement_count'),
        dbc.Label('Total Labor Time'),
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
        
        dbc.Row([
            dbc.Col(html.Br()),
            dbc.Col(dbc.Button('Predict', id='submit-val', n_clicks=0, color="primary",style={'font-size': '16px', 'width': '50%', 'background-color': 'white', 'color': 'black', 'border-color': 'grey', 'margin-top': '10%', 'margin-left': '-50%'})),
        ], style={'margin-top': '-2%',}),

        ], style={'width': '75%', 'height': '100%','margin-top': '3.5%', 'margin-left': '1.5%', 'margin-bottom': '3%', 'text-align': 'left', 'padding':'1%','background-color': '#f5f5f5', 'border': '1px solid #e3e3e3', 'border-radius': '4px'})),
        
        dbc.Col(html.Div([
            html.Br(),
            dbc.Row([html.Div(id='prediction output-1')]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Img(src='/assets/UMSI.png', style={'width': '50%', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': '48%'})),
                dbc.Col(html.Img(src='/assets/SE.png', style={'width': '50%', 'display': 'block', 'margin-right': 'auto',  'margin-top': '59%'}))
            ], justify='center')
        ], style={'margin-top': '15%', 'font-weight': 'bold', 'font-size': '25px'}))

    ], style={'margin-top': '2%'}),
], fluid=True)


@app.callback(
    # Output('bar-1', component_property= 'figure'),
    Output('prediction output-1', 'children'),
    Input('submit-val', 'n_clicks'),
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

def update_output(n_clicks, agreement_count, total_labor_time, 
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
        prediction2 = mod2.predict(dtrain)[0]
        prediction2 = round(prediction2, 1)
        return html.Div([
            html.Div([
                html.Span(f"{prediction}% ", style={'font-size': '4rem', 'margin-bottom': '.75rem', 'color': 'red' if prediction > 10 else 'orange' if prediction > 5 else 'green'}),
                html.Span("chance they churn", style={'font-size': '4rem', 'margin-bottom': '.75rem', 'color': 'black'})
            ]),
            html.Div(f"{prediction2:.1f} months until they churn", style={'font-size': '4rem', 'margin-top': '.75em'})
        ])

if __name__ == '__main__':
    app.run_server(debug=True)

