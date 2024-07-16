import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import os
import requests

# URL del modelo en GitHub
model_url = 'https://github.com/JorgeHdzRiv/TitanicProjectCienciaDatos/raw/main/models/best_rf_model.joblib'
model_path = 'model/best_rf_model.joblib'

# Crear la carpeta 'modelo' si no existe
os.makedirs('model', exist_ok=True)

# Descargar el modelo si no existe localmente
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, 'wb') as file:
        file.write(response.content)

# Cargar el modelo
model = joblib.load(model_path)

# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout de la aplicación
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Aplicación de Predicción de Supervivencia del Titanic", className="text-center mt-4"))),
    dbc.Row([
        dbc.Col([
            html.Label('Clase del Pasajero (1 = Primera, 2 = Segunda, 3 = Tercera)'),
            dcc.Dropdown(
                id='pclass',
                options=[{'label': str(i), 'value': i} for i in range(1, 4)],
                value=1
            ),
            html.Label('Sexo (0 = Hombre, 1 = Mujer)'),
            dcc.Dropdown(
                id='sex',
                options=[{'label': 'Hombre', 'value': 0}, {'label': 'Mujer', 'value': 1}],
                value=0
            ),
            html.Label('Edad'),
            dcc.Input(id='age', type='number', value=30),
            html.Label('Tarifa'),
            dcc.Input(id='fare', type='number', value=32.204),
            html.Label('Embarcado (0 = Cherbourg, 1 = Queenstown, 2 = Southampton)'),
            dcc.Dropdown(
                id='embarked',
                options=[
                    {'label': 'Cherbourg', 'value': 0},
                    {'label': 'Queenstown', 'value': 1},
                    {'label': 'Southampton', 'value': 2}
                ],
                value=2
            ),
            html.Label('Tamaño de la Familia'),
            dcc.Input(id='family_size', type='number', value=1),
            html.Label('Está Solo (0 = No, 1 = Sí)'),
            dcc.Dropdown(
                id='is_alone',
                options=[{'label': 'No', 'value': 0}, {'label': 'Sí', 'value': 1}],
                value=1
            ),
            dbc.Button("Predecir Supervivencia", id='predict-button', color='primary', className='mt-3')
        ], width=4),
        dbc.Col([
            html.H3("Resultado de la Predicción"),
            html.Div(id='prediction-output', className='mt-4')
        ])
    ])
], fluid=True)

# Callback para realizar la predicción
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('pclass', 'value'),
    State('sex', 'value'),
    State('age', 'value'),
    State('fare', 'value'),
    State('embarked', 'value'),
    State('family_size', 'value'),
    State('is_alone', 'value')
)
def predict_survival(n_clicks, pclass, sex, age, fare, embarked, family_size, is_alone):
    if n_clicks is None:
        return ""

    # Crear un DataFrame con los valores de entrada
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'Fare': [fare],
        'Embarked_C': [1 if embarked == 0 else 0],
        'Embarked_Q': [1 if embarked == 1 else 0],
        'Embarked_S': [1 if embarked == 2 else 0],
        'Family_Size': [family_size],
        'IsAlone': [is_alone]
    })

    # Realizar la predicción
    prediction = model.predict(input_data)[0]

    # Devolver el resultado de la predicción
    if prediction == 1:
        return html.Div("El pasajero habría sobrevivido.", style={'color': 'green', 'fontSize': 24, 'fontWeight': 'bold'})
    else:
        return html.Div("El pasajero no habría sobrevivido.", style={'color': 'red', 'fontSize': 24, 'fontWeight': 'bold'})

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)