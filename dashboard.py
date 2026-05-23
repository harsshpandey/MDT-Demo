"""
MDT Wind Power Forecasting Dashboard
======================================
Interactive Dash web application for visualizing results.
Run: python dashboard.py
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input

# ─── Load Results ───
RESULTS_DIR = 'results'
PRED_DIR = os.path.join(RESULTS_DIR, 'predictions')

def load_all_data():
    """Load all prediction CSVs and metric CSVs."""
    models = {}
    for name in ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']:
        path = os.path.join(PRED_DIR, f'{name}_predictions.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            models[name] = df
    
    single_metrics = None
    m1_metrics = None
    m2_metrics = None
    
    p = os.path.join(RESULTS_DIR, 'single_dt_metrics.csv')
    if os.path.exists(p):
        single_metrics = pd.read_csv(p, index_col=0)
    
    p = os.path.join(RESULTS_DIR, 'method1_fusion_metrics.csv')
    if os.path.exists(p):
        m1_metrics = pd.read_csv(p, index_col=0)
    
    p = os.path.join(RESULTS_DIR, 'method2_fusion_metrics.csv')
    if os.path.exists(p):
        m2_metrics = pd.read_csv(p, index_col=0)
    
    return models, single_metrics, m1_metrics, m2_metrics

models_data, single_metrics, m1_metrics, m2_metrics = load_all_data()

COLORS = {
    'LSTM': '#FF6B6B', 'GRU': '#4ECDC4',
    'LSTMCNN': '#45B7D1', 'GRUCNN': '#96CEB4',
    'actual': '#FFEAA7',
}

# ─── App ───
app = Dash(__name__)
app.title = "MDT Wind Power Dashboard"

app.layout = html.Div(style={
    'backgroundColor': '#0f0f23', 'minHeight': '100vh',
    'fontFamily': "'Inter', 'Segoe UI', sans-serif", 'color': '#e0e0e0',
    'padding': '20px'
}, children=[
    
    # Header
    html.Div(style={
        'textAlign': 'center', 'padding': '30px 0',
        'background': 'linear-gradient(135deg, #1a1a3e 0%, #0f0f23 100%)',
        'borderRadius': '16px', 'marginBottom': '20px',
        'border': '1px solid rgba(69,183,209,0.3)',
        'boxShadow': '0 8px 32px rgba(0,0,0,0.3)',
    }, children=[
        html.H1("Multi-Digital Twin Wind Power Forecasting",
                style={'color': '#45B7D1', 'fontSize': '2.2em', 'margin': '0',
                       'fontWeight': '700', 'letterSpacing': '1px'}),
        html.P("Indian Wind Dataset | SiteID: 36565 | Lat: 23.03 N, Lon: 72.56 E | Year 2014",
               style={'color': '#888', 'fontSize': '1em', 'marginTop': '8px'}),
        html.P("P = 0.5 x rho x A x Cp x v^3 | 120m Wind Speed | Window=10",
               style={'color': '#666', 'fontSize': '0.9em'}),
    ]),
    
    # Controls Row
    html.Div(style={
        'display': 'flex', 'gap': '20px', 'marginBottom': '20px',
        'flexWrap': 'wrap',
    }, children=[
        html.Div(style={
            'flex': '1', 'minWidth': '250px',
            'background': 'rgba(255,255,255,0.05)', 'borderRadius': '12px',
            'padding': '16px', 'border': '1px solid rgba(255,255,255,0.1)',
        }, children=[
            html.Label("Select Day (Test Set Index):", style={'fontWeight': '600', 'color': '#45B7D1'}),
            dcc.Slider(id='day-slider', min=0,
                      max=max(0, (len(list(models_data.values())[0]) if models_data else 24) // 24 - 1),
                      value=0, step=1,
                      marks={i: str(i+1) for i in range(0, max(1, (len(list(models_data.values())[0]) if models_data else 24) // 24), 5)},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ]),
        html.Div(style={
            'flex': '1', 'minWidth': '250px',
            'background': 'rgba(255,255,255,0.05)', 'borderRadius': '12px',
            'padding': '16px', 'border': '1px solid rgba(255,255,255,0.1)',
        }, children=[
            html.Label("Select Models:", style={'fontWeight': '600', 'color': '#45B7D1'}),
            dcc.Checklist(
                id='model-checklist',
                options=[{'label': f'  {n}', 'value': n} for n in ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']],
                value=['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN'],
                inline=True,
                style={'marginTop': '8px'},
                inputStyle={'marginRight': '4px'},
                labelStyle={'marginRight': '20px', 'color': '#ccc'},
            ),
        ]),
    ]),
    
    # 1-Day Forecast Chart
    html.Div(style={
        'background': 'rgba(255,255,255,0.03)', 'borderRadius': '12px',
        'padding': '20px', 'marginBottom': '20px',
        'border': '1px solid rgba(255,255,255,0.08)',
    }, children=[
        html.H3("1-Day Forecast", style={'color': '#FFEAA7', 'margin': '0 0 10px 0'}),
        dcc.Graph(id='day-forecast-chart', config={'displayModeBar': True}),
    ]),
    
    # Metrics Cards Row
    html.Div(id='metrics-cards', style={
        'display': 'flex', 'gap': '16px', 'marginBottom': '20px', 'flexWrap': 'wrap',
    }),
    
    # Two-column layout
    html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
        # Metrics Comparison Chart
        html.Div(style={
            'flex': '1', 'minWidth': '400px',
            'background': 'rgba(255,255,255,0.03)', 'borderRadius': '12px',
            'padding': '20px', 'border': '1px solid rgba(255,255,255,0.08)',
        }, children=[
            html.H3("Single DT Metrics", style={'color': '#FFEAA7', 'margin': '0 0 10px 0'}),
            dcc.Graph(id='metrics-chart'),
        ]),
        # Scatter
        html.Div(style={
            'flex': '1', 'minWidth': '400px',
            'background': 'rgba(255,255,255,0.03)', 'borderRadius': '12px',
            'padding': '20px', 'border': '1px solid rgba(255,255,255,0.08)',
        }, children=[
            html.H3("Actual vs Predicted", style={'color': '#FFEAA7', 'margin': '0 0 10px 0'}),
            dcc.Graph(id='scatter-chart'),
        ]),
    ]),
    
    # Fusion Results
    html.Div(style={
        'background': 'rgba(255,255,255,0.03)', 'borderRadius': '12px',
        'padding': '20px', 'marginTop': '20px',
        'border': '1px solid rgba(255,255,255,0.08)',
    }, children=[
        html.H3("MDT Fusion Results", style={'color': '#FFEAA7', 'margin': '0 0 10px 0'}),
        html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
            html.Div(style={'flex': '1', 'minWidth': '400px'}, children=[
                html.H4("Method 1: Single Metric Preference", style={'color': '#E74C3C'}),
                html.Div(id='m1-table'),
            ]),
            html.Div(style={'flex': '1', 'minWidth': '400px'}, children=[
                html.H4("Method 2: DS Evidence Fusion", style={'color': '#3498DB'}),
                html.Div(id='m2-table'),
            ]),
        ]),
    ]),
    
    # Footer
    html.Div(style={'textAlign': 'center', 'padding': '20px', 'color': '#555', 'marginTop': '30px'},
             children=[html.P("Multi-Digital Twin Wind Power Forecasting | BTP Project 2024")]),
])


def make_metric_card(title, value, color):
    return html.Div(style={
        'flex': '1', 'minWidth': '150px',
        'background': f'linear-gradient(135deg, {color}22, {color}11)',
        'borderRadius': '12px', 'padding': '20px',
        'border': f'1px solid {color}44', 'textAlign': 'center',
    }, children=[
        html.P(title, style={'color': '#999', 'margin': '0', 'fontSize': '0.85em'}),
        html.H2(f"{value:.4f}" if isinstance(value, float) else str(value),
                style={'color': color, 'margin': '8px 0 0 0', 'fontSize': '1.6em'}),
    ])


def make_html_table(df):
    if df is None:
        return html.P("Run notebook first to generate results", style={'color': '#666'})
    header = [html.Th(c, style={'padding': '8px 12px', 'borderBottom': '2px solid #333',
                                 'color': '#45B7D1'}) for c in ['Combination'] + list(df.columns)]
    rows = []
    for idx, row in df.iterrows():
        cells = [html.Td(str(idx), style={'padding': '6px 12px', 'color': '#ccc', 'fontWeight': '600'})]
        cells += [html.Td(f"{v:.4f}", style={'padding': '6px 12px', 'color': '#aaa'}) for v in row]
        rows.append(html.Tr(cells, style={'borderBottom': '1px solid #222'}))
    return html.Table([html.Thead(html.Tr(header)), html.Tbody(rows)],
                     style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '0.9em'})


@callback(
    [Output('day-forecast-chart', 'figure'),
     Output('metrics-cards', 'children'),
     Output('metrics-chart', 'figure'),
     Output('scatter-chart', 'figure'),
     Output('m1-table', 'children'),
     Output('m2-table', 'children')],
    [Input('day-slider', 'value'),
     Input('model-checklist', 'value')]
)
def update_dashboard(day_idx, selected_models):
    template = 'plotly_dark'
    bg = 'rgba(0,0,0,0)'
    
    # 1-Day Forecast
    fig_day = go.Figure()
    fig_day.update_layout(template=template, paper_bgcolor=bg, plot_bgcolor=bg,
                         xaxis_title='Hour', yaxis_title='Wind Power (normalized)',
                         height=350, margin=dict(l=40, r=20, t=30, b=40))
    
    start = day_idx * 24
    if models_data:
        first_model = list(models_data.values())[0]
        end = min(start + 24, len(first_model))
        hours = list(range(end - start))
        
        actual = first_model['actual'].values[start:end]
        fig_day.add_trace(go.Scatter(x=hours, y=actual, mode='lines+markers',
                                    name='Actual', line=dict(color=COLORS['actual'], width=3),
                                    marker=dict(size=6)))
        
        for name in selected_models:
            if name in models_data:
                pred = models_data[name]['prediction'].values[start:end]
                fig_day.add_trace(go.Scatter(x=hours, y=pred, mode='lines',
                                            name=name, line=dict(color=COLORS[name], width=2, dash='dash')))
    
    # Metrics Cards
    cards = []
    if single_metrics is not None:
        best = single_metrics['MAE'].idxmin()
        cards.append(make_metric_card("Best Model", best, '#45B7D1'))
        cards.append(make_metric_card("Best MAE", single_metrics.loc[best, 'MAE'], '#4ECDC4'))
        cards.append(make_metric_card("Best RMSE", single_metrics.loc[best, 'RMSE'], '#FF6B6B'))
        cards.append(make_metric_card("Best R2", single_metrics.loc[best, 'R2'], '#96CEB4'))
    else:
        cards.append(make_metric_card("Status", "Run notebook first", '#666'))
    
    # Metrics Bar Chart
    fig_metrics = go.Figure()
    fig_metrics.update_layout(template=template, paper_bgcolor=bg, plot_bgcolor=bg,
                             height=350, margin=dict(l=40, r=20, t=30, b=40),
                             barmode='group')
    if single_metrics is not None:
        for metric in ['MAE', 'RMSE']:
            fig_metrics.add_trace(go.Bar(
                x=single_metrics.index, y=single_metrics[metric],
                name=metric, marker_color='#FF6B6B' if metric == 'MAE' else '#45B7D1'
            ))
    
    # Scatter
    fig_scatter = go.Figure()
    fig_scatter.update_layout(template=template, paper_bgcolor=bg, plot_bgcolor=bg,
                             height=350, margin=dict(l=40, r=20, t=30, b=40),
                             xaxis_title='Actual', yaxis_title='Predicted')
    for name in selected_models:
        if name in models_data:
            df = models_data[name]
            fig_scatter.add_trace(go.Scatter(
                x=df['actual'], y=df['prediction'], mode='markers',
                name=name, marker=dict(color=COLORS[name], size=3, opacity=0.4)
            ))
    if models_data:
        all_vals = pd.concat([df[['actual', 'prediction']] for df in models_data.values()])
        mn, mx = all_vals.min().min(), all_vals.max().max()
        fig_scatter.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                        name='Perfect', line=dict(color='red', dash='dash', width=1),
                                        showlegend=False))
    
    return fig_day, cards, fig_metrics, fig_scatter, make_html_table(m1_metrics), make_html_table(m2_metrics)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  MDT Wind Power Forecasting Dashboard")
    print("  Open: http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=True, port=8050)
