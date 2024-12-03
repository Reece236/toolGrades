import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Union, List
import plotly.express as px
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

class GradesAnalyzer:
    SCALE_REFERENCE = {
        20: "Poor",
        30: "Well Below Average",
        40: "Below Average",
        50: "Average",
        60: "Above Average",
        70: "Well Above Average",
        80: "Elite"
    }
    
    def __init__(self, data_path: str = "results/grades_2024Grades.csv"):
        self.data = pd.read_csv(data_path, index_col="Name")
        self.data = self.data.dropna(subset=['decScore_bayes_grade','decScore_ci_lower_grade','decScore_ci_upper_grade','95th_pBat_speed_bayes_grade','95th_pBat_speed_ci_lower_grade','95th_pBat_speed_ci_upper_grade','smash_factor_bayes_grade','smash_factor_ci_lower_grade','smash_factor_ci_upper_grade','conScore_bayes_grade','conScore_ci_lower_grade','conScore_ci_upper_grade'])
        self.grade_types = [col.replace('_bayes_grade', '') 
                           for col in self.data.columns if '_bayes_grade' in col]
        self.app = dash.Dash(__name__)
        self.setup_layout()
        
    def setup_layout(self):
        # Create more readable column names
        self.column_labels = {  # Made this an instance variable
            'decScore': 'Swing Decision',
            '95th_pBat_speed': 'Power (95th)',
            'smash_factor': 'Smash Factor',
            'conScore': 'Contact',
            'OVRGrade': 'Overall Grade'
        }
        
        leaderboard_data = self.show_leaderboard()
        
        self.app.layout = html.Div([
            dcc.Tabs([
                dcc.Tab(label='Player Analysis', children=[
                    html.Div([
                        dcc.Dropdown(
                            id='player-dropdown',
                            options=[{'label': name, 'value': name} for name in sorted(self.data.index)],
                            multi=True,
                            placeholder='Select players to analyze...'
                        ),
                        dcc.Graph(id='analysis-graph'),
                        html.Div(id='numerical-summary')
                    ])
                ]),
                dcc.Tab(label='Leaderboard', children=[
                    dash_table.DataTable(
                        id='leaderboard-table',
                        data=leaderboard_data.round(2).to_dict('records'),
                        columns=[{
                            'id': col,
                            'name': self.column_labels.get(col, col),
                            'type': 'numeric',
                            'format': {'specifier': '.2f'}
                        } for col in leaderboard_data.columns],
                        sort_action='native',
                        sort_mode='multi',
                        style_table={
                            'overflowX': 'auto',
                            'maxHeight': '800px',
                            'overflowY': 'auto'
                        },
                        style_cell={
                            'textAlign': 'center',
                            'padding': '10px',
                            'minWidth': '100px'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            },
                            {
                                'if': {'column_type': 'numeric'},
                                'format': {'specifier': '.2f'}
                            }
                        ],
                        page_size=20,
                        filter_action='native'  # Enable filtering
                    )
                ])
            ])
        ])
        
        @self.app.callback(
            [Output('analysis-graph', 'figure'),
             Output('numerical-summary', 'children')],
            [Input('player-dropdown', 'value')]
        )
        def update_analysis(selected_players):
            if not selected_players:
                return {}, ''
            
            # Create a color map for selected players using hex colors
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(selected_players)]
            player_colors = dict(zip(selected_players, colors))
            
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=[self.column_labels.get(grade, grade) 
                                           for grade in self.grade_types])
            
            # Add reference lines and grade labels
            for idx, grade in enumerate(self.grade_types, 1):
                row = (idx - 1) // 2 + 1
                col = (idx - 1) % 2 + 1
                
                for grade_point, label in self.SCALE_REFERENCE.items():
                    fig.add_vline(x=grade_point, line_dash="dot", line_width=1,
                                line_color="gray", opacity=0.5,
                                annotation_text=f"{grade_point}: {label}",
                                annotation_position="top",
                                row=row, col=col)
            
            summary_text = []
            for player in selected_players:
                player_data = self.data.loc[player]
                player_color = player_colors[player]
                
                for idx, grade in enumerate(self.grade_types, 1):
                    row = (idx - 1) // 2 + 1
                    col = (idx - 1) % 2 + 1
                    
                    mean = player_data[f'{grade}_bayes_grade']
                    ci_lower = player_data[f'{grade}_ci_lower_grade']
                    ci_upper = player_data[f'{grade}_ci_upper_grade']
                    
                    if grade == '95th_pBat_speed':
                        # Just add point estimate for bat speed
                        fig.add_trace(
                            go.Scatter(x=[mean], y=[1], 
                                     mode='markers',
                                     name=player,  # Simplified name
                                     marker=dict(size=15, color=player_color)),
                            row=row, col=col
                        )
                    else:
                        # Create distribution for other metrics
                        std = (ci_upper - ci_lower) / (2 * 1.96)
                        x = np.linspace(20, 80, 100)
                        y = np.exp(-0.5 * ((x - mean) / std) ** 2)
                        y = y / np.max(y)
                        
                        fig.add_trace(
                            go.Scatter(x=x, y=y, 
                                     name=player,  # Simplified name
                                     fill='tozeroy',
                                     line=dict(color=player_color),
                                     fillcolor=f'rgba{tuple(int(player_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}'),
                            row=row, col=col
                        )
                    
                    # Add vertical lines with player color
                    fig.add_vline(x=mean, line_dash="solid", line_width=2,
                                line_color=player_color, row=row, col=col)
                    if grade != '95th_pBat_speed':
                        fig.add_vline(x=ci_lower, line_dash="dash", 
                                    line_color=player_color, row=row, col=col)
                        fig.add_vline(x=ci_upper, line_dash="dash",
                                    line_color=player_color, row=row, col=col)
                
                # Update summary text with readable labels
                summary_text.append(html.H4(player))
                for grade in self.grade_types:
                    mean = player_data[f'{grade}_bayes_grade']
                    label = self.column_labels.get(grade, grade)
                    summary_text.append(html.P(f"{label}: {mean:.2f}"))
                avg = player_data[[c for c in self.data.columns if '_bayes_grade' in c]].mean()
                summary_text.append(html.P(f"Overall Grade: {avg:.2f}"))
            
            fig.update_layout(height=800, showlegend=True, template="simple_white")
            
            return fig, html.Div(summary_text)
    
    def analyze_player(self, player_name: Union[str, List[str]], save_plot: bool = False):
        """Analyze one or more players and show their posterior distributions"""
        if isinstance(player_name, str):
            player_name = [player_name]
        
        # Debug print available columns
        print("Available columns:", self.data.columns.tolist())
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=self.grade_types)
        
        for player in player_name:
            player_data = self.data.loc[player]
            print(f"\nData for {player}:")
            print(player_data)
            
            for idx, grade in enumerate(self.grade_types, 1):
                row = (idx - 1) // 2 + 1
                col = (idx - 1) % 2 + 1
                
                # Get values using consistent column names
                mean = player_data[f'{grade}_bayes_grade']
                ci_lower = player_data[f'{grade}_ci_lower_grade']
                ci_upper = player_data[f'{grade}_ci_upper_grade']
                
                # Create proper normal distribution
                std = (ci_upper - ci_lower) / (2 * 1.96)  # Approximate std from CI
                x = np.linspace(20, 80, 100)  # Full 20-80 scale range
                y = np.exp(-0.5 * ((x - mean) / std) ** 2)
                y = y / np.max(y)  # Normalize to [0,1]
                
                fig.add_trace(
                    go.Scatter(x=x, y=y, name=f"{player} - {grade}", fill='tozeroy'),
                    row=row, col=col
                )
                
                # Add vertical lines
                fig.add_vline(x=mean, line_dash="solid", line_width=2, 
                             line_color="black", row=row, col=col)
                fig.add_vline(x=ci_lower, line_dash="dash", row=row, col=col)
                fig.add_vline(x=ci_upper, line_dash="dash", row=row, col=col)
                
                # Update axes
                fig.update_xaxes(title="Grade (20-80)", range=[20, 80], row=row, col=col)
                fig.update_yaxes(title="Density", range=[0, 1.1], row=row, col=col)

        fig.update_layout(
            height=800, 
            title="Player Analysis",
            showlegend=True,
            template="simple_white"
        )
        
        if save_plot:
            fig.write_html(f"player_analysis_{'_'.join(player_name)}.html")
        else:
            fig.show()
            
        # Print numerical summary
        print("\nNumerical Summary:")
        for player in player_name:
            player_data = self.data.loc[player]
            print(f"\n{player}:")
            for grade in self.grade_types:
                mean = player_data[f'{grade}_bayes_grade']
                print(f"{grade}: {mean:.2f}")
            print(f"Average: {player_data[[c for c in self.data.columns if '_bayes_grade' in c]].mean():.2f}")

    def show_leaderboard(self, save_csv: bool = False):
        """Display and optionally save the leaderboard"""
        grade_cols = [col for col in self.data.columns if '_bayes_grade' in col]
        
        # Create leaderboard with all scores and average
        leaderboard = self.data[grade_cols].copy()
        leaderboard['Average'] = leaderboard.mean(axis=1)
        
        # Add player names as a column instead of index
        leaderboard = leaderboard.reset_index()
        leaderboard = leaderboard.rename(columns={'index': 'Name'})
        
        # Reorder columns to put name first
        cols = ['Name'] + [col for col in leaderboard.columns if col != 'Name']
        leaderboard = leaderboard[cols]
        
        # Sort by average score
        leaderboard = leaderboard.sort_values('Average', ascending=False)
        leaderboard = leaderboard.round(2)
        
        if save_csv:
            leaderboard.to_csv('leaderboard.csv', index=False)
        
        return leaderboard
    
    def run_server(self, debug=True, port=8050):
        self.app.run_server(debug=debug, port=port)

# Example usage
if __name__ == "__main__":
    analyzer = GradesAnalyzer()
    analyzer.run_server()