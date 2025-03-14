def create_visualizations(predictions):
    """
    Create visualizations for the dashboard with updated metrics focus
    
    Parameters:
    - predictions: Formatted prediction results from format_predictions()
    
    Returns:
    - Dictionary of plotly figures for different visualizations
    """
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    
    figures = {}
    
    # 1. OPEN RATE COMPARISON CHART
    open_rate_data = pd.DataFrame({
        'Scenario': ['Current Campaign', 'Recommended Targeting', 'Recommended Subject', 'Combined'],
        'Open Rate (%)': [
            predictions['current']['open_rate'],
            predictions['targeting']['open_rate'],
            predictions['subject']['open_rate'],
            predictions['combined']['open_rate']
        ]
    })
    
    # Calculate improvements for open rate
    open_rate_data['Improvement (%)'] = open_rate_data['Open Rate (%)'] - predictions['current']['open_rate']
    open_rate_data['Improvement (%)'] = open_rate_data['Improvement (%)'].round(2)
    
    # Create figure
    fig_open_rate = px.bar(
        open_rate_data, 
        x='Scenario', 
        y='Open Rate (%)',
        color='Scenario',
        text=open_rate_data['Open Rate (%)'].round(2).astype(str) + '%',
        title='Predicted Open Rates by Scenario',
        height=400
    )
    
    # Add improvement annotations
    for i, row in open_rate_data.iterrows():
        if i > 0:  # Skip the current scenario
            fig_open_rate.add_annotation(
                x=row['Scenario'],
                y=row['Open Rate (%)'] + 1,  # Slightly above the bar
                text=f"+{row['Improvement (%)']}%" if row['Improvement (%)'] >= 0 else f"{row['Improvement (%)']}%",
                showarrow=False,
                font=dict(color="green" if row['Improvement (%)'] >= 0 else "red", size=14)
            )
    
    fig_open_rate.update_traces(textposition='auto')
    fig_open_rate.update_layout(
        xaxis_title="",
        yaxis_title="Open Rate (%)",
        legend_title="Scenario"
    )
    
    figures['open_rate'] = fig_open_rate
    
    # 2. TARGETING METRICS COMPARISON (only for current vs targeting vs combined)
    # Prepare data for multi-metric comparison
    targeting_metrics = pd.DataFrame({
        'Metric': ['Open Rate (%)', 'Click Rate (%)', 'Optout Rate (%)'],
        'Current': [
            predictions['current']['open_rate'],
            predictions['current']['click_rate'],
            predictions['current']['optout_rate']
        ],
        'Recommended Targeting': [
            predictions['targeting']['open_rate'],
            predictions['targeting']['click_rate'],
            predictions['targeting']['optout_rate']
        ],
        'Combined Recommendation': [
            predictions['combined']['open_rate'],
            predictions['combined']['click_rate'],
            predictions['combined']['optout_rate']
        ]
    })
    
    # Melt the dataframe for easier plotting
    targeting_metrics_melted = pd.melt(
        targeting_metrics, 
        id_vars=['Metric'], 
        var_name='Scenario', 
        value_name='Value'
    )
    
    # Create a grouped bar chart
    fig_targeting = px.bar(
        targeting_metrics_melted,
        x='Metric',
        y='Value',
        color='Scenario',
        barmode='group',
        text=targeting_metrics_melted['Value'].round(2).astype(str) + '%',
        title='Targeting Impact on All Metrics',
        height=500
    )
    
    fig_targeting.update_traces(textposition='auto')
    fig_targeting.update_layout(
        xaxis_title="",
        yaxis_title="Percentage (%)",
        legend_title="Scenario"
    )
    
    figures['targeting_metrics'] = fig_targeting
    
    # 3. SUBJECT LINE IMPACT (focusing only on open rate)
    # Create a gauge chart to show the potential impact on open rate
    current_open = predictions['current']['open_rate']
    subject_open = predictions['subject']['open_rate']
    improvement_percentage = ((subject_open - current_open) / current_open) * 100 if current_open > 0 else 0
    
    fig_subject_impact = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=subject_open,
        delta={'reference': current_open, 'relative': False, 'valueformat': '.2f'},
        title={'text': "Subject Line Impact on Open Rate (%)"},
        gauge={
            'axis': {'range': [None, max(subject_open * 1.2, current_open * 1.2)]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, current_open], 'color': "lightgray"},
                {'range': [current_open, subject_open], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': current_open
            }
        }
    ))
    
    figures['subject_impact'] = fig_subject_impact
    
    # 4. COMBINED RECOMMENDATION IMPACT VISUALIZATION
    # Create a radar chart for the combined recommendation vs current
    categories = ['Open Rate', 'Click Rate', 'CTR', 'Engagement', 'Retention']
    
    # Calculate CTR (Click-to-Open Rate)
    current_ctr = (predictions['current']['click_rate'] / predictions['current']['open_rate'] * 100) if predictions['current']['open_rate'] > 0 else 0
    combined_ctr = (predictions['combined']['click_rate'] / predictions['combined']['open_rate'] * 100) if predictions['combined']['open_rate'] > 0 else 0
    
    # Normalize values between 0-100 for radar chart
    max_open = max(predictions['current']['open_rate'], predictions['combined']['open_rate'])
    max_click = max(predictions['current']['click_rate'], predictions['combined']['click_rate'])
    max_ctr = max(current_ctr, combined_ctr)
    
    # Engagement score (example calculation - customize as needed)
    current_engagement = max(0, 60 - predictions['current']['optout_rate'] * 2)  # Lower optout = higher engagement
    combined_engagement = max(0, 60 - predictions['combined']['optout_rate'] * 2)
    
    # Retention score (inverse of optout rate)
    current_retention = max(0, 100 - predictions['current']['optout_rate'] * 5)
    combined_retention = max(0, 100 - predictions['combined']['optout_rate'] * 5)
    
    # Create data for radar chart
    radar_data = [
        go.Scatterpolar(
            r=[
                predictions['current']['open_rate'] / max_open * 100 if max_open > 0 else 0,
                predictions['current']['click_rate'] / max_click * 100 if max_click > 0 else 0,
                current_ctr / max_ctr * 100 if max_ctr > 0 else 0,
                current_engagement,
                current_retention
            ],
            theta=categories,
            fill='toself',
            name='Current Campaign'
        ),
        go.Scatterpolar(
            r=[
                predictions['combined']['open_rate'] / max_open * 100 if max_open > 0 else 0,
                predictions['combined']['click_rate'] / max_click * 100 if max_click > 0 else 0,
                combined_ctr / max_ctr * 100 if max_ctr > 0 else 0,
                combined_engagement,
                combined_retention
            ],
            theta=categories,
            fill='toself',
            name='Optimized Campaign'
        )
    ]
    
    layout_radar = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Campaign Performance Metrics"
    )
    
    fig_radar = go.Figure(data=radar_data, layout=layout_radar)
    figures['radar'] = fig_radar
    
    # 5. IMPROVEMENT SUMMARY TABLE
    improvement_data = pd.DataFrame({
        'Metric': ['Open Rate (%)', 'Click Rate (%)', 'Optout Rate (%)'],
        'Current': [
            predictions['current']['open_rate'],
            predictions['current']['click_rate'],
            predictions['current']['optout_rate']
        ],
        'Targeting Recommendation': [
            predictions['targeting']['open_rate'],
            predictions['targeting']['click_rate'],
            predictions['targeting']['optout_rate']
        ],
        'Subject Recommendation': [
            predictions['subject']['open_rate'],
            # For subject, we only predict open rate
            predictions['current']['click_rate'],
            predictions['current']['optout_rate']
        ],
        'Combined Recommendation': [
            predictions['combined']['open_rate'],
            predictions['combined']['click_rate'],
            predictions['combined']['optout_rate']
        ]
    })
    
    # Calculate absolute improvements
    improvement_data['Targeting Improvement'] = improvement_data['Targeting Recommendation'] - improvement_data['Current']
    improvement_data['Subject Improvement'] = improvement_data['Subject Recommendation'] - improvement_data['Current']
    improvement_data['Combined Improvement'] = improvement_data['Combined Recommendation'] - improvement_data['Current']
    
    # For the table visualization, use a styled table
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=['Metric', 'Current', 'Targeting', 'Targeting Δ', 'Subject', 'Subject Δ', 'Combined', 'Combined Δ'],
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[
                improvement_data['Metric'],
                improvement_data['Current'].round(2).astype(str) + '%',
                improvement_data['Targeting Recommendation'].round(2).astype(str) + '%',
                improvement_data['Targeting Improvement'].round(2).astype(str) + '%',
                improvement_data['Subject Recommendation'].round(2).astype(str) + '%',
                improvement_data['Subject Improvement'].round(2).astype(str) + '%',
                improvement_data['Combined Recommendation'].round(2).astype(str) + '%',
                improvement_data['Combined Improvement'].round(2).astype(str) + '%'
            ],
            fill_color=[
                'white',
                'white',
                'white',
                [get_color(val) for val in improvement_data['Targeting Improvement']],
                'white', 
                [get_color(val) for val in improvement_data['Subject Improvement']],
                'white',
                [get_color(val) for val in improvement_data['Combined Improvement']]
            ],
            align='left'
        ))
    ])
    
    fig_table.update_layout(title="Detailed Metrics Comparison")
    figures['table'] = fig_table
    
    return figures


def get_color(value):
    """Get color based on value (green for positive, red for negative)"""
    if value > 0:
        return 'palegreen'
    elif value < 0:
        return 'mistyrose'
    else:
        return 'white'