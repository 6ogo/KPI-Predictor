def calculate_confidence(prediction, model_performance, data_quality=0.8):
    """
    Calculate a confidence score for predictions.
    
    Parameters:
    - prediction: The predicted value
    - model_performance: Model performance metrics (MAE)
    - data_quality: A factor indicating data quality (0-1)
    
    Returns:
    - Confidence score as a percentage (0-100)
    """
    # Base confidence based on model MAE relative to prediction
    if prediction == 0:
        base_confidence = 50  # Default for zero predictions
    else:
        # Calculate how many MAEs the prediction is from zero
        maes_from_zero = abs(prediction) / model_performance
        # Higher value means higher confidence
        base_confidence = min(95, 50 + (maes_from_zero * 10))
    
    # Apply data quality factor
    confidence = base_confidence * data_quality
    
    # Ensure confidence is between 20-95%
    confidence = max(20, min(95, confidence))
    
    return confidence

def get_confidence_color(confidence):
    """Return color based on confidence level"""
    if confidence >= 80:
        return "green"
    elif confidence >= 60:
        return "orange"
    else:
        return "red"

def create_visualizations(predictions):
    """
    Create visualizations for the dashboard with updated metrics focus
    and improved color schemes for better visibility
    """
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    
    figures = {}
    
    # Add confidence to predictions if not already there
    if 'confidence' not in predictions['current']:
        # Sample confidence values - in real implementation, these would be calculated
        # based on model performance from predictions['model_performance']
        model_performance = {
            'open_rate': 2.0,  # Sample MAE
            'click_rate': 1.0, 
            'optout_rate': 0.5
        }
        
        # Add confidence to each prediction set
        for scenario in ['current', 'targeting', 'subject', 'combined']:
            predictions[scenario]['confidence'] = {}
            for metric in ['open_rate', 'click_rate', 'optout_rate']:
                if metric in predictions[scenario]:
                    mae = model_performance.get(metric, 1.0)
                    predictions[scenario]['confidence'][metric] = calculate_confidence(
                        predictions[scenario][metric], 
                        mae
                    )
    
    # 1. OPEN RATE COMPARISON CHART
    open_rate_data = pd.DataFrame({
        'Scenario': ['Current Campaign', 'Recommended Targeting', 'Recommended Subject', 'Combined'],
        'Open Rate (%)': [
            predictions['current']['open_rate'],
            predictions['targeting']['open_rate'],
            predictions['subject']['open_rate'],
            predictions['combined']['open_rate']
        ],
        'Confidence (%)': [
            predictions['current']['confidence']['open_rate'],
            predictions['targeting']['confidence']['open_rate'],
            predictions['subject']['confidence']['open_rate'],
            predictions['combined']['confidence']['open_rate']
        ]
    })
    
    # Calculate improvements for open rate
    open_rate_data['Improvement (%)'] = open_rate_data['Open Rate (%)'] - predictions['current']['open_rate']
    open_rate_data['Improvement (%)'] = open_rate_data['Improvement (%)'].round(2)
    
    # Create figure with confidence
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
        
        # Add confidence annotation
        fig_open_rate.add_annotation(
            x=row['Scenario'],
            y=row['Open Rate (%)'] / 2,  # Middle of the bar
            text=f"{row['Confidence (%)']:.0f}% confidence",
            showarrow=False,
            font=dict(
                color="white", 
                size=11
            )
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
        height=500,
        color_discrete_sequence=px.colors.qualitative.Bold  # Use a more vibrant color scheme
    )
    
    fig_targeting.update_traces(textposition='auto')
    fig_targeting.update_layout(
        xaxis_title="",
        yaxis_title="Percentage (%)",
        legend_title="Scenario"
    )
    
    figures['targeting_metrics'] = fig_targeting
    
    # 3. SUBJECT LINE IMPACT WITH CONFIDENCE
    current_open = predictions['current']['open_rate']
    subject_open = predictions['subject']['open_rate']
    confidence = predictions['subject']['confidence']['open_rate']
    
    fig_subject_impact = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=subject_open,
        delta={'reference': current_open, 'relative': False, 'valueformat': '.2f'},
        title={'text': f"Subject Line Impact on Open Rate<br><span style='font-size:0.8em;color:{get_confidence_color(confidence)}'>Confidence: {confidence:.0f}%</span>"},
        gauge={
            'axis': {'range': [None, max(subject_open * 1.2, current_open * 1.2)]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, current_open], 'color': "lightgray"},
                {'range': [current_open, subject_open], 'color': "royalblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': current_open
            }
        }
    ))
    
    figures['subject_impact'] = fig_subject_impact
    
    # 4. IMPROVED RADAR CHART
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
            name='Current Campaign',
            line=dict(color='rgba(31, 119, 180, 0.8)', width=2),
            fillcolor='rgba(31, 119, 180, 0.2)'
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
            name='Optimized Campaign',
            line=dict(color='rgba(44, 160, 44, 0.8)', width=2),
            fillcolor='rgba(44, 160, 44, 0.2)'
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
    
    # 5. IMPROVED METRICS COMPARISON TABLE
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
    
    # Create a more readable table with improved colors
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=['Metric', 'Current', 'Targeting', 'Targeting Δ', 'Subject', 'Subject Δ', 'Combined', 'Combined Δ'],
            font=dict(size=14, color='white'),
            fill_color='#2c3e50',
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
            font=dict(size=13),
            fill_color=[
                ['#f8f9fa'] * len(improvement_data),
                ['#f8f9fa'] * len(improvement_data),
                ['#f8f9fa'] * len(improvement_data),
                [
                    '#8fff9c' if val > 0 else '#ff9c9c' if val < 0 else '#f8f9fa' 
                    for val in improvement_data['Targeting Improvement']
                ],
                ['#f8f9fa'] * len(improvement_data),
                [
                    '#8fff9c' if val > 0 else '#ff9c9c' if val < 0 else '#f8f9fa' 
                    for val in improvement_data['Subject Improvement']
                ],
                ['#f8f9fa'] * len(improvement_data),
                [
                    '#8fff9c' if val > 0 else '#ff9c9c' if val < 0 else '#f8f9fa' 
                    for val in improvement_data['Combined Improvement']
                ]
            ],
            align='left',
            height=30
        ))
    ])
    
    fig_table.update_layout(
        title="Detailed Metrics Comparison",
        margin=dict(l=10, r=10, t=30, b=10),
        height=150 + (len(improvement_data) * 30)  # Dynamic height based on data
    )
    
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