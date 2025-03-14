def create_visualizations(current_prediction, targeting_prediction, subject_prediction, combined_prediction, model, features_df):
    """Create enhanced visualizations for the dashboard"""
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    import streamlit as st
    
    # 1. MAIN COMPARISON CHART
    # Create data for the comparison chart
    comparison_data = pd.DataFrame({
        'Scenario': ['Current', 'Recommended Targeting', 'Recommended Subject', 'Combined'],
        'Open Rate (%)': [
            current_prediction, 
            targeting_prediction, 
            subject_prediction, 
            combined_prediction
        ]
    })
    
    # Calculate improvements
    comparison_data['Improvement (%)'] = comparison_data['Open Rate (%)'] - current_prediction
    comparison_data['Improvement (%)'] = comparison_data['Improvement (%)'].round(2)
    
    # Create figure
    fig_comparison = px.bar(
        comparison_data, 
        x='Scenario', 
        y='Open Rate (%)',
        color='Scenario',
        text=comparison_data['Open Rate (%)'].round(2).astype(str) + '%',
        title='Predicted Open Rates by Scenario',
        height=400
    )
    
    # Add improvement annotations
    for i, row in comparison_data.iterrows():
        if i > 0:  # Skip the current scenario
            fig_comparison.add_annotation(
                x=row['Scenario'],
                y=row['Open Rate (%)'] + 1,  # Slightly above the bar
                text=f"+{row['Improvement (%)']}%",
                showarrow=False,
                font=dict(color="green", size=14)
            )
    
    fig_comparison.update_traces(textposition='auto')
    fig_comparison.update_layout(
        xaxis_title="",
        yaxis_title="Open Rate (%)",
        legend_title="Scenario"
    )
    
    # 2. RADAR CHART FOR COMPARISON
    # Create categories for the radar chart
    categories = ['Open Rate', 'Subject Quality', 'Targeting Fit', 'Overall Score']
    
    # Normalize values between 0-100
    max_open_rate = max(current_prediction, targeting_prediction, subject_prediction, combined_prediction)
    
    # Calculate subject quality and targeting fit (examples - adjust with your actual metrics)
    current_subject_quality = 70  # Baseline
    recommended_subject_quality = 90
    
    current_targeting_fit = 65  # Baseline
    recommended_targeting_fit = 85
    
    # Create data for radar chart
    radar_data = [
        go.Scatterpolar(
            r=[
                current_prediction / max_open_rate * 100,
                current_subject_quality,
                current_targeting_fit,
                (current_prediction / max_open_rate * 100 + current_subject_quality + current_targeting_fit) / 3
            ],
            theta=categories,
            fill='toself',
            name='Current'
        ),
        go.Scatterpolar(
            r=[
                combined_prediction / max_open_rate * 100,
                recommended_subject_quality,
                recommended_targeting_fit,
                (combined_prediction / max_open_rate * 100 + recommended_subject_quality + recommended_targeting_fit) / 3
            ],
            theta=categories,
            fill='toself',
            name='Recommended'
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
    
    # 3. IMPACT VISUALIZATION
    # Create a gauge chart to show the potential impact
    improvement_percentage = ((combined_prediction - current_prediction) / current_prediction) * 100
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=combined_prediction,
        delta={'reference': current_prediction, 'relative': True, 'valueformat': '.1%'},
        title={'text': "Potential Open Rate with Recommendations"},
        gauge={
            'axis': {'range': [None, max(combined_prediction * 1.2, current_prediction * 1.2)]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, current_prediction], 'color': "lightgray"},
                {'range': [current_prediction, combined_prediction], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': current_prediction
            }
        }
    ))
    
    # 4. FEATURE IMPORTANCE CHART (if model supports it)
    fig_features = None
    if hasattr(model, 'feature_importances_'):
        # For simplicity, we'll assume this is an estimator with feature_importances_
        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        fig_features = px.bar(
            feature_importance,
            y='feature',
            x='importance',
            title='Top 10 Features Influencing Open Rate',
            orientation='h',
            color='importance'
        )
        fig_features.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    # Return all figures
    return {
        'comparison': fig_comparison,
        'radar': fig_radar,
        'gauge': fig_gauge,
        'features': fig_features
    }