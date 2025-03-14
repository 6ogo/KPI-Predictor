def create_bolag_analysis(delivery_df, customer_df):
    """
    Create analysis of metrics by Bolag (company/organization)
    
    Parameters:
    - delivery_df: DataFrame with delivery data
    - customer_df: DataFrame with customer data including Bolag information
    
    Returns:
    - Dictionary of plotly figures for Bolag analysis
    """
    import pandas as pd
    import plotly.express as px
    import numpy as np
    
    figures = {}
    
    # First, need to merge customer data (which has Bolag) with delivery data (which has metrics)
    # This needs to be done carefully to avoid duplicate counting
    
    # Step 1: Create a mapping of InternalName to metrics from delivery data
    delivery_metrics = delivery_df[['InternalName', 'open_rate', 'click_rate', 'optout_rate']].copy()
    
    # Step 2: Group customer data by Bolag and InternalName, to get one row per delivery per Bolag
    customer_deliveries = customer_df.groupby(['Bolag', 'InternalName']).agg(
        customer_count=pd.NamedAgg(column='Primary k isWoman OptOut', aggfunc='count')
    ).reset_index()
    
    # Step 3: Merge with delivery metrics
    bolag_metrics = customer_deliveries.merge(delivery_metrics, on='InternalName', how='left')
    
    # Step 4: Group by Bolag to get averages
    bolag_performance = bolag_metrics.groupby('Bolag').agg(
        avg_open_rate=pd.NamedAgg(column='open_rate', aggfunc='mean'),
        avg_click_rate=pd.NamedAgg(column='click_rate', aggfunc='mean'),
        avg_optout_rate=pd.NamedAgg(column='optout_rate', aggfunc='mean'),
        total_deliveries=pd.NamedAgg(column='InternalName', aggfunc='count'),
        total_customers=pd.NamedAgg(column='customer_count', aggfunc='sum')
    ).reset_index()
    
    # Handle potential NaN values
    bolag_performance.fillna({
        'avg_open_rate': 0,
        'avg_click_rate': 0,
        'avg_optout_rate': 0
    }, inplace=True)
    
    # Create visualizations
    
    # 1. Open Rate by Bolag
    bolag_performance_sorted = bolag_performance.sort_values('avg_open_rate', ascending=False)
    
    fig_open_rate = px.bar(
        bolag_performance_sorted,
        x='Bolag',
        y='avg_open_rate',
        title='Average Open Rate by Company (Bolag)',
        color='avg_open_rate',
        text=bolag_performance_sorted['avg_open_rate'].round(1).astype(str) + '%',
        hover_data=['total_deliveries', 'total_customers']
    )
    fig_open_rate.update_traces(textposition='auto')
    fig_open_rate.update_layout(
        xaxis_title='Company (Bolag)',
        yaxis_title='Average Open Rate (%)',
        xaxis={'categoryorder': 'total descending'}
    )
    figures['open_rate'] = fig_open_rate
    
    # 2. Click Rate by Bolag
    bolag_performance_sorted = bolag_performance.sort_values('avg_click_rate', ascending=False)
    
    fig_click_rate = px.bar(
        bolag_performance_sorted,
        x='Bolag',
        y='avg_click_rate',
        title='Average Click Rate by Company (Bolag)',
        color='avg_click_rate',
        text=bolag_performance_sorted['avg_click_rate'].round(1).astype(str) + '%',
        hover_data=['total_deliveries', 'total_customers']
    )
    fig_click_rate.update_traces(textposition='auto')
    fig_click_rate.update_layout(
        xaxis_title='Company (Bolag)',
        yaxis_title='Average Click Rate (%)',
        xaxis={'categoryorder': 'total descending'}
    )
    figures['click_rate'] = fig_click_rate
    
    # 3. Optout Rate by Bolag
    bolag_performance_sorted = bolag_performance.sort_values('avg_optout_rate', ascending=True)  # Lower is better
    
    fig_optout_rate = px.bar(
        bolag_performance_sorted,
        x='Bolag',
        y='avg_optout_rate',
        title='Average Optout Rate by Company (Bolag)',
        color='avg_optout_rate',
        color_continuous_scale='Reds_r',  # Reversed scale (red is bad)
        text=bolag_performance_sorted['avg_optout_rate'].round(1).astype(str) + '%',
        hover_data=['total_deliveries', 'total_customers']
    )
    fig_optout_rate.update_traces(textposition='auto')
    fig_optout_rate.update_layout(
        xaxis_title='Company (Bolag)',
        yaxis_title='Average Optout Rate (%)',
        xaxis={'categoryorder': 'total ascending'}
    )
    figures['optout_rate'] = fig_optout_rate
    
    # 4. Combined metrics comparison
    # Prepare data for a grouped bar chart
    metrics_data = []
    for _, row in bolag_performance.iterrows():
        metrics_data.extend([
            {'Bolag': row['Bolag'], 'Metric': 'Open Rate', 'Value': row['avg_open_rate']},
            {'Bolag': row['Bolag'], 'Metric': 'Click Rate', 'Value': row['avg_click_rate']},
            {'Bolag': row['Bolag'], 'Metric': 'Optout Rate', 'Value': row['avg_optout_rate']}
        ])
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Get top 5 Bolag by total customers for cleaner visualization
    top_bolags = bolag_performance.nlargest(5, 'total_customers')['Bolag'].tolist()
    metrics_df_filtered = metrics_df[metrics_df['Bolag'].isin(top_bolags)]
    
    fig_comparison = px.bar(
        metrics_df_filtered,
        x='Bolag',
        y='Value',
        color='Metric',
        barmode='group',
        title='KPI Comparison Across Top Companies (Bolag)',
        labels={'Value': 'Rate (%)'}
    )
    fig_comparison.update_layout(
        xaxis_title='Company (Bolag)',
        yaxis_title='Rate (%)',
        legend_title='Metric'
    )
    figures['comparison'] = fig_comparison
    
    return figures, bolag_performance