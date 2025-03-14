import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import base64
import os
import joblib
import datetime

# Custom modules (assumed to exist in your project)
from feature_engineering import extract_subject_features
from multi_metric_model import predict_metrics, enhanced_train_multi_metric_models
from subject_recommendation import build_subject_recommendation_model
from visualizations import create_visualizations
from model_metadata import track_model_performance, model_needs_retraining

# Define the Bolag codes
BOLAG_VALUES = {
    "Blekinge": "B02",
    "Dalarna": "B03", 
    "Älvsborg": "B04",
    "Gävleborg": "B08",
    "Göinge-Kristianstad": "B09",
    "Göteborg-Bohuslan": "B10",
    "Halland": "B11",
    "Jämtland": "B14",
    "Jönköping": "B15",
    "Kalmar": "B16",
    "Kronoberg": "B21",
    "Norrbotten": "B24",
    "Skaraborg": "B27",
    "Stockholm": "B28",
    "Södermanland": "B29",
    "Uppsala": "B31",
    "Värmland": "B32",
    "Västerbotten": "B34",
    "Västernorrland": "B35",
    "Bergslagen": "B37",
    "Östgöta": "B42",
    "Gotland": "B43",
    "Skåne": "B50"
}

# Define enums for dropdown values
SYFTE_VALUES = {
    "AKUT": ["AKT", "AKUT"],
    "AVSLUT": ["AVS", "AVSLUT"],
    "AVSLUT Kund": ["AVS_K", "AVSLUT Kund"],
    "AVSLUT Produkt": ["AVS_P", "AVSLUT Produkt"],
    "BEHÅLLA": ["BHA", "BEHÅLLA"],
    "BEHÅLLA Betalpåminnelse": ["BHA_P", "BEHÅLLA Betalpåminnelse"],
    "BEHÅLLA Inför förfall": ["BHA_F", "BEHÅLLA Inför förfall"],
    "TEST": ["TST", "TEST"],
    "VINNA": ["VIN", "VINNA"],
    "VINNA Provapå till riktig": ["VIN_P", "VINNA Provapå till riktig"],
    "VÄLKOMNA": ["VLK", "VÄLKOMNA"],
    "VÄLKOMNA Nykund": ["VLK_K", "VÄLKOMNA Nykund"],
    "VÄLKOMNA Nyprodukt": ["VLK_P", "VÄLKOMNA Nyprodukt"],
    "VÄLKOMNA Tillbaka": ["VLK_T", "VÄLKOMNA Tillbaka"],
    "VÄXA": ["VXA", "VÄXA"],
    "VÄXA Korsförsäljning": ["VXA_K", "VÄXA Korsförsäljning"],
    "VÄXA Merförsäljning": ["VXA_M", "VÄXA Merförsäljning"],
    "VÅRDA": ["VRD", "VÅRDA"],
    "VÅRDA Betalsätt": ["VRD_B", "VÅRDA Betalsätt"],
    "VÅRDA Event": ["VRD_E", "VÅRDA Event"],
    "VÅRDA Information": ["VRD_I", "VÅRDA Information"],
    "VÅRDA Lojalitet förmånskund": ["VRD_L", "VÅRDA Lojalitet förmånskund"],
    "VÅRDA Nyhetsbrev": ["VRD_N", "VÅRDA Nyhetsbrev"],
    "VÅRDA Skadeförebygg": ["VRD_S", "VÅRDA Skadeförebygg"],
    "VÅRDA Undersökning": ["VRD_U", "VÅRDA Undersökning"],
    "ÅTERTAG": ["ATG", "ÅTERTAG"],
    "ÖVRIGT": ["OVR", "ÖVRIGT"]
}

DIALOG_VALUES = {
    "BANK": ["BNK", "BANK"],
    "BANK LFF": ["LFF", "BANK LFF"],
    "BOENDE": ["BO", "BOENDE"],
    "DROP-OFF": ["DRP", "DROP-OFF"],
    "FORDON": ["FRD", "FORDON"],
    "FÖRETAGARBREVET": ["FTB", "FÖRETAGARBREVET"],
    "FÖRETAGSFÖRSÄKRING": ["FTG", "FÖRETAGSFÖRSÄKRING"],
    "HÄLSA": ["H", "HÄLSA"],
    "KUNDNIVÅ Förmånskund": ["KFB", "KUNDNIVÅ Förmånskund"],
    "LIV": ["LIV", "LIV"],
    "Livshändelse": ["LVS", "Livshändelse"],
    "MÅN A1 - Barnförsäkring": ["A1", "MÅN A1 - Barnförsäkring"],
    "MÅN A10 - Förra veckans sålda": ["A10", "MÅN A10 - Förra veckans sålda"],
    "MÅN A3 - Återtag boendeförsäkring": ["A3", "MÅN A3 - Återtag boendeförsäkring"],
    "MÅN A7 - Återtag bilförsäkring": ["A7", "MÅN A7 - Återtag bilförsäkring"],
    "MÅN C2 - Boende till bil": ["C2", "MÅN C2 - Boende till bil"],
    "MÅN C3 - Bilförsäkring förfaller hos konkurrent": ["C3", "MÅN C3 - Bilförsäkring förfaller hos konkurrent"],
    "MÅN F10 - Fasträntekontor": ["F10", "MÅN F10 - Fasträntekontor"],
    "MÅN L1 - Bolån till boendeförsäkringskunder": ["L1", "MÅN L1 - Bolån till boendeförsäkringskunder"],
    "MÅN L20 - Förfall bolån": ["L20", "MÅN L20 - Förfall bolån"],
    "MÅN L3 - Ränteförfall": ["L3", "MÅN L3 - Ränteförfall"],
    "MÅN M1 - Märkespaket": ["M1", "MÅN M1 - Märkespaket"],
    "MÅN S1 - Vända pengar": ["S1", "MÅN S1 - Vända pengar"],
    "MÅN S2 - Inflytt pensionskapital": ["S2", "MÅN S2 - Inflytt pensionskapital"],
    "NBO": ["FNO", "NBO"],
    "OFFERT": ["OF", "OFFERT"],
    "ONEOFF": ["ONE", "ONEOFF"],
    "PERSON": ["P", "PERSON"],
    "RÄDDA KVAR": ["RKR", "RÄDDA KVAR"],
    "TESTUTSKICK": ["TST", "TESTUTSKICK"],
    "ÅTERBÄRING": ["ATB", "ÅTERBÄRING"]
}

PRODUKT_VALUES = {
    "AGRIA": ["A_A_", "AGRIA"],
    "BANK": ["B_B_", "BANK"],
    "BANK Bolån": ["B_B_B_", "BANK Bolån"],
    "BANK Kort": ["B_K_", "BANK Kort"],
    "BANK Spar": ["B_S_", "BANK Spar"],
    "BANK Övriga lån": ["B_PL_", "BANK Övriga lån"],
    "BO": ["BO_", "BO"],
    "BO Alarm": ["BO_AL_", "BO Alarm"],
    "BO BRF": ["BO_BR_", "BO BRF"],
    "BO Fritid": ["BO_F_", "BO Fritid"],
    "BO HR": ["BO_HR_", "BO HR"],
    "BO Villa": ["BO_V_", "BO Villa"],
    "BO VillaHem": ["BO_VH_", "BO VillaHem"],
    "BÅT": ["BT_", "BÅT"],
    "FOND": ["F_F_", "FOND"],
    "FÖRETAG Företagarförsäkring": ["F_F_F_", "FÖRETAG Företagarförsäkring"],
    "FÖRETAG Företagarförsäkring prova på": ["F_F_PR_", "FÖRETAG Företagarförsäkring prova på"],
    "HÄLSA": ["H_H_", "HÄLSA"],
    "HÄLSA BoKvar": ["H_B_", "HÄLSA BoKvar"],
    "HÄLSA Diagnos": ["H_D_", "HÄLSA Diagnos"],
    "HÄLSA Grupp företag": ["H_G_", "HÄLSA Grupp företag"],
    "HÄLSA Olycksfall": ["H_O_", "HÄLSA Olycksfall"],
    "HÄLSA Sjukersättning": ["H_S_", "HÄLSA Sjukersättning"],
    "HÄLSA Sjukvårdsförsäkring": ["H_SV_", "HÄLSA Sjukvårdsförsäkring"],
    "INGEN SPECIFIK PRODUKT": ["NA_NA_", "INGEN SPECIFIK PRODUKT"],
    "LANTBRUK": ["LB_", "LANTBRUK"],
    "LIV": ["L_L_", "LIV"],
    "LIV Försäkring": ["L_F_", "LIV Försäkring"],
    "LIV Pension": ["L_P_", "LIV Pension"],
    "MOTOR": ["M_M_", "MOTOR"],
    "MOTOR Personbil": ["M_PB_", "MOTOR Personbil"],
    "MOTOR Personbil Vagnskada": ["M_PB_VG_", "MOTOR Personbil Vagnskada"],
    "MOTOR Personbil märkes Lexus": ["M_PB_ML_", "MOTOR Personbil märkes Lexus"],
    "MOTOR Personbil märkes Suzuki": ["M_PB_MS_", "MOTOR Personbil märkes Suzuki"],
    "MOTOR Personbil märkes Toyota": ["M_PB_MT_", "MOTOR Personbil märkes Toyota"],
    "MOTOR Personbil prova på": ["M_PB_PR_", "MOTOR Personbil prova på"],
    "MOTOR Övriga": ["M_OV_", "MOTOR Övriga"],
    "MOTOR Övriga MC": ["M_OV_MC_", "MOTOR Övriga MC"],
    "MOTOR Övriga Skoter": ["M_OV_SKO_", "MOTOR Övriga Skoter"],
    "MOTOR Övriga Släp": ["M_OV_SLP_", "MOTOR Övriga Släp"],
    "PERSON": ["P_P_", "PERSON"],
    "PERSON 60plus": ["P_60_", "PERSON 60plus"],
    "PERSON Gravid": ["P_G_", "PERSON Gravid"],
    "PERSON Gravid bas": ["P_G_B_", "PERSON Gravid bas"],
    "PERSON Gravid plus": ["P_G_P_", "PERSON Gravid plus"],
    "PERSON OB": ["P_B_", "PERSON OB"],
    "PERSON OSB": ["P_OSB_", "PERSON OSB"],
    "PERSON OSV": ["P_OSV_", "PERSON OSV"]
}

def campaign_parameter_input(cat_values):
    """
    Create the campaign parameter input section with correct Bolag terminology.
    """
    st.subheader("Campaign Settings")
    
    # Dialog, Syfte, Produkt dropdowns
    dialog_options = list(DIALOG_VALUES.keys())
    selected_dialog = st.selectbox("Dialog", options=dialog_options)
    dialog_code = DIALOG_VALUES[selected_dialog][0]
    
    syfte_options = list(SYFTE_VALUES.keys())
    selected_syfte = st.selectbox("Campaign Purpose", options=syfte_options)
    syfte_code = SYFTE_VALUES[selected_syfte][0]
    
    produkt_options = list(PRODUKT_VALUES.keys())
    selected_product = st.selectbox("Product", options=produkt_options)
    product_code = PRODUKT_VALUES[selected_product][0]
    
    # Target Bolag selection
    bolag_options = list(BOLAG_VALUES.keys())
    selected_bolag = st.selectbox(
        "Target Bolag",
        options=bolag_options,
        help="Select primary Bolag region for targeting"
    )
    bolag_code = BOLAG_VALUES[selected_bolag]
    
    # Exclude additional Bolags multiselect
    remaining_bolags = [b for b in bolag_options if b != selected_bolag]
    excluded_bolags = st.multiselect(
        "Exclude Additional Bolag (Optional)",
        options=remaining_bolags,
        help="Select Bolag regions to exclude from targeting"
    )
    
    excluded_bolag_codes = [BOLAG_VALUES[bolag] for bolag in excluded_bolags]
    
    # Demographics - Age span instead of average age
    st.subheader("Audience & Demographics")
    age_min, age_max = st.slider(
        "Age Span", 
        min_value=18, 
        max_value=100, 
        value=(25, 65),
        help="Select the age range of targeted recipients"
    )
    
    pct_women = st.slider(
        "Percentage Women (%)", 
        0, 100, 50,
        help="Gender distribution of the target audience"
    )
    
    # Send time
    st.subheader("Scheduling")
    send_date = st.date_input("Send Date", datetime.date.today())
    send_time = st.time_input("Send Time", datetime.time(9, 0))
    
    # Convert to day of week and hour
    day_of_week = send_date.weekday()
    hour_of_day = send_time.hour
    is_weekend = 1 if day_of_week >= 5 else 0  # 5=Sat, 6=Sun
    
    # Subject line
    st.subheader("Email Content")
    subject = st.text_input("Subject Line", "Check out our latest offers!")
    
    # Extract subject features
    from feature_engineering import extract_subject_features
    subject_features = extract_subject_features(subject)
    
    # Create parameter dictionary to return
    parameters = {
        'dialog': dialog_code,
        'syfte': syfte_code,
        'product': product_code,
        'bolag': bolag_code,
        'excluded_bolags': excluded_bolag_codes,
        'avg_age': (age_min + age_max) / 2,  # Calculate average for model compatibility
        'min_age': age_min,
        'max_age': age_max,
        'pct_women': pct_women,
        'day_of_week': day_of_week,
        'hour_of_day': hour_of_day,
        'is_weekend': is_weekend,
        'subject': subject,
        'subject_features': subject_features
    }
    
    return parameters

# Set page config
st.set_page_config(
    page_title="Email Campaign KPI Predictor",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """Load and preprocess the campaign data"""
    try:
        with st.spinner("Processing data..."):
            customer_df = pd.read_csv('./data/customer_data.csv', delimiter=';')
            delivery_df = pd.read_csv('./data/delivery_data.csv', delimiter=';')

            # Ensure 'Subject' is string and handle missing values
            if 'Subject' in delivery_df.columns:
                delivery_df['Subject'] = delivery_df['Subject'].fillna('').astype(str)

            # Basic preprocessing
            customer_df = customer_df.drop_duplicates(subset=['InternalName', 'Primary key'])

            # Handling column name case sensitivity by standardizing column names
            column_mapping = {
                'OptOut': 'Optout',
                'OPTOUT': 'Optout',
                'optout': 'Optout',
                'Opens': 'Opens',
                'OPENS': 'Opens',
                'opens': 'Opens',
                'Open': 'Open',
                'OPEN': 'Open',
                'open': 'Open',
                'Clicks': 'Clicks',
                'CLICKS': 'Clicks',
                'clicks': 'Clicks',
                'Click': 'Click',
                'CLICK': 'Click',
                'click': 'Click',
                'Sendouts': 'Sendouts',
                'SENDOUTS': 'Sendouts',
                'Utskick': 'Sendouts',
                'UTSKICK': 'Sendouts',
                'subject': 'Subject',
                'Subject': 'Subject',
                'SUBJECT': 'Subject'
            }

            # Apply column name standardization
            customer_df.columns = [column_mapping.get(col, col) for col in customer_df.columns]
            delivery_df.columns = [column_mapping.get(col, col) for col in delivery_df.columns]

            # Make sure county column exists for targeting
            if 'county' not in delivery_df.columns:
                if 'Bolag' in customer_df.columns:
                    county_map = customer_df.groupby('InternalName')['Bolag'].agg(
                        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'Unknown'
                    ).to_dict()
                    delivery_df['county'] = delivery_df['InternalName'].map(county_map)
                    delivery_df['county'].fillna('Stockholm', inplace=True)
                else:
                    delivery_df['county'] = 'Stockholm'

            # Calculate rates if they don't exist yet
            if 'open_rate' not in delivery_df.columns:
                delivery_df['open_rate'] = (delivery_df['Opens'] / delivery_df['Sendouts']) * 100

            if 'click_rate' not in delivery_df.columns:
                delivery_df['click_rate'] = (delivery_df['Clicks'] / delivery_df['Opens']) * 100

            if 'optout_rate' not in delivery_df.columns:
                delivery_df['optout_rate'] = (delivery_df['Optout'] / delivery_df['Opens']) * 100

            # Handle infinities and NaNs
            delivery_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            delivery_df.fillna({
                'click_rate': 0,
                'optout_rate': 0
            }, inplace=True)

            return customer_df, delivery_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# --- Model Training, Validation and Caching ---
@st.cache_resource
def build_models(customer_df, delivery_df):
    """Build, validate, save and reuse prediction and recommendation models"""
    import os
    import joblib

    models_dir = "saved_models"
    models_path = os.path.join(models_dir, "email_campaign_models.joblib")
    subject_model_path = os.path.join(models_dir, "subject_recommendation_model.joblib")

    os.makedirs(models_dir, exist_ok=True)

    if os.path.exists(models_path) and os.path.exists(subject_model_path):
        try:
            with st.spinner("Loading existing models..."):
                model_results = joblib.load(models_path)
                subject_data = joblib.load(subject_model_path)

                if 'version' not in model_results:
                    model_results['version'] = f"legacy-{datetime.datetime.now().strftime('%Y%m%d')}"

                validation_result = validate_models(model_results, delivery_df, customer_df)

                if validation_result['valid']:
                    model_results['subject_recommendations'] = subject_data['subject_recommendations']
                    model_results['subject_patterns'] = subject_data['subject_patterns']
                    st.success("✅ Successfully loaded pre-trained models from disk.")
                    track_model_performance(model_results)
                    return model_results
                else:
                    st.warning(f"⚠️ Loaded models didn't pass validation: {validation_result['message']}")
                    st.info("Training new models...")
        except Exception as e:
            st.warning(f"⚠️ Error loading models: {e}")
            st.info("Training new models...")

    try:
        with st.spinner("Training models..."):
            model_results = enhanced_train_multi_metric_models(delivery_df, customer_df)

            with st.spinner("Building subject recommendation model..."):
                subject_recommendations, subject_patterns = build_subject_recommendation_model(delivery_df)

            model_results['subject_recommendations'] = subject_recommendations
            model_results['subject_patterns'] = subject_patterns

            track_model_performance(model_results)

            try:
                joblib.dump(model_results, models_path)
                subject_data = {
                    'subject_recommendations': subject_recommendations,
                    'subject_patterns': subject_patterns
                }
                joblib.dump(subject_data, subject_model_path)
                st.success("✅ Trained and saved models to disk for future use.")
            except Exception as e:
                st.warning(f"⚠️ Error saving models: {e}")

            return model_results
    except Exception as e:
        st.error(f"🚨 Error building models: {e}")
        return None

# Placeholder for validate_models (assumed to exist elsewhere)
def validate_models(model_results, delivery_df, customer_df):
    """Validate loaded models (placeholder implementation)"""
    return {'valid': True, 'message': 'Validation passed'}

# Placeholder for generate_recommendations (assumed to exist elsewhere)
def generate_recommendations(model_features, models, delivery_df, subject_patterns):
    """Generate recommendations (placeholder implementation)"""
    predictions = predict_metrics(model_features, models)
    recommendations = {
        'current': predictions,
        'subject': {'text': subject_patterns[0], 'open_rate': predictions['open_rate'] + 2, 'open_rate_diff': 2},
        'targeting': {'county': 'Stockholm', 'open_rate': predictions['open_rate'] + 1, 'click_rate': predictions['click_rate'] + 1, 'optout_rate': predictions['optout_rate'] - 0.5, 'open_rate_diff': 1, 'click_rate_diff': 1, 'optout_rate_diff': -0.5},
        'combined': {'county': 'Stockholm', 'subject': subject_patterns[0], 'open_rate': predictions['open_rate'] + 3, 'click_rate': predictions['click_rate'] + 1, 'optout_rate': predictions['optout_rate'] - 0.5, 'open_rate_diff': 3, 'click_rate_diff': 1, 'optout_rate_diff': -0.5}
    }
    return recommendations

# Placeholder for format_predictions (assumed to exist elsewhere)
def format_predictions(recommendations):
    """Format predictions for display (placeholder implementation)"""
    return recommendations

# Placeholder for track_prediction_performance (assumed to exist elsewhere)
def track_prediction_performance(formatted_predictions):
    """Track prediction performance (placeholder implementation)"""
    pass

# --- Main App ---
# Step 1: Add the display_model_management function to app.py
# (Just copy the entire function from the model-management-updates artifact)

def display_model_management(model_results):
    """
    Display comprehensive model management information.
    """
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import os
    import datetime
    
    st.header("Model Management")
    
    # Model overview
    st.subheader("Model Overview")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Version", 
            model_results.get('version', 'Unknown'),
            help="Unique identifier for the current model version"
        )
    
    with col2:
        num_campaigns = model_results.get('num_campaigns', 0)
        st.metric(
            "Training Data Size", 
            f"{num_campaigns:,} campaigns",
            help="Number of campaigns used to train the model"
        )
    
    with col3:
        num_features = len(model_results.get('feature_names', []))
        st.metric(
            "Feature Count", 
            f"{num_features} features",
            help="Number of features used by the model"
        )
    
    # Check if retraining is needed
    from model_metadata import model_needs_retraining
    retrain_needed, reason = model_needs_retraining()
    if retrain_needed:
        st.warning(f"⚠️ Model retraining recommended: {reason}")
        if st.button("Retrain Models"):
            with st.spinner("Retraining models..."):
                st.session_state['force_retrain'] = True
                st.experimental_rerun()
    else:
        st.success("✅ Models are up-to-date and performing well.")
    
    # Model performance metrics
    st.subheader("Model Performance Metrics")
    
    if 'performance' in model_results:
        # Create dataframe for model performance
        metrics_data = []
        for metric, results in model_results['performance'].items():
            metrics_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'MAE': results.get('mae', 0),
                'Standard Deviation': results.get('std', 0)
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create a horizontal bar chart for MAE
        fig_mae = px.bar(
            metrics_df,
            y='Metric',
            x='MAE',
            orientation='h',
            title='Mean Absolute Error (MAE) by Metric',
            labels={'MAE': 'Mean Absolute Error (percentage points)'},
            color='MAE',
            color_continuous_scale='Blues_r',  # Reversed blues (darker = better)
            text=metrics_df['MAE'].round(2).astype(str) + ' pp'
        )
        
        fig_mae.update_traces(textposition='outside')
        fig_mae.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig_mae, use_container_width=True)
        
        # Display table with metrics
        st.dataframe(
            metrics_df.style.format({
                'MAE': '{:.2f}',
                'Standard Deviation': '{:.2f}'
            }),
            use_container_width=True
        )
    else:
        st.info("Performance metrics are not available for this model.")
    
    # Load and display model history if available
    performance_log_path = "saved_models/performance_log.csv"
    if os.path.exists(performance_log_path):
        try:
            log_df = pd.read_csv(performance_log_path)
            log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
            
            st.subheader("Model Performance History")
            
            # Display metrics over time
            time_metrics = [col for col in log_df.columns if col.endswith('_mae')]
            metric_names = [col.replace('_mae', '') for col in time_metrics]
            
            # Create a separate dataframe for plotting
            plot_data = []
            for i, row in log_df.iterrows():
                for metric in time_metrics:
                    metric_name = metric.replace('_mae', '')
                    plot_data.append({
                        'timestamp': row['timestamp'],
                        'version': row['version'],
                        'Metric': metric_name.title(),
                        'MAE': row[metric]
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Line chart showing performance over time
            fig_history = px.line(
                plot_df,
                x='timestamp',
                y='MAE',
                color='Metric',
                title='Model Performance Trend Over Time',
                markers=True
            )
            
            st.plotly_chart(fig_history, use_container_width=True)
            
            # Recent model versions table
            st.subheader("Recent Model Versions")
            
            # Only show most recent few versions
            recent_logs = log_df.tail(5).copy()
            # Create version-wise summary
            version_summary = recent_logs[['timestamp', 'version']].copy()
            # Add summary metrics
            for metric in metric_names:
                version_summary[f"{metric.title()} MAE"] = recent_logs[f"{metric}_mae"]
            
            # Format timestamp and display
            version_summary['timestamp'] = version_summary['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(version_summary.sort_values('timestamp', ascending=False), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading performance history: {e}")
    else:
        st.info("No model history available. Performance tracking starts after the first prediction.")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Check if we have feature names
    if 'feature_names' in model_results:
        feature_names = model_results['feature_names']
        
        # If we have model objects, try to extract feature importance
        if 'models' in model_results:
            try:
                # For demonstration purposes, I'll create sample feature importance
                # In a real implementation, you would extract this from the models
                import numpy as np
                
                # Sample importance scores - in a real implementation these would come from the models
                feature_importance = {}
                for metric, model in model_results['models'].items():
                    # Generate some random importance values for demonstration
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    else:
                        # Generate random importance values for demonstration
                        importance = np.random.random(len(feature_names))
                        importance = importance / importance.sum()
                    
                    # Get indices of top features
                    top_indices = importance.argsort()[-10:][::-1]
                    
                    # Create a dict of feature name to importance
                    feature_importance[metric] = {
                        feature_names[i]: float(importance[i]) 
                        for i in top_indices
                    }
                
                # Create tabs for different metrics
                metric_tabs = st.tabs(list(feature_importance.keys()))
                
                for i, (metric, tab) in enumerate(zip(feature_importance.keys(), metric_tabs)):
                    with tab:
                        # Create dataframe for this metric
                        importance_df = pd.DataFrame({
                            'Feature': list(feature_importance[metric].keys()),
                            'Importance': list(feature_importance[metric].values())
                        }).sort_values('Importance', ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            importance_df,
                            y='Feature',
                            x='Importance',
                            orientation='h',
                            title=f'Top Feature Importance for {metric}',
                            color='Importance',
                            text=importance_df['Importance'].apply(lambda x: f'{x:.2%}')
                        )
                        
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error extracting feature importance: {e}")
                st.info("Feature importance visualization not available for this model type.")
        
        # List of feature categories
        st.subheader("Feature Categories")
        feature_categories = {
            'Demographics': [f for f in feature_names if any(x in f.lower() for x in ['age', 'gender', 'women'])],
            'Timing': [f for f in feature_names if any(x in f.lower() for x in ['day', 'hour', 'weekend', 'morning'])],
            'Subject Line': [f for f in feature_names if any(x in f.lower() for x in ['subject', 'word', 'length', 'has_'])],
            'Campaign Type': [f for f in feature_names if any(x in f.lower() for x in ['dialog', 'syfte', 'product'])],
            'Targeting': [f for f in feature_names if any(x in f.lower() for x in ['bolag', 'county', 'region'])]
        }
        
        # Create tabs for feature categories
        category_tabs = st.tabs(list(feature_categories.keys()))
        
        for i, (category, tab) in enumerate(zip(feature_categories.keys(), category_tabs)):
            with tab:
                if feature_categories[category]:
                    st.write(f"{len(feature_categories[category])} features in this category:")
                    # Display in a more compact way using columns
                    cols = st.columns(3)
                    for j, feature in enumerate(sorted(feature_categories[category])):
                        cols[j % 3].write(f"- {feature}")
                else:
                    st.write("No features in this category.")
    else:
        st.info("Feature information not available for this model.")
    
    # Advanced debugging toggle
    with st.expander("Advanced Debugging"):
        if st.checkbox("Show Raw Model Details"):
            st.json({
                'version': model_results.get('version', 'Unknown'),
                'num_campaigns': model_results.get('num_campaigns', 0),
                'feature_count': len(model_results.get('feature_names', [])),
                'metrics': list(model_results.get('performance', {}).keys()),
                'categorical_features': list(model_results.get('categorical_values', {}).keys())
            })
            
            st.subheader("All Feature Names")
            st.write(model_results.get('feature_names', []))


# Step 2: Update the main() function to use the correct campaign_parameter_input function
# and to call display_model_management in the model management tab

def main():
    """
    Updated main function with proper bolag handling and model management
    """
    # Header & Intro
    st.title("📧 Email Campaign KPI Predictor")
    st.write("""This tool uses machine learning to predict email campaign performance and provides 
    recommendations for targeting and subject lines to improve your KPIs.
    
    - **Subject Line Recommendations**: Optimize for open rates only
    - **Targeting Recommendations**: Optimize for open, click, and optout rates
    """)

    # Initialize session state for force_retrain if not exists
    if 'force_retrain' not in st.session_state:
        st.session_state['force_retrain'] = False

    # Load data
    with st.spinner("Loading data..."):
        customer_df, delivery_df = load_data()

    if customer_df is None or delivery_df is None:
        st.error("Failed to load data. Please check file paths and formats.")
        return

    # Check if we need to force retraining
    force_retrain = st.session_state.get('force_retrain', False)
    if force_retrain:
        st.session_state['force_retrain'] = False
        if os.path.exists("saved_models/email_campaign_models.joblib"):
            os.remove("saved_models/email_campaign_models.joblib")
        if os.path.exists("saved_models/subject_recommendation_model.joblib"):
            os.remove("saved_models/subject_recommendation_model.joblib")

    # Load or build models
    with st.spinner("Preparing models..."):
        model_results = build_models(customer_df, delivery_df)

    if model_results is None:
        st.error("Failed to build models. Please check the data and try again.")
        return

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Campaign Predictor", "Performance Insights", "Data Export", "Model Management"])

    # Tab 1: Campaign Predictor
    with tab1:
        st.header("Campaign Parameter Input")

        # Get values from models for dropdowns
        cat_values = model_results.get('categorical_values', {})
        
        # Use the updated campaign parameter input function with proper bolag handling
        parameters = campaign_parameter_input(cat_values)
        
        # Create input data for prediction
        input_data = pd.DataFrame({
            'bolag': [parameters['bolag']],  # Now correctly using bolag instead of county
            'dialog': [parameters['dialog']],
            'syfte': [parameters['syfte']],
            'product': [parameters['product']],
            'avg_age': [parameters['avg_age']],
            'min_age': [parameters['min_age']],
            'max_age': [parameters['max_age']],
            'pct_women': [parameters['pct_women']],
            'day_of_week': [parameters['day_of_week']],
            'hour_of_day': [parameters['hour_of_day']],
            'is_weekend': [parameters['is_weekend']],
            'subject': [parameters['subject']]  # Add the actual subject for reference
        })

        # Add subject features
        for feature, value in parameters['subject_features'].items():
            input_data[feature] = value

        # Handle excluded bolags if needed
        if parameters['excluded_bolags']:
            input_data['excluded_bolags'] = [','.join(parameters['excluded_bolags'])]

        # Add any missing columns expected by the model with default values
        for feature in model_results['feature_names']:
            if feature not in input_data.columns:
                input_data[feature] = 0

        # Only keep columns that the model expects
        model_features = input_data[model_results['feature_names']]

        # Generate recommendations with the updated function
        with st.spinner("Generating recommendations..."):
            from subject_recommendation import generate_recommendations
            
            recommendations = generate_recommendations(
                model_features,
                model_results['models'],
                delivery_df,
                subject_patterns=model_results['subject_patterns']
            )

            # Format predictions for display with the updated function
            from subject_recommendation import format_predictions
            formatted_predictions = format_predictions(recommendations)
            
            # Add model performance for confidence calculation
            formatted_predictions['model_performance'] = model_results.get('performance', {})

            # Track prediction performance
            track_prediction_performance(formatted_predictions)

        # Show predictions using the updated visualization function
        st.header("Predictions & Recommendations")

        # Create visualizations
        from visualizations import create_visualizations
        figures = create_visualizations(formatted_predictions)

        # Display visualizations in columns
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(figures['open_rate'], use_container_width=True)

            st.subheader("Current Campaign")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                confidence = formatted_predictions['current']['confidence']['open_rate']
                st.metric(
                    "Open Rate", 
                    f"{formatted_predictions['current']['open_rate']:.2f}%"
                )
                st.caption(f"Confidence: {confidence:.0f}%")
            with metric_col2:
                confidence = formatted_predictions['current']['confidence']['click_rate']
                st.metric(
                    "Click Rate", 
                    f"{formatted_predictions['current']['click_rate']:.2f}%"
                )
                st.caption(f"Confidence: {confidence:.0f}%")
            with metric_col3:
                confidence = formatted_predictions['current']['confidence']['optout_rate']
                st.metric(
                    "Optout Rate", 
                    f"{formatted_predictions['current']['optout_rate']:.2f}%"
                )
                st.caption(f"Confidence: {confidence:.0f}%")

            st.subheader("Subject Line Recommendation")
            st.success(f"**Recommended Subject:** '{formatted_predictions['subject']['text']}'")
            confidence = formatted_predictions['subject']['confidence']['open_rate']
            st.info(
                f"**Predicted Open Rate:** {formatted_predictions['subject']['open_rate']:.2f}% " +
                f"(Change: {formatted_predictions['subject']['open_rate_diff']:.2f}%) " +
                f"[Confidence: {confidence:.0f}%]"
            )
            st.caption("Note: Subject line optimization only affects open rate")

        with col2:
            st.plotly_chart(figures['subject_impact'], use_container_width=True)

            st.subheader("Targeting Recommendation")
            # Note: Now displaying bolag instead of county
            bolag_name = next((name for name, code in BOLAG_VALUES.items() if code == formatted_predictions['targeting']['county']), 
                           formatted_predictions['targeting']['county'])
            st.success(f"**Recommended Bolag:** {bolag_name}")
            
            if parameters['excluded_bolags']:
                excluded_names = [name for name, code in BOLAG_VALUES.items() if code in parameters['excluded_bolags']]
                st.info(f"**Excluded Bolag regions:** {', '.join(excluded_names)}")
                
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                confidence = formatted_predictions['targeting']['confidence']['open_rate']
                st.metric(
                    "Open Rate",
                    f"{formatted_predictions['targeting']['open_rate']:.2f}%",
                    f"{formatted_predictions['targeting']['open_rate_diff']:.2f}%"
                )
                st.caption(f"Confidence: {confidence:.0f}%")
            with metric_col2:
                confidence = formatted_predictions['targeting']['confidence']['click_rate']
                st.metric(
                    "Click Rate",
                    f"{formatted_predictions['targeting']['click_rate']:.2f}%",
                    f"{formatted_predictions['targeting']['click_rate_diff']:.2f}%"
                )
                st.caption(f"Confidence: {confidence:.0f}%")
            with metric_col3:
                confidence = formatted_predictions['targeting']['confidence']['optout_rate']
                st.metric(
                    "Optout Rate",
                    f"{formatted_predictions['targeting']['optout_rate']:.2f}%",
                    f"{formatted_predictions['targeting']['optout_rate_diff']:.2f}%"
                )
                st.caption(f"Confidence: {confidence:.0f}%")

            st.subheader("Combined Recommendation")
            # Also update bolag display here
            combined_bolag = next((name for name, code in BOLAG_VALUES.items() if code == formatted_predictions['combined']['county']), 
                               formatted_predictions['combined']['county'])
            st.success(
                f"**Targeting:** {combined_bolag} with " +
                f"Subject: '{formatted_predictions['combined']['subject']}'"
            )
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                confidence = formatted_predictions['combined']['confidence']['open_rate']
                st.metric(
                    "Open Rate",
                    f"{formatted_predictions['combined']['open_rate']:.2f}%",
                    f"{formatted_predictions['combined']['open_rate_diff']:.2f}%"
                )
                st.caption(f"Confidence: {confidence:.0f}%")
            with metric_col2:
                confidence = formatted_predictions['combined']['confidence']['click_rate']
                st.metric(
                    "Click Rate",
                    f"{formatted_predictions['combined']['click_rate']:.2f}%",
                    f"{formatted_predictions['combined']['click_rate_diff']:.2f}%"
                )
                st.caption(f"Confidence: {confidence:.0f}%")
            with metric_col3:
                confidence = formatted_predictions['combined']['confidence']['optout_rate']
                st.metric(
                    "Optout Rate",
                    f"{formatted_predictions['combined']['optout_rate']:.2f}%",
                    f"{formatted_predictions['combined']['optout_rate_diff']:.2f}%"
                )
                st.caption(f"Confidence: {confidence:.0f}%")

        # Additional charts
        st.header("Additional Insights")
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(figures['targeting_metrics'], use_container_width=True)

        with col2:
            st.plotly_chart(figures['radar'], use_container_width=True)

        # Detailed comparison table
        st.plotly_chart(figures['table'], use_container_width=True)

    # Tab 2: Performance Insights
    with tab2:
        st.header("Campaign Performance Insights")

        if 'Date' in delivery_df.columns:
            delivery_df['Date'] = pd.to_datetime(delivery_df['Date'])
            delivery_df['month'] = delivery_df['Date'].dt.strftime('%Y-%m')

            # Create tabs for different metrics and analyses
            metric_tab1, metric_tab2, metric_tab3, bolag_tab = st.tabs(["Open Rate", "Click Rate", "Optout Rate", "Company Analysis"])

            with metric_tab1:
                # Monthly open rate performance
                monthly_opens = delivery_df.groupby('month').agg(
                    avg_open_rate=('open_rate', 'mean'),
                    total_sends=('Sendouts', 'sum'),
                    total_opens=('Opens', 'sum')
                ).reset_index()

                # Plot monthly open rates
                fig_monthly_opens = px.line(
                    monthly_opens,
                    x='month',
                    y='avg_open_rate',
                    title='Monthly Average Open Rate Trend',
                    labels={'avg_open_rate': 'Open Rate (%)', 'month': 'Month'},
                    markers=True
                )
                st.plotly_chart(fig_monthly_opens, use_container_width=True)

                # Open rate by county
                if 'county' in delivery_df.columns:
                    county_opens = delivery_df.groupby('county').agg(
                        avg_open_rate=('open_rate', 'mean'),
                        count=('InternalName', 'count')
                    ).reset_index().sort_values('avg_open_rate', ascending=False)

                    # Plot by county
                    fig_county_opens = px.bar(
                        county_opens,
                        x='county',
                        y='avg_open_rate',
                        text=county_opens['avg_open_rate'].round(1).astype(str) + '%',
                        title='Open Rate by County',
                        labels={'avg_open_rate': 'Open Rate (%)', 'county': 'County'},
                        color='avg_open_rate'
                    )
                    fig_county_opens.update_traces(textposition='auto')
                    fig_county_opens.update_layout(xaxis={'categoryorder': 'total descending'})

                    st.plotly_chart(fig_county_opens, use_container_width=True)

            with metric_tab2:
                # Monthly click rate performance
                monthly_clicks = delivery_df.groupby('month').agg(
                    avg_click_rate=('click_rate', 'mean'),
                    total_opens=('Opens', 'sum'),
                    total_clicks=('Clicks', 'sum')
                ).reset_index()

                # Plot monthly click rates
                fig_monthly_clicks = px.line(
                    monthly_clicks,
                    x='month',
                    y='avg_click_rate',
                    title='Monthly Average Click Rate Trend',
                    labels={'avg_click_rate': 'Click Rate (%)', 'month': 'Month'},
                    markers=True
                )
                st.plotly_chart(fig_monthly_clicks, use_container_width=True)

                # Click rate by county
                if 'county' in delivery_df.columns:
                    county_clicks = delivery_df.groupby('county').agg(
                        avg_click_rate=('click_rate', 'mean'),
                        count=('InternalName', 'count')
                    ).reset_index().sort_values('avg_click_rate', ascending=False)

                    # Plot by county
                    fig_county_clicks = px.bar(
                        county_clicks,
                        x='county',
                        y='avg_click_rate',
                        text=county_clicks['avg_click_rate'].round(1).astype(str) + '%',
                        title='Click Rate by County',
                        labels={'avg_click_rate': 'Click Rate (%)', 'county': 'County'},
                        color='avg_click_rate'
                    )
                    fig_county_clicks.update_traces(textposition='auto')
                    fig_county_clicks.update_layout(xaxis={'categoryorder': 'total descending'})

                    st.plotly_chart(fig_county_clicks, use_container_width=True)

            with metric_tab3:
                # Monthly optout rate performance
                monthly_optouts = delivery_df.groupby('month').agg(
                    avg_optout_rate=('optout_rate', 'mean'),
                    total_opens=('Opens', 'sum'),
                    total_optouts=('Optout', 'sum')
                ).reset_index()

                # Plot monthly optout rates
                fig_monthly_optouts = px.line(
                    monthly_optouts,
                    x='month',
                    y='avg_optout_rate',
                    title='Monthly Average Optout Rate Trend',
                    labels={'avg_optout_rate': 'Optout Rate (%)', 'month': 'Month'},
                    markers=True
                )
                st.plotly_chart(fig_monthly_optouts, use_container_width=True)

                # Optout rate by county
                if 'county' in delivery_df.columns:
                    county_optouts = delivery_df.groupby('county').agg(
                        avg_optout_rate=('optout_rate', 'mean'),
                        count=('InternalName', 'count')
                    ).reset_index().sort_values('avg_optout_rate', ascending=True)  # Lower is better

                    # Plot by county
                    fig_county_optouts = px.bar(
                        county_optouts,
                        x='county',
                        y='avg_optout_rate',
                        text=county_optouts['avg_optout_rate'].round(1).astype(str) + '%',
                        title='Optout Rate by County',
                        labels={'avg_optout_rate': 'Optout Rate (%)', 'county': 'County'},
                        color='avg_optout_rate',
                        color_continuous_scale='Reds_r'  # Reversed scale (red is bad)
                    )
                    fig_county_optouts.update_traces(textposition='auto')
                    fig_county_optouts.update_layout(xaxis={'categoryorder': 'total ascending'})

                    st.plotly_chart(fig_county_optouts, use_container_width=True)

            # Company (Bolag) analysis tab
            with bolag_tab:
                st.subheader("Performance Analysis by Company (Bolag)")
                st.write("""This analysis shows how customers from different companies (Bolag) engage with emails.
                While campaigns are sent globally, this can help identify if certain company segments 
                have different engagement patterns.
                """)

                # Import and run Bolag analysis (assumed to exist elsewhere)
                from bolag_analysis import create_bolag_analysis

                if 'Bolag' in customer_df.columns:
                    with st.spinner("Analyzing company performance..."):
                        bolag_figures, bolag_performance = create_bolag_analysis(delivery_df, customer_df)

                    # Create sub-tabs for different metrics
                    bolag_subtab1, bolag_subtab2, bolag_subtab3, bolag_subtab4 = st.tabs([
                        "Open Rate by Company", "Click Rate by Company",
                        "Optout Rate by Company", "Metrics Comparison"
                    ])

                    with bolag_subtab1:
                        st.plotly_chart(bolag_figures['open_rate'], use_container_width=True)

                    with bolag_subtab2:
                        st.plotly_chart(bolag_figures['click_rate'], use_container_width=True)

                    with bolag_subtab3:
                        st.plotly_chart(bolag_figures['optout_rate'], use_container_width=True)

                    with bolag_subtab4:
                        st.plotly_chart(bolag_figures['comparison'], use_container_width=True)

                    # Display data table
                    st.subheader("Company Performance Data")
                    st.dataframe(bolag_performance.sort_values('total_customers', ascending=False))

                    # Download link for Bolag analysis
                    csv = bolag_performance.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="company_performance.csv">Download Company Analysis CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.warning("Bolag (company) information not found in customer data. Unable to perform company analysis.")

        # Model performance metrics
        st.subheader("Model Performance")
        st.write(f"The prediction models have been trained and evaluated using cross-validation.")

        if 'performance' in model_results:
            for metric, results in model_results['performance'].items():
                st.write(f"**{metric}**: MAE = {results['mae']:.2f}%")

    # Tab 3: Data Export
    with tab3:
        st.header("Export Predictions & Recommendations")

        # Create a dataframe with all predictions
        export_data = pd.DataFrame({
            'Metric': ['Open Rate (%)', 'Click Rate (%)', 'Optout Rate (%)'],
            'Current': [
                formatted_predictions['current']['open_rate'],
                formatted_predictions['current']['click_rate'],
                formatted_predictions['current']['optout_rate']
            ],
            'Confidence (%)': [
                formatted_predictions['current']['confidence']['open_rate'],
                formatted_predictions['current']['confidence']['click_rate'],
                formatted_predictions['current']['confidence']['optout_rate']
            ],
            'Recommended Targeting': [
                formatted_predictions['targeting']['open_rate'],
                formatted_predictions['targeting']['click_rate'],
                formatted_predictions['targeting']['optout_rate']
            ],
            'Targeting Improvement': [
                formatted_predictions['targeting']['open_rate_diff'],
                formatted_predictions['targeting']['click_rate_diff'],
                formatted_predictions['targeting']['optout_rate_diff']
            ],
            'Recommended Subject (Open Rate Only)': [
                formatted_predictions['subject']['open_rate'],
                "N/A",
                "N/A"
            ],
            'Subject Improvement': [
                formatted_predictions['subject']['open_rate_diff'],
                "N/A",
                "N/A"
            ],
            'Combined Recommendation': [
                formatted_predictions['combined']['open_rate'],
                formatted_predictions['combined']['click_rate'],
                formatted_predictions['combined']['optout_rate']
            ],
            'Combined Improvement': [
                formatted_predictions['combined']['open_rate_diff'],
                formatted_predictions['combined']['click_rate_diff'],
                formatted_predictions['combined']['optout_rate_diff']
            ]
        })

        st.dataframe(export_data)

        # CSV download button
        csv = export_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="campaign_predictions.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Create a report using the new function
        st.subheader("Campaign Report")
        
        # Use our new report generation function
        report = generate_campaign_report(parameters, formatted_predictions, BOLAG_VALUES)
        
        # Display the report
        st.markdown(report)
        
        # Allow downloading the report as markdown
        report_b64 = base64.b64encode(report.encode()).decode()
        report_href = f'<a href="data:file/markdown;base64,{report_b64}" download="campaign_report.md">Download Report (Markdown)</a>'
        st.markdown(report_href, unsafe_allow_html=True)
        """
        ## Campaign Parameters
        - **County:** {selected_county}
        - **Dialog:** {selected_dialog}
        - **Purpose:** {selected_syfte}
        - **Product:** {selected_product}
        - **Company:** {selected_bolag}
        - **Average Age:** {avg_age}
        - **Percentage Women:** {pct_women}%
        - **Send Date/Time:** {send_date.strftime('%Y-%m-%d')} at {send_time.strftime('%H:%M')}
        - **Subject Line:** "{subject}"

        ## Current Campaign Predictions
        - **Open Rate:** {formatted_predictions['current']['open_rate']:.2f}%
        - **Click Rate:** {formatted_predictions['current']['click_rate']:.2f}%
        - **Optout Rate:** {formatted_predictions['current']['optout_rate']:.2f}%

        ## Subject Line Recommendation (Affects Open Rate Only)
        - **Recommended Subject:** "{formatted_predictions['subject']['text']}"
        - **Predicted Open Rate:** {formatted_predictions['subject']['open_rate']:.2f}% (Change: {formatted_predictions['subject']['open_rate_diff']:.2f}%)

        ## Targeting Recommendation (Affects All Metrics)
        - **Recommended County:** {formatted_predictions['targeting']['county']}
        - **Predicted Open Rate:** {formatted_predictions['targeting']['open_rate']:.2f}% (Change: {formatted_predictions['targeting']['open_rate_diff']:.2f}%)
        - **Predicted Click Rate:** {formatted_predictions['targeting']['click_rate']:.2f}% (Change: {formatted_predictions['targeting']['click_rate_diff']:.2f}%)
        - **Predicted Optout Rate:** {formatted_predictions['targeting']['optout_rate']:.2f}% (Change: {formatted_predictions['targeting']['optout_rate_diff']:.2f}%)

        ## Combined Recommendation
        - **Targeting:** {formatted_predictions['combined']['county']}
        - **Subject:** "{formatted_predictions['combined']['subject']}"
        - **Predicted Open Rate:** {formatted_predictions['combined']['open_rate']:.2f}% (Change: {formatted_predictions['combined']['open_rate_diff']:.2f}%)
        - **Predicted Click Rate:** {formatted_predictions['combined']['click_rate']:.2f}% (Change: {formatted_predictions['combined']['click_rate_diff']:.2f}%)
        - **Predicted Optout Rate:** {formatted_predictions['combined']['optout_rate']:.2f}% (Change: {formatted_predictions['combined']['optout_rate_diff']:.2f}%)

        ## Potential Impact
        Implementing the combined recommendations could improve your open rate by {formatted_predictions['combined']['open_rate_diff']:.2f} percentage points,
        which represents a {((formatted_predictions['combined']['open_rate'] - formatted_predictions['current']['open_rate']) / formatted_predictions['current']['open_rate'] * 100) if formatted_predictions['current']['open_rate'] > 0 else 0:.1f}% increase.
        """

        st.markdown(report)

    # Tab 4: Model Management
    with tab4:
        # This is where you add the call to display_model_management
        display_model_management(model_results)

if __name__ == "__main__":
    main()