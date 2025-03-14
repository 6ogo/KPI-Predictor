import streamlit as st
import pandas as pd
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
    """Load and preprocess the campaign data with improved error handling"""
    try:
        with st.spinner("Processing data..."):
            # Import necessary libraries
            import numpy as np
            import pandas as pd
            import logging
            
            # Load data files with explicit error handling
            try:
                customer_df = pd.read_csv('./data/customer_data.csv', delimiter=';')
                st.success(f"Successfully loaded customer_data.csv with {len(customer_df)} rows")
            except Exception as e:
                st.warning(f"Error loading customer data: {str(e)}. Creating empty dataframe.")
                # Create empty dataframe with required columns
                customer_df = pd.DataFrame(columns=['Primary key', 'InternalName', 'OptOut', 'Open', 'Click', 'Gender', 'Age', 'Bolag'])
            
            try:
                delivery_df = pd.read_csv('./data/delivery_data.csv', delimiter=';')
                st.success(f"Successfully loaded delivery_data.csv with {len(delivery_df)} rows")
            except Exception as e:
                st.warning(f"Error loading delivery data: {str(e)}. Creating empty dataframe.")
                # Create empty dataframe with required columns
                delivery_df = pd.DataFrame(columns=['InternalName', 'Subject', 'Date', 'Sendouts', 'Opens', 'Clicks', 'Optouts', 'Dialog', 'Syfte', 'Produkt'])
            
            # Print column names for debugging
            logging.info(f"Delivery columns: {delivery_df.columns.tolist()}")
            logging.info(f"Customer columns: {customer_df.columns.tolist()}")
            
            # Ensure 'Subject' is string and handle missing values
            if 'Subject' in delivery_df.columns:
                delivery_df['Subject'] = delivery_df['Subject'].fillna('').astype(str)
            
            # Basic preprocessing
            if len(customer_df) > 0:
                customer_df = customer_df.drop_duplicates(subset=['InternalName', 'Primary key'])
            
            # Ensure numeric columns are properly converted in delivery data
            numeric_cols = ['Sendouts', 'Opens', 'Clicks', 'Optouts']
            for col in numeric_cols:
                if col in delivery_df.columns:
                    # Convert to numeric with errors coerced
                    delivery_df[col] = pd.to_numeric(delivery_df[col], errors='coerce')
                    # Fill NaN values with 0
                    delivery_df[col] = delivery_df[col].fillna(0).astype(float)
                    # Log column stats for debugging
                    logging.info(f"{col} stats: min={delivery_df[col].min()}, max={delivery_df[col].max()}, mean={delivery_df[col].mean()}")
            
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
            if len(customer_df) > 0:
                customer_df.columns = [column_mapping.get(col, col) for col in customer_df.columns]
            
            if len(delivery_df) > 0:
                delivery_df.columns = [column_mapping.get(col, col) for col in delivery_df.columns]
            
            # Make sure county column exists for targeting
            if 'county' not in delivery_df.columns:
                if 'Bolag' in customer_df.columns and len(customer_df) > 0:
                    try:
                        # Calculate most common Bolag per delivery
                        county_map = customer_df.groupby('InternalName')['Bolag'].agg(
                            lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'Unknown'
                        ).to_dict()
                        
                        delivery_df['county'] = delivery_df['InternalName'].map(county_map)
                        delivery_df['county'].fillna('Stockholm', inplace=True)
                    except Exception as e:
                        logging.error(f"Error mapping Bolag to county: {str(e)}")
                        delivery_df['county'] = 'Stockholm'
                else:
                    delivery_df['county'] = 'Stockholm'
            
            # Enforce logical constraints on the data (Opens ≤ Sendouts, Clicks ≤ Opens, Optouts ≤ Opens)
            logging.info("Enforcing logical constraints on data...")

            # Check and fix: Opens should not exceed Sendouts
            if all(col in delivery_df.columns for col in ['Opens', 'Sendouts']):
                invalid_opens = delivery_df['Opens'] > delivery_df['Sendouts']
                if invalid_opens.any():
                    logging.warning(f"Found {invalid_opens.sum()} rows where Opens > Sendouts. Fixing...")
                    # Set Opens equal to Sendouts where the constraint is violated
                    delivery_df.loc[invalid_opens, 'Opens'] = delivery_df.loc[invalid_opens, 'Sendouts']

            # Check and fix: Clicks should not exceed Opens
            if all(col in delivery_df.columns for col in ['Clicks', 'Opens']):
                invalid_clicks = delivery_df['Clicks'] > delivery_df['Opens']
                if invalid_clicks.any():
                    logging.warning(f"Found {invalid_clicks.sum()} rows where Clicks > Opens. Fixing...")
                    # Set Clicks equal to Opens where the constraint is violated
                    delivery_df.loc[invalid_clicks, 'Clicks'] = delivery_df.loc[invalid_clicks, 'Opens']

            # Check and fix: Optouts should not exceed Opens
            if all(col in delivery_df.columns for col in ['Optouts', 'Opens']):
                invalid_optouts = delivery_df['Optouts'] > delivery_df['Opens']
                if invalid_optouts.any():
                    logging.warning(f"Found {invalid_optouts.sum()} rows where Optouts > Opens. Fixing...")
                    # Set Optouts equal to Opens where the constraint is violated
                    delivery_df.loc[invalid_optouts, 'Optouts'] = delivery_df.loc[invalid_optouts, 'Opens']

            # Calculate rates safely with numpy to avoid division by zero issues
            if 'Sendouts' in delivery_df.columns and 'Opens' in delivery_df.columns:
                # Calculate open rate: (Opens / Sendouts) * 100
                delivery_df['open_rate'] = np.where(
                    delivery_df['Sendouts'] > 0,
                    (delivery_df['Opens'] / delivery_df['Sendouts']) * 100,
                    0  # Default when sendouts is 0
                )
                logging.info(f"Calculated open_rate: min={delivery_df['open_rate'].min()}, max={delivery_df['open_rate'].max()}, mean={delivery_df['open_rate'].mean()}")
            else:
                delivery_df['open_rate'] = 0
                logging.warning("Could not calculate open_rate, missing required columns")
            
            if 'Opens' in delivery_df.columns and 'Clicks' in delivery_df.columns:
                # Calculate click rate: (Clicks / Opens) * 100
                delivery_df['click_rate'] = np.where(
                    delivery_df['Opens'] > 0,
                    (delivery_df['Clicks'] / delivery_df['Opens']) * 100,
                    0  # Default when opens is 0
                )
                logging.info(f"Calculated click_rate: min={delivery_df['click_rate'].min()}, max={delivery_df['click_rate'].max()}, mean={delivery_df['click_rate'].mean()}")
            else:
                delivery_df['click_rate'] = 0
                logging.warning("Could not calculate click_rate, missing required columns")
            
            if 'Opens' in delivery_df.columns and 'Optouts' in delivery_df.columns:
                # Calculate optout rate: (Optouts / Opens) * 100
                delivery_df['optout_rate'] = np.where(
                    delivery_df['Opens'] > 0,
                    (delivery_df['Optouts'] / delivery_df['Opens']) * 100,
                    0  # Default when opens is 0
                )
                logging.info(f"Calculated optout_rate: min={delivery_df['optout_rate'].min()}, max={delivery_df['optout_rate'].max()}, mean={delivery_df['optout_rate'].mean()}")
            else:
                delivery_df['optout_rate'] = 0
                logging.warning("Could not calculate optout_rate, missing required columns")
            
            # Handle infinities and NaNs
            delivery_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Fill NaNs with reasonable defaults
            delivery_df.fillna({
                'open_rate': 0,
                'click_rate': 0,
                'optout_rate': 0
            }, inplace=True)
            
            # Cap extreme values if needed (sometimes rates can be >100% due to data issues)
            delivery_df['open_rate'] = delivery_df['open_rate'].clip(0, 100)
            delivery_df['click_rate'] = delivery_df['click_rate'].clip(0, 100)
            delivery_df['optout_rate'] = delivery_df['optout_rate'].clip(0, 100)
            
            return customer_df, delivery_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        logging.error(f"Detailed error: {traceback.format_exc()}")
        return None, None

def campaign_parameter_input(cat_values):
    """
    Create the campaign parameter input section without target bolag field.
    Now only focusing on excluded bolags, as we're targeting all bolags except excluded ones.
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
    
    # Exclude Bolags multiselect - this becomes the primary targeting control
    bolag_options = list(BOLAG_VALUES.keys())
    excluded_bolags = st.multiselect(
        "Exclude Bolag Regions",
        options=bolag_options,
        help="Select Bolag regions to exclude from targeting. By default, we target all regions."
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
    subject = st.text_input("Subject Line", "Glöm inte ditt bolåneskydd!")
    
    # Extract subject features
    from feature_engineering import extract_subject_features
    subject_features = extract_subject_features(subject)
    
    # Create parameter dictionary to return
    parameters = {
        'dialog': dialog_code,
        'syfte': syfte_code,
        'product': product_code,
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

def analyze_age_groups(customer_df, delivery_df):
    """
    Analyze campaign performance by age groups
    
    Parameters:
    - customer_df: DataFrame with customer data including age information
    - delivery_df: DataFrame with delivery metrics
    
    Returns:
    - DataFrame with age group performance metrics
    - Plotly figure showing age group performance
    """
    import pandas as pd
    import plotly.express as px
    import numpy as np
    
    # Check if Age column exists
    if 'Age' not in customer_df.columns:
        return None, None
    
    # Merge customer data with delivery data
    # First compute average metrics per delivery
    delivery_metrics = delivery_df[['InternalName', 'open_rate', 'click_rate', 'optout_rate']].copy()
    
    # Join with customer data
    merged_data = customer_df.merge(delivery_metrics, on='InternalName', how='left')
    
    # Create age groups
    bins = [0, 18, 25, 35, 45, 55, 65, 75, 100]
    labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
    merged_data['age_group'] = pd.cut(merged_data['Age'], bins=bins, labels=labels, right=False)
    
    # Aggregate metrics by age group
    age_group_metrics = merged_data.groupby('age_group').agg(
        avg_open_rate=('open_rate', 'mean'),
        avg_click_rate=('click_rate', 'mean'),
        avg_optout_rate=('optout_rate', 'mean'),
        count=('Age', 'count')
    ).reset_index()
    
    # Fill NaN values for numeric columns only - THIS FIXES THE ERROR
    numeric_columns = ['avg_open_rate', 'avg_click_rate', 'avg_optout_rate', 'count']
    for col in numeric_columns:
        age_group_metrics[col] = age_group_metrics[col].fillna(0)
    
    # Create visualization
    metrics_data = []
    for _, row in age_group_metrics.iterrows():
        metrics_data.extend([
            {'age_group': row['age_group'], 'Metric': 'Open Rate', 'Value': row['avg_open_rate'], 'Count': row['count']},
            {'age_group': row['age_group'], 'Metric': 'Click Rate', 'Value': row['avg_click_rate'], 'Count': row['count']},
            {'age_group': row['age_group'], 'Metric': 'Optout Rate', 'Value': row['avg_optout_rate'], 'Count': row['count']}
        ])
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create visualization
    fig = px.bar(
        metrics_df,
        x='age_group',
        y='Value',
        color='Metric',
        barmode='group',
        text=metrics_df['Value'].round(2).astype(str) + '%',
        title='Campaign Performance by Age Group',
        hover_data=['Count'],
        labels={'Value': 'Rate (%)', 'age_group': 'Age Group'}
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Age Group',
        yaxis_title='Rate (%)'
    )
    
    return age_group_metrics, fig

def get_best_age_groups(age_group_metrics):
    """
    Get the best performing age groups for each metric
    
    Parameters:
    - age_group_metrics: DataFrame with age group performance metrics
    
    Returns:
    - Dictionary with best age groups for each metric
    """
    best_age_groups = {}
    
    if age_group_metrics is not None:
        # Best for open rate
        best_open = age_group_metrics.loc[age_group_metrics['avg_open_rate'].idxmax()]
        best_age_groups['open_rate'] = {
            'group': best_open['age_group'],
            'rate': best_open['avg_open_rate']
        }
        
        # Best for click rate
        best_click = age_group_metrics.loc[age_group_metrics['avg_click_rate'].idxmax()]
        best_age_groups['click_rate'] = {
            'group': best_click['age_group'],
            'rate': best_click['avg_click_rate']
        }
        
        # Best for optout rate (lowest is best)
        best_optout = age_group_metrics.loc[age_group_metrics['avg_optout_rate'].idxmin()]
        best_age_groups['optout_rate'] = {
            'group': best_optout['age_group'],
            'rate': best_optout['avg_optout_rate']
        }
    
    return best_age_groups

def analyze_time_patterns(delivery_df):
    """
    Analyze campaign performance by day of week and hour of day
    
    Parameters:
    - delivery_df: DataFrame with delivery metrics
    
    Returns:
    - Dictionary with day and time analysis results and figures
    """
    import pandas as pd
    import plotly.express as px
    import numpy as np
    
    results = {}
    
    # Check if Date column exists
    if 'Date' not in delivery_df.columns:
        return {'error': 'Date column not found in delivery data'}
    
    # Ensure Date is datetime
    delivery_df['Date'] = pd.to_datetime(delivery_df['Date'])
    
    # Add day of week
    if 'day_of_week' not in delivery_df.columns:
        delivery_df['day_of_week'] = delivery_df['Date'].dt.dayofweek
    
    # Add hour of day
    if 'hour_of_day' not in delivery_df.columns:
        delivery_df['hour_of_day'] = delivery_df['Date'].dt.hour
    
    # Get day of week names for better display
    day_names = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    
    delivery_df['day_name'] = delivery_df['day_of_week'].map(day_names)
    
    # Day of week analysis
    day_metrics = delivery_df.groupby('day_name').agg(
        avg_open_rate=('open_rate', 'mean'),
        avg_click_rate=('click_rate', 'mean'),
        avg_optout_rate=('optout_rate', 'mean'),
        count=('InternalName', 'count')
    ).reset_index()
    
    # Hour of day analysis
    hour_metrics = delivery_df.groupby('hour_of_day').agg(
        avg_open_rate=('open_rate', 'mean'),
        avg_click_rate=('click_rate', 'mean'),
        avg_optout_rate=('optout_rate', 'mean'),
        count=('InternalName', 'count')
    ).reset_index()
    
    # Daily trends (time series)
    delivery_df['date_only'] = delivery_df['Date'].dt.date
    daily_metrics = delivery_df.groupby('date_only').agg(
        avg_open_rate=('open_rate', 'mean'),
        avg_click_rate=('click_rate', 'mean'),
        avg_optout_rate=('optout_rate', 'mean'),
        count=('InternalName', 'count')
    ).reset_index()
    
    # Find best and worst days/times
    best_worst = {}
    
    # For open rate
    best_day_open = day_metrics.loc[day_metrics['avg_open_rate'].idxmax()]
    worst_day_open = day_metrics.loc[day_metrics['avg_open_rate'].idxmin()]
    
    best_hour_open = hour_metrics.loc[hour_metrics['avg_open_rate'].idxmax()]
    worst_hour_open = hour_metrics.loc[hour_metrics['avg_open_rate'].idxmin()]
    
    best_worst['open_rate'] = {
        'best_day': best_day_open['day_name'],
        'best_day_rate': best_day_open['avg_open_rate'],
        'worst_day': worst_day_open['day_name'],
        'worst_day_rate': worst_day_open['avg_open_rate'],
        'best_hour': best_hour_open['hour_of_day'],
        'best_hour_rate': best_hour_open['avg_open_rate'],
        'worst_hour': worst_hour_open['hour_of_day'],
        'worst_hour_rate': worst_hour_open['avg_open_rate']
    }
    
    # For click rate
    best_day_click = day_metrics.loc[day_metrics['avg_click_rate'].idxmax()]
    worst_day_click = day_metrics.loc[day_metrics['avg_click_rate'].idxmin()]
    
    best_hour_click = hour_metrics.loc[hour_metrics['avg_click_rate'].idxmax()]
    worst_hour_click = hour_metrics.loc[hour_metrics['avg_click_rate'].idxmin()]
    
    best_worst['click_rate'] = {
        'best_day': best_day_click['day_name'],
        'best_day_rate': best_day_click['avg_click_rate'],
        'worst_day': worst_day_click['day_name'],
        'worst_day_rate': worst_day_click['avg_click_rate'],
        'best_hour': best_hour_click['hour_of_day'],
        'best_hour_rate': best_hour_click['avg_click_rate'],
        'worst_hour': worst_hour_click['hour_of_day'],
        'worst_hour_rate': worst_hour_click['avg_click_rate']
    }
    
    # For optout rate (lower is better)
    best_day_optout = day_metrics.loc[day_metrics['avg_optout_rate'].idxmin()]
    worst_day_optout = day_metrics.loc[day_metrics['avg_optout_rate'].idxmax()]
    
    best_hour_optout = hour_metrics.loc[hour_metrics['avg_optout_rate'].idxmin()]
    worst_hour_optout = hour_metrics.loc[hour_metrics['avg_optout_rate'].idxmax()]
    
    best_worst['optout_rate'] = {
        'best_day': best_day_optout['day_name'],
        'best_day_rate': best_day_optout['avg_optout_rate'],
        'worst_day': worst_day_optout['day_name'],
        'worst_day_rate': worst_day_optout['avg_optout_rate'],
        'best_hour': best_hour_optout['hour_of_day'],
        'best_hour_rate': best_hour_optout['avg_optout_rate'],
        'worst_hour': worst_hour_optout['hour_of_day'],
        'worst_hour_rate': worst_hour_optout['avg_optout_rate']
    }
    
    # Create visualizations
    
    # Day of week charts
    day_metrics_melted = pd.melt(
        day_metrics, 
        id_vars=['day_name', 'count'], 
        value_vars=['avg_open_rate', 'avg_click_rate', 'avg_optout_rate'],
        var_name='Metric', 
        value_name='Rate'
    )
    
    # Make sure days are in correct order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_metrics_melted['day_name'] = pd.Categorical(
        day_metrics_melted['day_name'], 
        categories=day_order, 
        ordered=True
    )
    
    day_metrics_melted = day_metrics_melted.sort_values('day_name')
    
    # Map metric names for better display
    metric_names = {
        'avg_open_rate': 'Open Rate',
        'avg_click_rate': 'Click Rate',
        'avg_optout_rate': 'Optout Rate'
    }
    
    day_metrics_melted['Metric'] = day_metrics_melted['Metric'].map(metric_names)
    
    fig_days = px.line(
        day_metrics_melted,
        x='day_name',
        y='Rate',
        color='Metric',
        markers=True,
        title='Campaign Performance by Day of Week',
        hover_data=['count']
    )
    
    fig_days.update_layout(
        xaxis_title='Day of Week',
        yaxis_title='Rate (%)'
    )
    
    # Hour of day charts
    hour_metrics_melted = pd.melt(
        hour_metrics, 
        id_vars=['hour_of_day', 'count'], 
        value_vars=['avg_open_rate', 'avg_click_rate', 'avg_optout_rate'],
        var_name='Metric', 
        value_name='Rate'
    )
    
    hour_metrics_melted['Metric'] = hour_metrics_melted['Metric'].map(metric_names)
    
    fig_hours = px.line(
        hour_metrics_melted,
        x='hour_of_day',
        y='Rate',
        color='Metric',
        markers=True,
        title='Campaign Performance by Hour of Day',
        hover_data=['count']
    )
    
    fig_hours.update_layout(
        xaxis_title='Hour of Day (24h)',
        yaxis_title='Rate (%)'
    )
    
    # Daily trend charts
    daily_metrics_melted = pd.melt(
        daily_metrics, 
        id_vars=['date_only', 'count'], 
        value_vars=['avg_open_rate', 'avg_click_rate', 'avg_optout_rate'],
        var_name='Metric', 
        value_name='Rate'
    )
    
    daily_metrics_melted['Metric'] = daily_metrics_melted['Metric'].map(metric_names)
    
    fig_daily = px.line(
        daily_metrics_melted,
        x='date_only',
        y='Rate',
        color='Metric',
        title='Daily Campaign Performance Trends',
        hover_data=['count']
    )
    
    fig_daily.update_layout(
        xaxis_title='Date',
        yaxis_title='Rate (%)'
    )
    
    # Save results
    results['day_metrics'] = day_metrics
    results['hour_metrics'] = hour_metrics
    results['daily_metrics'] = daily_metrics
    results['best_worst'] = best_worst
    results['fig_days'] = fig_days
    results['fig_hours'] = fig_hours
    results['fig_daily'] = fig_daily
    
    return results

def create_forecast_tab(customer_df, delivery_df, formatted_predictions, parameters, BOLAG_VALUES):
    """
    Create the forecast tab content that shows potential reach and performance improvements
    
    Parameters:
    - customer_df: DataFrame with customer data
    - delivery_df: DataFrame with delivery metrics
    - formatted_predictions: Formatted prediction results
    - parameters: Input parameters from the form
    - BOLAG_VALUES: Dictionary mapping bolag names to codes
    
    Returns:
    - None (displays content directly via Streamlit)
    """
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    
    st.header("Campaign Performance Forecast")
    
    # Calculate total potential reach
    total_customers = len(customer_df['Primary key'].unique()) if 'Primary key' in customer_df.columns else 0
    
    # Get excluded bolag names
    excluded_bolag_names = []
    if parameters['excluded_bolags']:
        excluded_bolag_names = [
            name for name, code in BOLAG_VALUES.items() 
            if code in parameters['excluded_bolags']
        ]
    
    # Calculate customers by bolag
    bolag_counts = None
    if 'Bolag' in customer_df.columns:
        bolag_counts = customer_df['Bolag'].value_counts().reset_index()
        bolag_counts.columns = ['Bolag', 'Count']
        
        # Map bolag codes to names if needed
        bolag_mapping = {code: name for name, code in BOLAG_VALUES.items()}
        if bolag_counts['Bolag'].iloc[0] in bolag_mapping:
            bolag_counts['Bolag Name'] = bolag_counts['Bolag'].map(bolag_mapping)
        else:
            bolag_counts['Bolag Name'] = bolag_counts['Bolag']
    
    # Calculate customers by age group
    age_counts = None
    if 'Age' in customer_df.columns:
        # Create age groups
        bins = [0, 18, 25, 35, 45, 55, 65, 75, 100]
        labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
        customer_df['age_group'] = pd.cut(customer_df['Age'], bins=bins, labels=labels, right=False)
        
        age_counts = customer_df['age_group'].value_counts().reset_index()
        age_counts.columns = ['Age Group', 'Count']
    
    # Calculate estimated reach
    excluded_customer_count = 0
    if bolag_counts is not None and parameters['excluded_bolags']:
        for bolag_code in parameters['excluded_bolags']:
            # Find the corresponding bolag in our counts
            bolag_name = next((name for name, code in BOLAG_VALUES.items() if code == bolag_code), bolag_code)
            matching_bolag = bolag_counts[bolag_counts['Bolag'] == bolag_code]
            if not matching_bolag.empty:
                excluded_customer_count += matching_bolag.iloc[0]['Count']
    
    estimated_reach = total_customers - excluded_customer_count
    
    # Calculate age filter impact
    age_filter_exclusions = 0
    if age_counts is not None:
        # Find customers outside the age range
        age_counts_filtered = age_counts.copy()
        
        # Convert age group labels to start age (for comparison)
        def extract_min_age(age_group):
            if age_group == '<18':
                return 0
            elif '-' in str(age_group):
                return int(str(age_group).split('-')[0])
            else:
                return 75  # For 75+ group
        
        age_counts_filtered['min_age'] = age_counts_filtered['Age Group'].apply(extract_min_age)
        
        # Filter out age groups outside our range
        excluded_ages = age_counts_filtered[
            (age_counts_filtered['min_age'] < parameters['min_age']) | 
            (age_counts_filtered['min_age'] >= parameters['max_age'])
        ]
        
        age_filter_exclusions = excluded_ages['Count'].sum() if not excluded_ages.empty else 0
    
    final_estimated_reach = estimated_reach - age_filter_exclusions
    
    # Display reach metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Available Customers",
            f"{total_customers:,}",
            help="Total number of customers in the database"
        )
    
    with col2:
        st.metric(
            "Estimated Campaign Reach",
            f"{final_estimated_reach:,}",
            f"{final_estimated_reach - total_customers:,}" if final_estimated_reach != total_customers else None,
            help="Estimated number of customers who will receive this campaign, after applying filters"
        )
    
    with col3:
        reach_percentage = (final_estimated_reach / total_customers * 100) if total_customers > 0 else 0
        st.metric(
            "Reach Percentage",
            f"{reach_percentage:.1f}%",
            help="Percentage of total customers who will receive this campaign"
        )
    
    # Section explaining excluded segments
    if excluded_bolag_names or age_filter_exclusions > 0:
        st.subheader("Segment Exclusions")
        
        if excluded_bolag_names:
            st.info(f"**Excluded Bolag Regions:** {', '.join(excluded_bolag_names)}")
            st.write(f"This excludes approximately {excluded_customer_count:,} customers.")
        
        if age_filter_exclusions > 0:
            st.info(f"**Age Range Filter:** {parameters['min_age']} to {parameters['max_age']} years")
            st.write(f"This excludes approximately {age_filter_exclusions:,} customers outside the age range.")
    
    # Performance forecast
    st.subheader("Performance Forecast")
    
    # Calculate expected metrics based on reach and rates
    current_metrics = {
        'open_rate': formatted_predictions['current']['open_rate'],
        'click_rate': formatted_predictions['current']['click_rate'],
        'optout_rate': formatted_predictions['current']['optout_rate']
    }
    
    recommended_metrics = {
        'open_rate': formatted_predictions['combined']['open_rate'],
        'click_rate': formatted_predictions['combined']['click_rate'],
        'optout_rate': formatted_predictions['combined']['optout_rate']
    }
    
    # Calculate absolute numbers
    current_estimates = {
        'reaches': final_estimated_reach,
        'opens': int(final_estimated_reach * current_metrics['open_rate'] / 100),
        'clicks': int(final_estimated_reach * current_metrics['open_rate'] / 100 * current_metrics['click_rate'] / 100),
        'optouts': int(final_estimated_reach * current_metrics['open_rate'] / 100 * current_metrics['optout_rate'] / 100)
    }
    
    recommended_estimates = {
        'reaches': final_estimated_reach,
        'opens': int(final_estimated_reach * recommended_metrics['open_rate'] / 100),
        'clicks': int(final_estimated_reach * recommended_metrics['open_rate'] / 100 * recommended_metrics['click_rate'] / 100),
        'optouts': int(final_estimated_reach * recommended_metrics['open_rate'] / 100 * recommended_metrics['optout_rate'] / 100)
    }
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Estimated Reaches",
            f"{current_estimates['reaches']:,}"
        )
        
    with col2:
        st.metric(
            "Estimated Opens",
            f"{current_estimates['opens']:,}",
            f"{recommended_estimates['opens'] - current_estimates['opens']:,}"
        )
        
    with col3:
        st.metric(
            "Estimated Clicks",
            f"{current_estimates['clicks']:,}",
            f"{recommended_estimates['clicks'] - current_estimates['clicks']:,}"
        )
        
    with col4:
        st.metric(
            "Estimated Optouts",
            f"{current_estimates['optouts']:,}",
            f"{recommended_estimates['optouts'] - current_estimates['optouts']:,}"
        )
    
    # Create comparison chart
    forecast_data = pd.DataFrame({
        'Metric': ['Opens', 'Clicks', 'Optouts'],
        'Current Strategy': [
            current_estimates['opens'],
            current_estimates['clicks'],
            current_estimates['optouts']
        ],
        'Recommended Strategy': [
            recommended_estimates['opens'],
            recommended_estimates['clicks'],
            recommended_estimates['optouts']
        ]
    })
    
    forecast_melted = pd.melt(
        forecast_data,
        id_vars='Metric',
        var_name='Strategy',
        value_name='Count'
    )
    
    fig_forecast = px.bar(
        forecast_melted,
        x='Metric',
        y='Count',
        color='Strategy',
        barmode='group',
        title='Estimated Campaign Performance Comparison',
        text=forecast_melted['Count'].apply(lambda x: f"{x:,}")
    )
    
    fig_forecast.update_traces(textposition='outside')
    fig_forecast.update_layout(
        xaxis_title='Metric',
        yaxis_title='Estimated Count'
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
        
    # Recommendations summary
    st.subheader("Key Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("#### Targeting Recommendations")
        
        if not parameters['excluded_bolags']:
            st.success("✅ **Targeting all Bolag regions** - Maximizing reach")
        else:
            st.warning(f"⚠️ **Excluding {len(parameters['excluded_bolags'])} Bolag regions** - Consider expanding reach")
        
        # Age range recommendation
        age_range_text = f"**Current Age Range:** {parameters['min_age']} to {parameters['max_age']} years"
        if parameters['min_age'] > 25 or parameters['max_age'] < 65:
            st.warning(f"⚠️ {age_range_text} - Consider expanding to capture more customers")
        else:
            st.success(f"✅ {age_range_text} - Good coverage of key segments")
    
    with rec_col2:
        st.markdown("#### Content Recommendations")
        
        st.success(f"✅ **Recommended Subject Line:** '{formatted_predictions['subject']['text']}'")
        st.info(f"**Predicted Open Rate:** {formatted_predictions['subject']['open_rate']:.2f}% (Increase: {formatted_predictions['subject']['open_rate_diff']:.2f}%)")
        
        # Dialog, syfte, product recommendations
        dialog_name = next((name for name, code in DIALOG_VALUES.items() if code[0] == parameters['dialog']), parameters['dialog'])
        syfte_name = next((name for name, code in SYFTE_VALUES.items() if code[0] == parameters['syfte']), parameters['syfte'])
        
        st.markdown(f"**Selected Dialog:** {dialog_name}")
        st.markdown(f"**Selected Purpose:** {syfte_name}")

def display_kpi_dashboard(formatted_predictions, delivery_df):
    """
    Display a KPI dashboard showing current and recommended KPIs
    
    Parameters:
    - formatted_predictions: Formatted prediction results
    - delivery_df: DataFrame with delivery metrics
    """
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    
    st.header("Campaign KPI Dashboard")
    
    # Calculate historical averages
    historical_metrics = {
        'open_rate': delivery_df['open_rate'].mean() if 'open_rate' in delivery_df.columns else 0,
        'click_rate': delivery_df['click_rate'].mean() if 'click_rate' in delivery_df.columns else 0,
        'optout_rate': delivery_df['optout_rate'].mean() if 'optout_rate' in delivery_df.columns else 0
    }
    
    # Calculate click-to-open rate (CTOR)
    current_ctor = (formatted_predictions['current']['click_rate'] / formatted_predictions['current']['open_rate'] * 100) if formatted_predictions['current']['open_rate'] > 0 else 0
    recommended_ctor = (formatted_predictions['combined']['click_rate'] / formatted_predictions['combined']['open_rate'] * 100) if formatted_predictions['combined']['open_rate'] > 0 else 0
    historical_ctor = (historical_metrics['click_rate'] / historical_metrics['open_rate'] * 100) if historical_metrics['open_rate'] > 0 else 0
    
    # Calculate engagement score (custom metric combining open and click rates, minus optout rate)
    def calculate_engagement(open_rate, click_rate, optout_rate):
        return (open_rate * 0.4) + (click_rate * 0.6) - (optout_rate * 2)
    
    current_engagement = calculate_engagement(
        formatted_predictions['current']['open_rate'],
        formatted_predictions['current']['click_rate'],
        formatted_predictions['current']['optout_rate']
    )
    
    recommended_engagement = calculate_engagement(
        formatted_predictions['combined']['open_rate'],
        formatted_predictions['combined']['click_rate'],
        formatted_predictions['combined']['optout_rate']
    )
    
    historical_engagement = calculate_engagement(
        historical_metrics['open_rate'],
        historical_metrics['click_rate'],
        historical_metrics['optout_rate']
    )
    
    # Create KPI tables
    kpi_data = pd.DataFrame({
        'KPI': ['Open Rate', 'Click Rate', 'Optout Rate', 'Click-to-Open Rate', 'Engagement Score'],
        'Current': [
            formatted_predictions['current']['open_rate'],
            formatted_predictions['current']['click_rate'],
            formatted_predictions['current']['optout_rate'],
            current_ctor,
            current_engagement
        ],
        'Recommended': [
            formatted_predictions['combined']['open_rate'],
            formatted_predictions['combined']['click_rate'],
            formatted_predictions['combined']['optout_rate'],
            recommended_ctor,
            recommended_engagement
        ],
        'Historical Average': [
            historical_metrics['open_rate'],
            historical_metrics['click_rate'],
            historical_metrics['optout_rate'],
            historical_ctor,
            historical_engagement
        ]
    })
    
    # Calculate differences
    kpi_data['Δ Recommended'] = kpi_data['Recommended'] - kpi_data['Current']
    kpi_data['Δ Historical'] = kpi_data['Current'] - kpi_data['Historical Average']
    
    # Check if values are too small or NaN/infinite - replace with reasonable defaults if needed
    kpi_data = kpi_data.replace([np.inf, -np.inf], np.nan)
    
    # For any row where Current is 0 or NaN, set a small default value
    for idx, row in kpi_data.iterrows():
        if pd.isna(row['Current']) or row['Current'] == 0:
            kpi_data.at[idx, 'Current'] = 0.1
        if pd.isna(row['Recommended']) or row['Recommended'] == 0:
            kpi_data.at[idx, 'Recommended'] = 0.1
        if pd.isna(row['Historical Average']) or row['Historical Average'] == 0:
            kpi_data.at[idx, 'Historical Average'] = 0.1
        if pd.isna(row['Δ Recommended']):
            kpi_data.at[idx, 'Δ Recommended'] = 0
        if pd.isna(row['Δ Historical']):
            kpi_data.at[idx, 'Δ Historical'] = 0
    
    # Display the KPI table
    st.dataframe(
        kpi_data.style.format({
            'Current': '{:.2f}%',
            'Recommended': '{:.2f}%',
            'Historical Average': '{:.2f}%',
            'Δ Recommended': '{:+.2f}%',
            'Δ Historical': '{:+.2f}%'
        }).background_gradient(
            subset=['Δ Recommended'], 
            cmap='RdYlGn', 
            vmin=-3, 
            vmax=3
        ).background_gradient(
            subset=['Δ Historical'], 
            cmap='RdYlGn', 
            vmin=-3, 
            vmax=3
        ),
        use_container_width=True
    )
    
    # Create KPI gauge charts
    st.subheader("Key Performance Indicators")
    
    # Create two rows of metrics
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2 = st.columns(2)
    
    # Define targets for each KPI
    targets = {
        'open_rate': historical_metrics['open_rate'] * 1.1,  # 10% above historical average
        'click_rate': historical_metrics['click_rate'] * 1.15,  # 15% above historical average
        'optout_rate': historical_metrics['optout_rate'] * 0.9,  # 10% below historical average
        'ctor': historical_ctor * 1.05,  # 5% above historical average
        'engagement': historical_engagement * 1.1  # 10% above historical average
    }
    
    # Ensure all targets have reasonable minimum values
    targets = {k: max(v, 0.1) for k, v in targets.items()}
    
    # Create gauge charts with unique keys
    with row1_col1:
        fig_open = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=formatted_predictions['current']['open_rate'],
            delta={'reference': historical_metrics['open_rate'], 'relative': False, 'valueformat': '.2f'},
            title={'text': "Open Rate (%)"},
            gauge={
                'axis': {'range': [0, max(formatted_predictions['combined']['open_rate'], targets['open_rate']) * 1.2]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, historical_metrics['open_rate']], 'color': "lightgray"},
                    {'range': [historical_metrics['open_rate'], targets['open_rate']], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': formatted_predictions['combined']['open_rate']
                }
            }
        ))
        st.plotly_chart(fig_open, use_container_width=True, key="open_rate_gauge")
    
    with row1_col2:
        fig_click = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=formatted_predictions['current']['click_rate'],
            delta={'reference': historical_metrics['click_rate'], 'relative': False, 'valueformat': '.2f'},
            title={'text': "Click Rate (%)"},
            gauge={
                'axis': {'range': [0, max(formatted_predictions['combined']['click_rate'], targets['click_rate']) * 1.2]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, historical_metrics['click_rate']], 'color': "lightgray"},
                    {'range': [historical_metrics['click_rate'], targets['click_rate']], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': formatted_predictions['combined']['click_rate']
                }
            }
        ))
        st.plotly_chart(fig_click, use_container_width=True, key="click_rate_gauge")
    
    with row1_col3:
        # For optout rate, lower is better
        fig_optout = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=formatted_predictions['current']['optout_rate'],
            title={'text': "Optout Rate (%)"},
            delta={
                'reference': historical_metrics['optout_rate'], 
                'relative': False, 
                'valueformat': '.2f',
                'decreasing': {'color': 'green'},
                'increasing': {'color': 'red'}
            },
            gauge={
                'axis': {'range': [0, max(formatted_predictions['current']['optout_rate'], historical_metrics['optout_rate']) * 1.5]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, targets['optout_rate']], 'color': "lightgreen"},
                    {'range': [targets['optout_rate'], historical_metrics['optout_rate']], 'color': "lightyellow"},
                    {'range': [historical_metrics['optout_rate'], historical_metrics['optout_rate'] * 1.5], 'color': "pink"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': formatted_predictions['combined']['optout_rate']
                }
            }
        ))
        st.plotly_chart(fig_optout, use_container_width=True, key="optout_rate_gauge")
    
    with row2_col1:
        fig_ctor = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_ctor,
            title={'text': "Click-to-Open Rate (%)"},
            delta={'reference': historical_ctor, 'relative': False, 'valueformat': '.2f'},
            gauge={
                'axis': {'range': [0, max(recommended_ctor, targets['ctor']) * 1.2]},
                'bar': {'color': "purple"},
                'steps': [
                    {'range': [0, historical_ctor], 'color': "lightgray"},
                    {'range': [historical_ctor, targets['ctor']], 'color': "lavender"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': recommended_ctor
                }
            }
        ))
        st.plotly_chart(fig_ctor, use_container_width=True, key="ctor_gauge")
    
    with row2_col2:
        fig_engagement = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_engagement,
            title={'text': "Engagement Score"},
            delta={'reference': historical_engagement, 'relative': False, 'valueformat': '.2f'},
            gauge={
                'axis': {'range': [0, max(recommended_engagement, targets['engagement']) * 1.2]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, historical_engagement], 'color': "lightgray"},
                    {'range': [historical_engagement, targets['engagement']], 'color': "peachpuff"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': recommended_engagement
                }
            }
        ))
        st.plotly_chart(fig_engagement, use_container_width=True, key="engagement_gauge")
    
    # Display KPI descriptions
    with st.expander("KPI Descriptions"):
        st.markdown("""
        ### KPI Definitions
        
        - **Open Rate**: Percentage of delivered emails that were opened.
        - **Click Rate**: Percentage of opened emails that had at least one click.
        - **Optout Rate**: Percentage of opened emails that resulted in an unsubscribe.
        - **Click-to-Open Rate (CTOR)**: Ratio of clicks to opens, indicating how effective the email content is at generating clicks once opened.
        - **Engagement Score**: A composite score calculated from open rate, click rate, and optout rate to measure overall engagement.
        
        ### KPI Calculation
        
        - **Open Rate** = (Number of Opens / Number of Sends) * 100%
        - **Click Rate** = (Number of Clicks / Number of Opens) * 100%
        - **Optout Rate** = (Number of Optouts / Number of Opens) * 100%
        - **Click-to-Open Rate** = (Click Rate / Open Rate) * 100%
        - **Engagement Score** = (Open Rate * 0.4) + (Click Rate * 0.6) - (Optout Rate * 2)
        
        ### Target Values
        
        Targets are calculated based on historical performance:
        - **Open Rate Target**: 10% above historical average
        - **Click Rate Target**: 15% above historical average
        - **Optout Rate Target**: 10% below historical average
        - **CTOR Target**: 5% above historical average
        - **Engagement Score Target**: 10% above historical average
        """)

def validate_predictions(predictions):
    """
    Validates prediction results and fixes any issues with zeros, NaNs, or unrealistic values.
    
    Parameters:
    - predictions: Dictionary of prediction results from format_predictions
    
    Returns:
    - Validated prediction results
    """
    import logging
    import numpy as np
    
    # Define reasonable ranges for each metric
    valid_ranges = {
        'open_rate': (5.0, 80.0),  # 5% to 80%
        'click_rate': (0.5, 30.0),  # 0.5% to 30%
        'optout_rate': (0.01, 5.0)  # 0.01% to 5%
    }
    
    # Default values to use if predicted values are invalid
    default_values = {
        'open_rate': 25.0,  # 25% is a reasonable default open rate
        'click_rate': 3.0,   # 3% is a reasonable default click rate
        'optout_rate': 0.2   # 0.2% is a reasonable default optout rate
    }
    
    # Deep copy the predictions to avoid modifying the original
    validated = {}
    for key, value in predictions.items():
        if isinstance(value, dict):
            validated[key] = value.copy()
        else:
            validated[key] = value
            
    # Helper function to validate a specific metric value
    def validate_metric(value, metric_name):
        # Convert to float if it's not already
        try:
            if value is None:
                logging.warning(f"Invalid {metric_name} value: None, using default {default_values[metric_name]}")
                return default_values[metric_name]
            
            # Handle non-numeric values
            if isinstance(value, str) or isinstance(value, bool):
                logging.warning(f"Invalid {metric_name} value type: {type(value)}, using default {default_values[metric_name]}")
                return default_values[metric_name]
                
            # Convert to float and check for NaN/infinity
            float_value = float(value)
            
            # Check if the value is NaN or infinite
            if np.isnan(float_value) or np.isinf(float_value):
                logging.warning(f"Invalid {metric_name} value: {value}, using default {default_values[metric_name]}")
                return default_values[metric_name]
                
            # Check if the value is within a reasonable range
            min_val, max_val = valid_ranges[metric_name]
            if float_value < min_val or float_value > max_val:
                logging.warning(f"Invalid {metric_name} value: {float_value}, using default {default_values[metric_name]}")
                return default_values[metric_name]
                
            return float_value
                
        except Exception as e:
            logging.warning(f"Error validating {metric_name}: {e}, using default {default_values[metric_name]}")
            return default_values[metric_name]
    
    # Validate current predictions
    if 'current' in validated:
        for metric in ['open_rate', 'click_rate', 'optout_rate']:
            if metric in validated['current']:
                validated['current'][metric] = validate_metric(validated['current'][metric], metric)
    
    # Validate targeting predictions
    if 'targeting' in validated:
        if 'predictions' in validated['targeting']:
            for metric in ['open_rate', 'click_rate', 'optout_rate']:
                if metric in validated['targeting']['predictions']:
                    validated['targeting']['predictions'][metric] = validate_metric(
                        validated['targeting']['predictions'][metric], metric
                    )
        else:
            # For older format where metrics are directly in the targeting dict
            for metric in ['open_rate', 'click_rate', 'optout_rate']:
                if metric in validated['targeting']:
                    validated['targeting'][metric] = validate_metric(validated['targeting'][metric], metric)
    
    # Validate subject predictions (only open_rate)
    if 'subject' in validated and 'open_rate' in validated['subject']:
        validated['subject']['open_rate'] = validate_metric(validated['subject']['open_rate'], 'open_rate')
    
    # Validate combined predictions
    if 'combined' in validated:
        if 'predictions' in validated['combined']:
            for metric in ['open_rate', 'click_rate', 'optout_rate']:
                if metric in validated['combined']['predictions']:
                    validated['combined']['predictions'][metric] = validate_metric(
                        validated['combined']['predictions'][metric], metric
                    )
        else:
            # For older format where metrics are directly in the combined dict
            for metric in ['open_rate', 'click_rate', 'optout_rate']:
                if metric in validated['combined']:
                    validated['combined'][metric] = validate_metric(validated['combined'][metric], metric)
    
    return validated
       
def display_targeting_recommendations(formatted_predictions, parameters, time_analysis, age_group_metrics):
    """
    Display targeting recommendations with a focus on excluded bolags and timing,
    instead of suggesting a single bolag.
    
    Parameters:
    - formatted_predictions: Formatted prediction results
    - parameters: Input parameters from the form
    - time_analysis: Results from time-based analysis
    - age_group_metrics: Results from age group analysis
    """
    import streamlit as st
    
    st.subheader("Targeting Recommendation")
    
    # Display excluded bolags
    if parameters['excluded_bolags']:
        st.info(f"**Current Excluded Bolag Regions:** {', '.join(parameters['excluded_bolags'])}")
    else:
        st.success("**Current Strategy:** Targeting all Bolag regions (no exclusions)")
    
    # Display metrics
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
    
    # Display timing recommendations if available
    if time_analysis and 'best_worst' in time_analysis:
        st.markdown("### Optimal Timing Recommendations")
        
        best_worst = time_analysis['best_worst']
        
        timing_col1, timing_col2 = st.columns(2)
        
        with timing_col1:
            st.markdown("#### Best Time to Send")
            
            # Best day for open rate
            if 'open_rate' in best_worst:
                best_day = best_worst['open_rate']['best_day']
                best_hour = best_worst['open_rate']['best_hour']
                st.success(f"📈 **For highest open rates:** Send on **{best_day}** at **{best_hour}:00**")
                
                st.markdown(f"Open Rate: {best_worst['open_rate']['best_day_rate']:.2f}% (day), {best_worst['open_rate']['best_hour_rate']:.2f}% (hour)")
            
            # Best day for click rate
            if 'click_rate' in best_worst:
                best_day = best_worst['click_rate']['best_day']
                best_hour = best_worst['click_rate']['best_hour']
                st.success(f"🖱️ **For highest click rates:** Send on **{best_day}** at **{best_hour}:00**")
                
                st.markdown(f"Click Rate: {best_worst['click_rate']['best_day_rate']:.2f}% (day), {best_worst['click_rate']['best_hour_rate']:.2f}% (hour)")
            
            # Best day for optout rate (lowest is best)
            if 'optout_rate' in best_worst:
                best_day = best_worst['optout_rate']['best_day']
                best_hour = best_worst['optout_rate']['best_hour']
                st.success(f"👍 **For lowest optout rates:** Send on **{best_day}** at **{best_hour}:00**")
                
                st.markdown(f"Optout Rate: {best_worst['optout_rate']['best_day_rate']:.2f}% (day), {best_worst['optout_rate']['best_hour_rate']:.2f}% (hour)")
        
        with timing_col2:
            st.markdown("#### Times to Avoid")
            
            # Worst day for open rate
            if 'open_rate' in best_worst:
                worst_day = best_worst['open_rate']['worst_day']
                worst_hour = best_worst['open_rate']['worst_hour']
                st.error(f"📉 **For open rates, avoid:** Sending on **{worst_day}** at **{worst_hour}:00**")
                
                st.markdown(f"Open Rate: {best_worst['open_rate']['worst_day_rate']:.2f}% (day), {best_worst['open_rate']['worst_hour_rate']:.2f}% (hour)")
            
            # Worst day for click rate
            if 'click_rate' in best_worst:
                worst_day = best_worst['click_rate']['worst_day']
                worst_hour = best_worst['click_rate']['worst_hour']
                st.error(f"🖱️ **For click rates, avoid:** Sending on **{worst_day}** at **{worst_hour}:00**")
                
                st.markdown(f"Click Rate: {best_worst['click_rate']['worst_day_rate']:.2f}% (day), {best_worst['click_rate']['worst_hour_rate']:.2f}% (hour)")
            
            # Worst day for optout rate (highest)
            if 'optout_rate' in best_worst:
                worst_day = best_worst['optout_rate']['worst_day']
                worst_hour = best_worst['optout_rate']['worst_hour']
                st.error(f"👎 **For optout rates, avoid:** Sending on **{worst_day}** at **{worst_hour}:00**")
                
                st.markdown(f"Optout Rate: {best_worst['optout_rate']['worst_day_rate']:.2f}% (day), {best_worst['optout_rate']['worst_hour_rate']:.2f}% (hour)")
    
    # Display age group recommendations if available
    if age_group_metrics is not None:
        st.markdown("### Age Group Performance")
        
        best_age_open = age_group_metrics.loc[age_group_metrics['avg_open_rate'].idxmax()]
        best_age_click = age_group_metrics.loc[age_group_metrics['avg_click_rate'].idxmax()]
        best_age_optout = age_group_metrics.loc[age_group_metrics['avg_optout_rate'].idxmin()]  # Lowest is best
        
        age_col1, age_col2, age_col3 = st.columns(3)
        
        with age_col1:
            st.markdown("#### Best Age Group for Opens")
            st.success(f"**{best_age_open['age_group']}**")
            st.markdown(f"Open Rate: {best_age_open['avg_open_rate']:.2f}%")
        
        with age_col2:
            st.markdown("#### Best Age Group for Clicks")
            st.success(f"**{best_age_click['age_group']}**")
            st.markdown(f"Click Rate: {best_age_click['avg_click_rate']:.2f}%")
        
        with age_col3:
            st.markdown("#### Best Age Group for Retention")
            st.success(f"**{best_age_optout['age_group']}**")
            st.markdown(f"Optout Rate: {best_age_optout['avg_optout_rate']:.2f}%")
        
        # Age range recommendation
        current_age_range = f"{parameters['min_age']} to {parameters['max_age']}"
        
        if (parameters['min_age'] > int(str(best_age_open['age_group']).split('-')[0] if '-' in str(best_age_open['age_group']) else 0) or
            parameters['max_age'] < int(str(best_age_open['age_group']).split('-')[1] if '-' in str(best_age_open['age_group']) else 100)):
            st.warning(f"⚠️ Current age range (**{current_age_range}**) might be missing the best performing age group for opens.")
        else:
            st.success(f"✅ Current age range (**{current_age_range}**) includes the best performing age group for opens.")
    
    # Display combined strategy recommendation
    st.markdown("### Combined Targeting Strategy")
    
    # Determine the optimal day and hour (prioritizing open rate)
    optimal_day = "unknown"
    optimal_hour = "unknown"
    
    if time_analysis and 'best_worst' in time_analysis and 'open_rate' in time_analysis['best_worst']:
        optimal_day = time_analysis['best_worst']['open_rate']['best_day']
        optimal_hour = time_analysis['best_worst']['open_rate']['best_hour']
    
    # Recommended age range
    recommended_min_age = parameters['min_age']
    recommended_max_age = parameters['max_age']
    
    if age_group_metrics is not None:
        # Find the best performing age groups
        top_performing_groups = age_group_metrics.nlargest(3, 'avg_open_rate')['age_group'].tolist()
        
        # Extract min and max ages from these groups
        min_ages = []
        max_ages = []
        
        for group in top_performing_groups:
            group_str = str(group)
            if group_str == '<18':
                min_ages.append(0)
                max_ages.append(18)
            elif '-' in group_str:
                parts = group_str.split('-')
                min_ages.append(int(parts[0]))
                max_ages.append(int(parts[1]))
            elif group_str == '75+':
                min_ages.append(75)
                max_ages.append(100)
        
        if min_ages and max_ages:
            recommended_min_age = min(min_ages)
            recommended_max_age = max(max_ages)
    
    # Generate recommendation text
    recommendations = []
    
    # Timing recommendation
    if optimal_day != "unknown" and optimal_hour != "unknown":
        recommendations.append(f"**Send time:** {optimal_day} at {optimal_hour}:00")
    
    # Age range recommendation
    if recommended_min_age != parameters['min_age'] or recommended_max_age != parameters['max_age']:
        recommendations.append(f"**Consider age range:** {recommended_min_age} to {recommended_max_age} years")
    else:
        recommendations.append(f"**Age range:** {parameters['min_age']} to {parameters['max_age']} years is optimal")
    
    # Bolag recommendation
    if parameters['excluded_bolags']:
        recommendations.append(f"**Consider including:** some of the currently excluded Bolag regions to increase reach")
    else:
        recommendations.append("**Targeting all Bolag regions:** maximize reach")
    
    # Display recommendations
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # Display estimated impact
    st.markdown("### Estimated Impact")
    
    impact_col1, impact_col2, impact_col3 = st.columns(3)
    
    with impact_col1:
        st.metric(
            "Open Rate Impact", 
            f"{formatted_predictions['targeting']['open_rate']:.2f}%",
            f"{formatted_predictions['targeting']['open_rate_diff']:.2f}%"
        )
    
    with impact_col2:
        st.metric(
            "Click Rate Impact",
            f"{formatted_predictions['targeting']['click_rate']:.2f}%",
            f"{formatted_predictions['targeting']['click_rate_diff']:.2f}%"
        )
    
    with impact_col3:
        st.metric(
            "Optout Rate Impact",
            f"{formatted_predictions['targeting']['optout_rate']:.2f}%",
            f"{formatted_predictions['targeting']['optout_rate_diff']:.2f}%"
        )
        
        
def generate_campaign_report(parameters, formatted_predictions, BOLAG_VALUES):
    """
    Generate a comprehensive campaign report with all predictions and parameters.
    
    Parameters:
    - parameters: Parameters dictionary from campaign_parameter_input
    - formatted_predictions: Formatted prediction results
    - BOLAG_VALUES: Dictionary mapping bolag names to codes
    
    Returns:
    - Report text in markdown format
    """
    import datetime
    
    # Get the bolag name from the code
    selected_bolag_code = parameters['bolag']
    selected_bolag_name = next((name for name, code in BOLAG_VALUES.items() if code == selected_bolag_code), selected_bolag_code)
    
    # Get excluded bolags
    excluded_bolags = []
    if 'excluded_bolags' in parameters and parameters['excluded_bolags']:
        excluded_bolags = [
            next((name for name, code in BOLAG_VALUES.items() if code == bolag_code), bolag_code)
            for bolag_code in parameters['excluded_bolags']
        ]
    
    # Create the report text
    report = f"""
    # Email Campaign Prediction Report
    **Date:** {datetime.date.today().strftime('%Y-%m-%d')}

    """
    
    # Add excluded bolags if any
    if excluded_bolags:
        report += f"- **Excluded Bolag:** {', '.join(excluded_bolags)}\n"
    
    # Add the rest of the parameters
    report += f"""
    - **Dialog:** {parameters['dialog']}
    - **Purpose:** {parameters['syfte']}
    - **Product:** {parameters['product']}
    - **Age Range:** {parameters['min_age']} to {parameters['max_age']} years
    - **Percentage Women:** {parameters['pct_women']}%
    - **Send Date/Time:** {parameters['day_of_week']} at {parameters['hour_of_day']}:00
    - **Subject Line:** "{parameters['subject']}"

    ## Current Campaign Predictions
    - **Open Rate:** {formatted_predictions['current']['open_rate']:.2f}% (Confidence: {formatted_predictions['current']['confidence']['open_rate']:.0f}%)
    - **Click Rate:** {formatted_predictions['current']['click_rate']:.2f}% (Confidence: {formatted_predictions['current']['confidence']['click_rate']:.0f}%)
    - **Optout Rate:** {formatted_predictions['current']['optout_rate']:.2f}% (Confidence: {formatted_predictions['current']['confidence']['optout_rate']:.0f}%)

    ## Subject Line Recommendation (Affects Open Rate Only)
    - **Recommended Subject:** "{formatted_predictions['subject']['text']}"
    - **Predicted Open Rate:** {formatted_predictions['subject']['open_rate']:.2f}% (Change: {formatted_predictions['subject']['open_rate_diff']:.2f}%)
    - **Confidence:** {formatted_predictions['subject']['confidence']['open_rate']:.0f}%

    ## Targeting Recommendation (Affects All Metrics)
    """
    
    # Get the recommended bolag name
    recommended_bolag_code = formatted_predictions['targeting']['county']  # Still 'county' for compatibility
    recommended_bolag_name = next((name for name, code in BOLAG_VALUES.items() if code == recommended_bolag_code), recommended_bolag_code)
    
    report += f"""
    - **Recommended Bolag:** {recommended_bolag_name}
    - **Predicted Open Rate:** {formatted_predictions['targeting']['open_rate']:.2f}% (Change: {formatted_predictions['targeting']['open_rate_diff']:.2f}%)
    - **Predicted Click Rate:** {formatted_predictions['targeting']['click_rate']:.2f}% (Change: {formatted_predictions['targeting']['click_rate_diff']:.2f}%)
    - **Predicted Optout Rate:** {formatted_predictions['targeting']['optout_rate']:.2f}% (Change: {formatted_predictions['targeting']['optout_rate_diff']:.2f}%)
    - **Confidence Levels:** Open {formatted_predictions['targeting']['confidence']['open_rate']:.0f}%, Click {formatted_predictions['targeting']['confidence']['click_rate']:.0f}%, Optout {formatted_predictions['targeting']['confidence']['optout_rate']:.0f}%

    ## Combined Recommendation
    """
    
    # Get the combined recommendation bolag name
    combined_bolag_code = formatted_predictions['combined']['county']  # Still 'county' for compatibility
    combined_bolag_name = next((name for name, code in BOLAG_VALUES.items() if code == combined_bolag_code), combined_bolag_code)
    
    report += f"""
    - **Targeting:** {combined_bolag_name}
    - **Subject:** "{formatted_predictions['combined']['subject']}"
    - **Predicted Open Rate:** {formatted_predictions['combined']['open_rate']:.2f}% (Change: {formatted_predictions['combined']['open_rate_diff']:.2f}%)
    - **Predicted Click Rate:** {formatted_predictions['combined']['click_rate']:.2f}% (Change: {formatted_predictions['combined']['click_rate_diff']:.2f}%)
    - **Predicted Optout Rate:** {formatted_predictions['combined']['optout_rate']:.2f}% (Change: {formatted_predictions['combined']['optout_rate_diff']:.2f}%)
    - **Confidence Levels:** Open {formatted_predictions['combined']['confidence']['open_rate']:.0f}%, Click {formatted_predictions['combined']['confidence']['click_rate']:.0f}%, Optout {formatted_predictions['combined']['confidence']['optout_rate']:.0f}%

    ## Potential Impact
    Implementing the combined recommendations could improve your open rate by {formatted_predictions['combined']['open_rate_diff']:.2f} percentage points,
    which represents a {((formatted_predictions['combined']['open_rate'] - formatted_predictions['current']['open_rate']) / formatted_predictions['current']['open_rate'] * 100) if formatted_predictions['current']['open_rate'] > 0 else 0:.1f}% increase.
    """
    
    return report

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
def create_improved_comparison_table(formatted_predictions):
    """
    Creates an improved metrics comparison table that's easier to read.
    
    Parameters:
    - formatted_predictions: Validated prediction results
    
    Returns:
    - Streamlit table or dataframe
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    
    # Create a dataframe with the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Open Rate (%)', 'Click Rate (%)', 'Optout Rate (%)'],
        'Current': [
            formatted_predictions['current']['open_rate'],
            formatted_predictions['current']['click_rate'],
            formatted_predictions['current']['optout_rate']
        ],
        'Targeting': [
            formatted_predictions['targeting']['open_rate'],
            formatted_predictions['targeting']['click_rate'],
            formatted_predictions['targeting']['optout_rate']
        ],
        'Targeting Δ': [
            formatted_predictions['targeting']['open_rate_diff'],
            formatted_predictions['targeting']['click_rate_diff'],
            formatted_predictions['targeting']['optout_rate_diff']
        ],
        'Subject': [
            formatted_predictions['subject']['open_rate'],
            formatted_predictions['current']['click_rate'],  # Subject only affects open rate
            formatted_predictions['current']['optout_rate']  # Subject only affects open rate
        ],
        'Subject Δ': [
            formatted_predictions['subject']['open_rate_diff'],
            0,  # No change for click rate
            0   # No change for optout rate
        ],
        'Combined': [
            formatted_predictions['combined']['open_rate'],
            formatted_predictions['combined']['click_rate'],
            formatted_predictions['combined']['optout_rate']
        ],
        'Combined Δ': [
            formatted_predictions['combined']['open_rate_diff'],
            formatted_predictions['combined']['click_rate_diff'],
            formatted_predictions['combined']['optout_rate_diff']
        ]
    })
    
    # Format values to show as percentages with 2 decimal places
    formatted_df = metrics_df.copy()
    
    # Format regular metrics columns
    for col in ['Current', 'Targeting', 'Subject', 'Combined']:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%")
    
    # Format delta columns with +/- sign
    for col in ['Targeting Δ', 'Subject Δ', 'Combined Δ']:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
    
    # Create colored cells for deltas
    def color_delta(val):
        """Colors positive deltas green and negative deltas red"""
        try:
            val_num = float(val.replace('%', '').replace('+', ''))
            if val_num > 0:
                return 'background-color: #8fff9c'  # Light green
            elif val_num < 0:
                return 'background-color: #ff9c9c'  # Light red
            return ''
        except:
            return ''
    
    # Apply styling to the dataframe
    styled_df = formatted_df.style.apply(
        lambda x: [''] * len(x) if x.name not in ['Targeting Δ', 'Subject Δ', 'Combined Δ'] 
                else [color_delta(val) for val in x],
        axis=1
    )
    
    return styled_df

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

def validate_data(customer_df, delivery_df):
    """
    Validate the loaded data and report any issues
    
    Parameters:
    - customer_df: DataFrame with customer data
    - delivery_df: DataFrame with delivery data
    
    Returns:
    - Tuple (is_valid, issues_list)
    """
    import pandas as pd
    import numpy as np
    import logging
    
    issues = []
    
    # Check if DataFrames exist
    if customer_df is None or delivery_df is None:
        return False, ["One or both DataFrames are missing"]
    
    # Check if DataFrames are empty
    if len(customer_df) == 0:
        issues.append("Customer data is empty")
    
    if len(delivery_df) == 0:
        issues.append("Delivery data is empty")
        return False, issues
    
    # Check required columns in delivery data
    required_delivery_cols = ['InternalName', 'Subject', 'Sendouts', 'Opens', 'Clicks', 'Optouts']  # Note: Optouts with an 's'
    missing_delivery_cols = [col for col in required_delivery_cols if col not in delivery_df.columns]
    if missing_delivery_cols:
        issues.append(f"Missing required delivery columns: {', '.join(missing_delivery_cols)}")
        
        # Special handling for Optout vs Optouts confusion
        if 'Optouts' not in delivery_df.columns and 'Optout' in delivery_df.columns:
            issues.append("Found 'Optout' column but expected 'Optouts'. Column naming may be inconsistent.")
    
    # Validate numeric columns in delivery data
    numeric_cols = ['Sendouts', 'Opens', 'Clicks', 'Optouts']  # Note: Optouts with an 's'
    for col in numeric_cols:
        if col in delivery_df.columns:
            # Check if column contains non-numeric data
            if not pd.api.types.is_numeric_dtype(delivery_df[col]):
                issues.append(f"Column {col} contains non-numeric data")
            
            # Check if column contains negative values
            if (delivery_df[col] < 0).any():
                issues.append(f"Column {col} contains negative values")
    
    # Check rate calculations
    if 'open_rate' in delivery_df.columns:
        if delivery_df['open_rate'].isna().any():
            issues.append("open_rate contains NaN values")
        if (delivery_df['open_rate'] < 0).any() or (delivery_df['open_rate'] > 100).any():
            issues.append("open_rate contains values outside 0-100 range")
    
    if 'click_rate' in delivery_df.columns:
        if delivery_df['click_rate'].isna().any():
            issues.append("click_rate contains NaN values")
        if (delivery_df['click_rate'] < 0).any() or (delivery_df['click_rate'] > 100).any():
            issues.append("click_rate contains values outside 0-100 range")
    
    if 'optout_rate' in delivery_df.columns:
        if delivery_df['optout_rate'].isna().any():
            issues.append("optout_rate contains NaN values")
        if (delivery_df['optout_rate'] < 0).any() or (delivery_df['optout_rate'] > 100).any():
            issues.append("optout_rate contains values outside 0-100 range")
    
    # Check logical constraints
    if all(col in delivery_df.columns for col in ['Opens', 'Sendouts']):
        if (delivery_df['Opens'] > delivery_df['Sendouts']).any():
            issues.append("Some rows have more Opens than Sendouts")
    
    if all(col in delivery_df.columns for col in ['Clicks', 'Opens']):
        if (delivery_df['Clicks'] > delivery_df['Opens']).any():
            issues.append("Some rows have more Clicks than Opens")
            
    if all(col in delivery_df.columns for col in ['Optouts', 'Opens']):
        if (delivery_df['Optouts'] > delivery_df['Opens']).any():
            issues.append("Some rows have more Optouts than Opens")
    
    # Log all issues
    for issue in issues:
        logging.warning(f"Data validation issue: {issue}")
    
    # Return results
    is_valid = len(issues) == 0
    return is_valid, issues

def main():
    """
    Updated main function with all the enhancements:
    - Remove Target Bolag field (only exclude bolags)
    - Improved targeting recommendations
    - Age group analytics
    - Daily trend charts
    - Best/worst timing analysis
    - KPI dashboard
    - Forecast tab
    """
    # Header & Intro
    st.title("📧 Email Campaign KPI Predictor")
    
    st.markdown("""This tool uses machine learning to predict email campaign performance and provides 
    recommendations for targeting and subject lines to improve your KPIs.
    
    - **Subject Line Recommendations**: Optimize for open rates only
    - **Targeting Recommendations**: Optimize for open, click, and optout rates
    - **Timing Optimization**: Identify best days and times to send
    - **Performance Forecasting**: Estimate campaign ROI and reach
    """)

    # Initialize session state for force_retrain if not exists
    if 'force_retrain' not in st.session_state:
        st.session_state['force_retrain'] = False

    # Load data
    with st.spinner("Processing data..."):
        customer_df, delivery_df = load_data()

    if customer_df is None or delivery_df is None:
        st.error("Failed to load data. Please check file paths and formats.")
        return

    # Validate the data
    is_valid, issues = validate_data(customer_df, delivery_df)
    if not is_valid:
        st.warning("⚠️ Data validation found issues:")
        for issue in issues:
            st.write(f"- {issue}")
        st.write("The application will try to continue, but results may be affected.")

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

    # Perform time and age group analysis once
    with st.spinner("Analyzing historical data patterns..."):
        # Time-based analysis
        time_analysis = analyze_time_patterns(delivery_df)
        
        # Age group analysis
        age_group_metrics, age_group_fig = analyze_age_groups(customer_df, delivery_df)
        best_age_groups = get_best_age_groups(age_group_metrics)

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Campaign Predictor", 
        "Performance Insights", 
        "Time Analysis", 
        "Audience Analysis",
        "KPI Dashboard",
        "Forecast"
    ])

    # Tab 1: Campaign Predictor
    with tab1:
        st.header("Campaign Parameter Input")

        # Get values from models for dropdowns
        cat_values = model_results.get('categorical_values', {})
        
        # Use the updated campaign parameter input function without target bolag
        parameters = campaign_parameter_input(cat_values)
        
        # Create input data for prediction
        input_data = pd.DataFrame({
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

        # Handle excluded bolags
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

            # Validate predictions to avoid zeros and unrealistic values
            formatted_predictions = validate_predictions(formatted_predictions)

        # Calculate confidence scores if not already present
        if 'confidence' not in formatted_predictions['current']:
            # Add confidence to each prediction set
            for scenario in ['current', 'targeting', 'subject', 'combined']:
                formatted_predictions[scenario]['confidence'] = {}
                for metric in ['open_rate', 'click_rate', 'optout_rate']:
                    if metric in formatted_predictions[scenario]:
                        formatted_predictions[scenario]['confidence'][metric] = 85  # Default confidence

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

            # Use the improved targeting recommendations display
            display_targeting_recommendations(
                formatted_predictions, 
                parameters, 
                time_analysis, 
                age_group_metrics
            )

        # Display combined recommendation
        st.subheader("Combined Optimization Strategy")
        st.success(
            f"**Recommended Strategy:** Target all bolags (except excluded ones), " +
            f"use subject '{formatted_predictions['combined']['subject']}', " +
            f"and send at the optimal time"
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

        # When displaying detailed metrics comparison, use improved table
        st.header("Detailed Metrics Comparison")
        styled_df = create_improved_comparison_table(formatted_predictions)
        st.dataframe(styled_df, use_container_width=True)
                
        # Add model management at the bottom
        display_model_management(model_results)
        
    # Tab 2: Performance Insights
    with tab2:
        st.header("Campaign Performance Insights")

        # Monthly open rate performance - protect against missing columns
        try:
            # First ensure Date is datetime
            if 'Date' in delivery_df.columns:
                delivery_df['Date'] = pd.to_datetime(delivery_df['Date'])
                delivery_df['month'] = delivery_df['Date'].dt.strftime('%Y-%m')
            
                monthly_opens = delivery_df.groupby('month').agg(
                    avg_open_rate=('open_rate', 'mean'),
                    total_sends=('Sendouts', 'sum') if 'Sendouts' in delivery_df.columns else None,
                    total_opens=('Opens', 'sum') if 'Opens' in delivery_df.columns else None
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
            else:
                st.warning("Date column is missing. Cannot perform monthly analysis.")
        except Exception as e:
            st.warning(f"Error in monthly open rate analysis: {str(e)}")

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
            
            metric_tab1, metric_tab2, metric_tab3, bolag_tab = st.tabs(["Open Rate", "Click Rate", "Optout Rate", "Company Analysis"])
            
            with metric_tab1:
                # Monthly click rate performance
                if 'Date' in delivery_df.columns and 'month' in delivery_df.columns:
                    monthly_clicks = delivery_df.groupby('month').agg(
                        avg_click_rate=('click_rate', 'mean'),
                        total_opens=('Opens', 'sum') if 'Opens' in delivery_df.columns else None,
                        total_clicks=('Clicks', 'sum') if 'Clicks' in delivery_df.columns else None
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

            with metric_tab2:
                # Monthly optout rate performance
                if 'Date' in delivery_df.columns and 'month' in delivery_df.columns:
                    monthly_optouts = delivery_df.groupby('month').agg(
                        avg_optout_rate=('optout_rate', 'mean'),
                        total_opens=('Opens', 'sum') if 'Opens' in delivery_df.columns else None,
                        total_optouts=('Optouts', 'sum') if 'Optouts' in delivery_df.columns else None
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

            with metric_tab3:
                # Company (Bolag) analysis tab
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

    # Tab 3: Time Analysis (New)
    with tab3:
        st.header("Campaign Timing Analysis")
        
        if 'fig_days' in time_analysis and 'fig_hours' in time_analysis and 'fig_daily' in time_analysis:
            st.subheader("Daily Performance Trends")
            
            # Display daily trends chart
            st.plotly_chart(time_analysis['fig_daily'], use_container_width=True)
            
            timing_col1, timing_col2 = st.columns(2)
            
            with timing_col1:
                st.subheader("Performance by Day of Week")
                st.plotly_chart(time_analysis['fig_days'], use_container_width=True)
            
            with timing_col2:
                st.subheader("Performance by Hour of Day")
                st.plotly_chart(time_analysis['fig_hours'], use_container_width=True)
            
            # Display best/worst times in an expandable section
            with st.expander("Best and Worst Send Times"):
                best_worst = time_analysis.get('best_worst', {})
                
                if best_worst:
                    best_worst_tab1, best_worst_tab2, best_worst_tab3 = st.tabs([
                        "Open Rate Timing", "Click Rate Timing", "Optout Rate Timing"
                    ])
                    
                    with best_worst_tab1:
                        if 'open_rate' in best_worst:
                            open_timing = best_worst['open_rate']
                            
                            timing_col1, timing_col2 = st.columns(2)
                            
                            with timing_col1:
                                st.subheader("Best Times for Opens")
                                st.metric("Best Day", open_timing['best_day'])
                                st.metric("Open Rate", f"{open_timing['best_day_rate']:.2f}%")
                                st.metric("Best Hour", f"{open_timing['best_hour']}:00")
                                st.metric("Open Rate", f"{open_timing['best_hour_rate']:.2f}%")
                            
                            with timing_col2:
                                st.subheader("Worst Times for Opens")
                                st.metric("Worst Day", open_timing['worst_day'])
                                st.metric("Open Rate", f"{open_timing['worst_day_rate']:.2f}%")
                                st.metric("Worst Hour", f"{open_timing['worst_hour']}:00")
                                st.metric("Open Rate", f"{open_timing['worst_hour_rate']:.2f}%")
                    
                    with best_worst_tab2:
                        if 'click_rate' in best_worst:
                            click_timing = best_worst['click_rate']
                            
                            timing_col1, timing_col2 = st.columns(2)
                            
                            with timing_col1:
                                st.subheader("Best Times for Clicks")
                                st.metric("Best Day", click_timing['best_day'])
                                st.metric("Click Rate", f"{click_timing['best_day_rate']:.2f}%")
                                st.metric("Best Hour", f"{click_timing['best_hour']}:00")
                                st.metric("Click Rate", f"{click_timing['best_hour_rate']:.2f}%")
                            
                            with timing_col2:
                                st.subheader("Worst Times for Clicks")
                                st.metric("Worst Day", click_timing['worst_day'])
                                st.metric("Click Rate", f"{click_timing['worst_day_rate']:.2f}%")
                                st.metric("Worst Hour", f"{click_timing['worst_hour']}:00")
                                st.metric("Click Rate", f"{click_timing['worst_hour_rate']:.2f}%")
                    
                    with best_worst_tab3:
                        if 'optout_rate' in best_worst:
                            optout_timing = best_worst['optout_rate']
                            
                            timing_col1, timing_col2 = st.columns(2)
                            
                            with timing_col1:
                                st.subheader("Best Times for Retention")
                                st.metric("Best Day", optout_timing['best_day'])
                                st.metric("Optout Rate", f"{optout_timing['best_day_rate']:.2f}%")
                                st.metric("Best Hour", f"{optout_timing['best_hour']}:00")
                                st.metric("Optout Rate", f"{optout_timing['best_hour_rate']:.2f}%")
                            
                            with timing_col2:
                                st.subheader("Worst Times for Retention")
                                st.metric("Worst Day", optout_timing['worst_day'])
                                st.metric("Optout Rate", f"{optout_timing['worst_day_rate']:.2f}%")
                                st.metric("Worst Hour", f"{optout_timing['worst_hour']}:00")
                                st.metric("Optout Rate", f"{optout_timing['worst_hour_rate']:.2f}%")
        else:
            st.warning("Unable to perform time analysis. Please check your data to ensure it includes dates and times.")

    # Tab 4: Audience Analysis (New)
    with tab4:
        st.header("Audience Analysis")
        
        # Age group analysis
        st.subheader("Campaign Performance by Age Group")
        
        if age_group_fig is not None:
            st.plotly_chart(age_group_fig, use_container_width=True)
            
            if best_age_groups:
                st.subheader("Best Performing Age Groups")
                
                age_col1, age_col2, age_col3 = st.columns(3)
                
                with age_col1:
                    if 'open_rate' in best_age_groups:
                        st.metric("Best Age Group for Opens", best_age_groups['open_rate']['group'])
                        st.metric("Open Rate", f"{best_age_groups['open_rate']['rate']:.2f}%")
                
                with age_col2:
                    if 'click_rate' in best_age_groups:
                        st.metric("Best Age Group for Clicks", best_age_groups['click_rate']['group'])
                        st.metric("Click Rate", f"{best_age_groups['click_rate']['rate']:.2f}%")
                
                with age_col3:
                    if 'optout_rate' in best_age_groups:
                        st.metric("Best Age Group for Retention", best_age_groups['optout_rate']['group'])
                        st.metric("Optout Rate", f"{best_age_groups['optout_rate']['rate']:.2f}%")
            
            if age_group_metrics is not None:
                st.subheader("Age Group Performance Data")
                st.dataframe(
                    age_group_metrics.style.format({
                        'avg_open_rate': '{:.2f}%',
                        'avg_click_rate': '{:.2f}%',
                        'avg_optout_rate': '{:.2f}%'
                    })
                )
        else:
            st.warning("Unable to perform age group analysis. Please check your data to ensure it includes age information.")
        
        # Gender analysis (if data is available)
        st.subheader("Campaign Performance by Gender")
        
        if 'Gender' in customer_df.columns:
            # Merge customer data with delivery data
            delivery_metrics = delivery_df[['InternalName', 'open_rate', 'click_rate', 'optout_rate']].copy()
            merged_data = customer_df.merge(delivery_metrics, on='InternalName', how='left')
            
            # Group by gender
            gender_metrics = merged_data.groupby('Gender').agg(
                avg_open_rate=('open_rate', 'mean'),
                avg_click_rate=('click_rate', 'mean'),
                avg_optout_rate=('optout_rate', 'mean'),
                count=('InternalName', 'count')
            ).reset_index()
            
            # Fill NaN values
            gender_metrics.fillna(0, inplace=True)
            
            # Create visualization
            metrics_data = []
            for _, row in gender_metrics.iterrows():
                metrics_data.extend([
                    {'Gender': row['Gender'], 'Metric': 'Open Rate', 'Value': row['avg_open_rate'], 'Count': row['count']},
                    {'Gender': row['Gender'], 'Metric': 'Click Rate', 'Value': row['avg_click_rate'], 'Count': row['count']},
                    {'Gender': row['Gender'], 'Metric': 'Optout Rate', 'Value': row['avg_optout_rate'], 'Count': row['count']}
                ])
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Create visualization
            fig_gender = px.bar(
                metrics_df,
                x='Gender',
                y='Value',
                color='Metric',
                barmode='group',
                text=metrics_df['Value'].round(2).astype(str) + '%',
                title='Campaign Performance by Gender',
                hover_data=['Count'],
                labels={'Value': 'Rate (%)', 'Gender': 'Gender'}
            )
            
            fig_gender.update_traces(textposition='outside')
            
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Display data table
            st.subheader("Gender Performance Data")
            st.dataframe(
                gender_metrics.style.format({
                    'avg_open_rate': '{:.2f}%',
                    'avg_click_rate': '{:.2f}%',
                    'avg_optout_rate': '{:.2f}%'
                })
            )
        else:
            st.warning("Gender information not found in customer data. Unable to perform gender analysis.")

    # Tab 5: KPI Dashboard (New)
    with tab5:
        display_kpi_dashboard(formatted_predictions, delivery_df)

    # Tab 6: Forecast (New)
    with tab6:
        create_forecast_tab(customer_df, delivery_df, formatted_predictions, parameters, BOLAG_VALUES)

if __name__ == "__main__":
    main()