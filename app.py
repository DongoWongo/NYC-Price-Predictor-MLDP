"""NYC Real Estate Sale Price Predictor -- Streamlit Application."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="NYC Property Price Predictor",
    page_icon="\U0001F3D9",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme
st.markdown(
    """
    <style>
    /* Main prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #1a1d23 0%, #2d3139 100%);
        border: 1px solid #FF6B35;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-price {
        font-size: 3rem;
        font-weight: 700;
        color: #FF6B35;
        margin: 0.5rem 0;
    }
    .prediction-range {
        font-size: 1rem;
        color: #aaaaaa;
        margin-top: 0.5rem;
    }
    .prediction-label {
        font-size: 1.1rem;
        color: #cccccc;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1A1D23;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #FAFAFA;
    }

    /* Section dividers */
    .section-header {
        color: #FF6B35;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #333;
    }

    /* Info boxes */
    .info-box {
        background-color: #1A1D23;
        border-left: 3px solid #FF6B35;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #cccccc;
    }

    /* Feature importance bars */
    .feature-bar {
        background: linear-gradient(90deg, #FF6B35, #FF8F65);
        border-radius: 4px;
        height: 22px;
        margin: 2px 0;
    }

    /* Hide default streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Load model (cached so they load only once)
@st.cache_resource
def load_model():
    return joblib.load("model/rf_model.joblib")


@st.cache_resource
def load_artefacts():
    feature_columns = joblib.load("model/feature_columns.joblib")
    impute_medians = joblib.load("model/impute_medians.joblib")
    lookup = joblib.load("model/lookup_tables.joblib")
    return feature_columns, impute_medians, lookup


model = load_model()
feature_columns, impute_medians, lookup = load_artefacts()

BOROUGH_MAP = lookup["borough_map"]
BOROUGH_REVERSE = {v: k for k, v in BOROUGH_MAP.items()}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# Feature engineering, same as ipynb pipeline
def build_feature_vector(
    borough, neighborhood, zip_code, building_category,
    tax_class_present, tax_class_sale, residential_units,
    commercial_units, gross_sqft, land_sqft, year_built, sale_month,
):
    """Construct the 22-feature vector from user inputs."""

    # Total units
    total_units = residential_units + commercial_units

    # Cap units at training 99th percentile
    residential_units = min(residential_units, lookup["residential_units_cap"])
    commercial_units = min(commercial_units, lookup["commercial_units_cap"])
    total_units = min(total_units, lookup["total_units_cap"])

    # Handle optional square footage (0 means unknown > NaN > median imputed)
    gross_sqft_val = gross_sqft if gross_sqft > 0 else np.nan
    land_sqft_val = land_sqft if land_sqft > 0 else np.nan

    # Temporal features
    sale_quarter = (sale_month - 1) // 3 + 1

    # Building age (capped at 200 years, relative to 2017 training period)
    building_age = min(2017 - year_built, 200)
    age_squared = building_age ** 2

    # Binary indicators
    is_manhattan = 1 if borough == 1 else 0
    has_commercial = 1 if commercial_units > 0 else 0
    is_multi_unit = 1 if total_units > 1 else 0

    # Log gross square feet (NaN-safe)
    log_gross_sqft = np.log1p(gross_sqft_val) if not np.isnan(gross_sqft_val) else np.nan

    # Location-based median prices from training data
    global_neigh_median = float(np.median(list(lookup["neigh_median_price"].values())))
    global_zip_median = float(np.median(list(lookup["zip_median_price"].values())))
    neigh_median_price = lookup["neigh_median_price"].get(neighborhood, global_neigh_median)
    zip_median_price = lookup["zip_median_price"].get(zip_code, global_zip_median)

    # Frequency encodings from training data
    borough_freq = lookup["borough_freq"].get(borough, 0.0)
    neighborhood_freq = lookup["neighborhood_freq"].get(neighborhood, 0.0)
    zip_code_freq = lookup["zip_code_freq"].get(zip_code, 0.0)
    building_class_category_freq = lookup["building_class_category_freq"].get(building_category, 0.0)
    tax_class_at_present_freq = lookup["tax_class_at_present_freq"].get(tax_class_present, 0.0)
    tax_class_at_time_of_sale_freq = lookup["tax_class_at_time_of_sale_freq"].get(tax_class_sale, 0.0)

    # Build df with exact column order from training
    row = {
        "BOROUGH": borough,
        "RESIDENTIAL UNITS": residential_units,
        "COMMERCIAL UNITS": commercial_units,
        "TOTAL UNITS": total_units,
        "LAND SQUARE FEET": land_sqft_val,
        "GROSS SQUARE FEET": gross_sqft_val,
        "sale_month": sale_month,
        "sale_quarter": sale_quarter,
        "building_age": building_age,
        "age_squared": age_squared,
        "is_manhattan": is_manhattan,
        "has_commercial": has_commercial,
        "is_multi_unit": is_multi_unit,
        "log_gross_sqft": log_gross_sqft,
        "neigh_median_price": neigh_median_price,
        "zip_median_price": zip_median_price,
        "borough_freq": borough_freq,
        "neighborhood_freq": neighborhood_freq,
        "zip_code_freq": zip_code_freq,
        "building_class_category_freq": building_class_category_freq,
        "tax_class_at_present_freq": tax_class_at_present_freq,
        "tax_class_at_time_of_sale_freq": tax_class_at_time_of_sale_freq,
    }

    X = pd.DataFrame([row], columns=feature_columns)

    # Median imputation for NaN values
    X = X.fillna(pd.Series(impute_medians))

    return X


# Sidebar inputs
st.sidebar.markdown("## Property Details")

# Location
st.sidebar.markdown('<div class="section-header">Location</div>', unsafe_allow_html=True)

borough_name = st.sidebar.selectbox(
    "Borough",
    options=list(BOROUGH_REVERSE.keys()),
    index=3,  # Default: Queens (most training data)
)
borough = BOROUGH_REVERSE[borough_name]

# Neighbourhoods filtered by borough
available_neighborhoods = lookup["borough_neighborhoods"].get(borough, [])
neighborhood = st.sidebar.selectbox("Neighbourhood", options=available_neighborhoods)

# ZIP codes filtered by borough
available_zips = lookup["borough_zipcodes"].get(borough, [])
zip_code = st.sidebar.selectbox("ZIP Code", options=available_zips)

# Property Type
st.sidebar.markdown('<div class="section-header">Property Type</div>', unsafe_allow_html=True)

building_category = st.sidebar.selectbox(
    "Building Class Category",
    options=lookup["building_categories"],
)

tax_class_present = st.sidebar.selectbox(
    "Tax Class at Present",
    options=lookup["tax_class_at_present_options"],
)

tax_class_sale = st.sidebar.selectbox(
    "Tax Class at Time of Sale",
    options=lookup["tax_class_at_time_of_sale_options"],
)

# Property Details
st.sidebar.markdown('<div class="section-header">Size & Age</div>', unsafe_allow_html=True)

residential_units = st.sidebar.number_input(
    "Residential Units", min_value=0, max_value=500, value=1, step=1,
)
commercial_units = st.sidebar.number_input(
    "Commercial Units", min_value=0, max_value=100, value=0, step=1,
)
gross_sqft = st.sidebar.number_input(
    "Gross Square Feet", min_value=0, max_value=1_000_000, value=1000, step=100,
    help="Set to 0 if unknown. The model will assume a typical property size.",
)
land_sqft = st.sidebar.number_input(
    "Land Square Feet", min_value=0, max_value=1_000_000, value=0, step=100,
    help="Set to 0 if unknown. The model will assume a typical lot size.",
)
year_built = st.sidebar.slider(
    "Year Built", min_value=1800, max_value=2017, value=1960,
)

# Sale Timing
st.sidebar.markdown('<div class="section-header">Sale Timing</div>', unsafe_allow_html=True)
sale_month = st.sidebar.slider("Sale Month", min_value=1, max_value=12, value=6)
st.sidebar.caption(f"Selected: {MONTH_NAMES[sale_month]}")


# Main content area
st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <h1 style="color: #FF6B35; margin-bottom: 0;">NYC Property Price Predictor</h1>
        <p style="color: #888; font-size: 1.1rem; margin-top: 0.3rem;">
            Machine learning model trained on 54,000+ NYC property sales (2016-2017)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Build features and predict (predict on every input change)
X = build_feature_vector(
    borough, neighborhood, zip_code, building_category,
    tax_class_present, tax_class_sale, residential_units,
    commercial_units, gross_sqft, land_sqft, year_built, sale_month,
)

# Point prediction
log_pred = model.predict(X)[0]
log_pred = np.clip(log_pred, 5, 25)
predicted_price = np.exp(log_pred)

# Prediction range from individual trees (10th to 90th percentile)
tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
tree_preds = np.clip(tree_preds, 5, 25)
tree_prices = np.exp(tree_preds)
price_low = np.percentile(tree_prices, 10)
price_high = np.percentile(tree_prices, 90)

# Display prediction
st.markdown(
    f"""
    <div class="prediction-card">
        <div class="prediction-label">Estimated Sale Price</div>
        <div class="prediction-price">${predicted_price:,.0f}</div>
        <div class="prediction-range">
            Prediction range: ${price_low:,.0f} &ndash; ${price_high:,.0f}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Capping warnings
warnings_list = []
if residential_units > lookup["residential_units_cap"]:
    warnings_list.append(
        f"Residential units capped from {residential_units} to "
        f"{int(lookup['residential_units_cap'])} (training data limit)"
    )
if commercial_units > lookup["commercial_units_cap"]:
    warnings_list.append(
        f"Commercial units capped from {commercial_units} to "
        f"{int(lookup['commercial_units_cap'])} (training data limit)"
    )
if gross_sqft == 0:
    warnings_list.append(
        "Gross square feet unknown -- model uses the training median "
        f"({int(impute_medians.get('GROSS SQUARE FEET', 0)):,} sqft)"
    )

if warnings_list:
    for w in warnings_list:
        st.info(w)

# Two-column layout with Input summary + Feature importance
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Input Summary")
    summary_data = {
        "Borough": borough_name,
        "Neighbourhood": neighborhood,
        "ZIP Code": str(zip_code),
        "Building Class": building_category,
        "Tax Class (Present)": str(tax_class_present),
        "Tax Class (Sale)": str(tax_class_sale),
        "Residential Units": str(residential_units),
        "Commercial Units": str(commercial_units),
        "Gross Sq Ft": f"{gross_sqft:,}" if gross_sqft > 0 else "Unknown",
        "Land Sq Ft": f"{land_sqft:,}" if land_sqft > 0 else "Unknown",
        "Year Built": str(year_built),
        "Building Age": f"{min(2017 - year_built, 200)} years",
        "Sale Month": MONTH_NAMES[sale_month],
    }
    summary_df = pd.DataFrame(
        list(summary_data.items()), columns=["Property Detail", "Value"]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### Feature Importance")
    st.caption("Top 10 features driving the model's predictions")

    importances = pd.Series(model.feature_importances_, index=feature_columns)
    top_features = importances.sort_values(ascending=True).tail(10)

    # Rename features for display
    display_names = {
        "neigh_median_price": "Neighbourhood Median Price",
        "zip_median_price": "ZIP Median Price",
        "log_gross_sqft": "Log(Gross Sq Ft)",
        "GROSS SQUARE FEET": "Gross Square Feet",
        "LAND SQUARE FEET": "Land Square Feet",
        "building_age": "Building Age",
        "age_squared": "Building Age (squared)",
        "BOROUGH": "Borough",
        "is_manhattan": "Is Manhattan",
        "RESIDENTIAL UNITS": "Residential Units",
        "COMMERCIAL UNITS": "Commercial Units",
        "TOTAL UNITS": "Total Units",
        "borough_freq": "Borough Frequency",
        "neighborhood_freq": "Neighbourhood Frequency",
        "zip_code_freq": "ZIP Code Frequency",
        "building_class_category_freq": "Building Class Frequency",
        "tax_class_at_present_freq": "Tax Class (Present) Frequency",
        "tax_class_at_time_of_sale_freq": "Tax Class (Sale) Frequency",
        "sale_month": "Sale Month",
        "sale_quarter": "Sale Quarter",
        "has_commercial": "Has Commercial Units",
        "is_multi_unit": "Multi-Unit Property",
    }

    chart_data = pd.DataFrame({
        "Feature": [display_names.get(f, f) for f in top_features.index],
        "Importance": top_features.values,
    })
    chart_data = chart_data.set_index("Feature")

    st.bar_chart(chart_data, color="#FF6B35", horizontal=True)

# Model information
st.markdown("---")

with st.expander("About This Model"):
    st.markdown(
        """
        **Algorithm:** Random Forest Regressor (scikit-learn)

        | Parameter | Value |
        |---|---|
        | Number of trees | 300 |
        | Max depth | 20 |
        | Min samples leaf | 1 |
        | Min samples split | 10 |

        **Training data:** 43,411 NYC property sales from 2016-2017
        (after cleaning non-market transfers, capping outliers at the 99th percentile,
        and removing properties with missing construction year).

        **Test set performance (dollar scale):**
        | Metric | Value |
        |---|---|
        | R-squared | 0.6332 |
        | RMSE | $1,215,329 |
        | MAE | $440,628 |
        | MAPE | 57.5% |

        **Notes:**
        - Building age is computed relative to 2017 (the training data period),
          so the maximum selectable year is 2017.
        - When square footage is unknown (set to 0), the model uses the
          training set median value for imputation.
        - The prediction range shown is the 10th-90th percentile spread
          across the 300 individual trees in the forest.
        - Source: NYC Department of Finance Rolling Sales Data (Kaggle).
        """
    )

st.caption("Data: NYC Dept. of Finance Rolling Sales (2016-2017).")
