import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import hashlib
from datetime import datetime, timedelta
import random
import json

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from persistent storage"""
    try:
        result = st.session_state.get('storage_users')
        if result is None:
            # Try to load from persistent storage
            import asyncio
            result = asyncio.run(load_users_async())
        return result
    except:
        # Default users if storage fails
        return {
            "doctor1": {
                "password": hash_password("doc123"),
                "role": "Doctor"
            },
            "nurse1": {
                "password": hash_password("nurse123"),
                "role": "Nurse"
            },
            "admin": {
                "password": hash_password("admin123"),
                "role": "Admin"
            }
        }

async def load_users_async():
    """Async function to load users"""
    try:
        result = await st.session_state.window.storage.get('hospital_users')
        if result and result.get('value'):
            return json.loads(result['value'])
    except:
        pass
    
    # Return default users
    return {
        "doctor1": {
            "password": hash_password("doc123"),
            "role": "Doctor"
        },
        "nurse1": {
            "password": hash_password("nurse123"),
            "role": "Nurse"
        },
        "admin": {
            "password": hash_password("admin123"),
            "role": "Admin"
        }
    }

def save_users(users_dict):
    """Save users to persistent storage"""
    try:
        import asyncio
        asyncio.run(save_users_async(users_dict))
        st.session_state.storage_users = users_dict
        return True
    except Exception as e:
        st.error(f"Error saving user: {str(e)}")
        return False

async def save_users_async(users_dict):
    """Async function to save users"""
    try:
        await st.session_state.window.storage.set('hospital_users', json.dumps(users_dict))
    except Exception as e:
        raise e

def verify_login(username, password, users_db):
    """Verify user credentials"""
    if username in users_db:
        if users_db[username]["password"] == hash_password(password):
            return True, users_db[username]["role"]
    return False, None

@st.cache_data
def generate_training_data():
    """Generate synthetic patient data for training"""
    np.random.seed(42)
    n_samples = 500
    
    # Generate features
    ages = np.random.randint(18, 90, n_samples)
    severity = np.random.choice(['Mild', 'Moderate', 'Severe', 'Critical'], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    departments = np.random.choice(['Emergency', 'Surgery', 'ICU', 'General Ward'], n_samples, p=[0.25, 0.25, 0.15, 0.35])
    num_comorbidities = np.random.randint(0, 5, n_samples)
    previous_admissions = np.random.randint(0, 8, n_samples)
    
    # Generate length of stay based on features (with some logic)
    los = []
    for i in range(n_samples):
        base_los = 3
        
        # Age factor
        if ages[i] > 70:
            base_los += 2
        elif ages[i] < 30:
            base_los += 1
            
        # Severity factor
        if severity[i] == 'Critical':
            base_los += 8
        elif severity[i] == 'Severe':
            base_los += 5
        elif severity[i] == 'Moderate':
            base_los += 2
            
        # Department factor
        if departments[i] == 'ICU':
            base_los += 6
        elif departments[i] == 'Surgery':
            base_los += 3
            
        # Comorbidities
        base_los += num_comorbidities[i] * 1.5
        
        # Previous admissions
        base_los += previous_admissions[i] * 0.5
        
        # Add some randomness
        base_los += np.random.normal(0, 2)
        
        los.append(max(1, int(base_los)))
    
    df = pd.DataFrame({
        'Age': ages,
        'Severity': severity,
        'Department': departments,
        'Num_Comorbidities': num_comorbidities,
        'Previous_Admissions': previous_admissions,
        'Length_of_Stay': los
    })
    
    return df

@st.cache_resource
def train_model():
    """Train the prediction model"""
    df = generate_training_data()
    
    # Encode categorical variables
    le_severity = LabelEncoder()
    le_department = LabelEncoder()
    
    df_encoded = df.copy()
    df_encoded['Severity_Encoded'] = le_severity.fit_transform(df['Severity'])
    df_encoded['Department_Encoded'] = le_department.fit_transform(df['Department'])
    
    # Features and target
    X = df_encoded[['Age', 'Severity_Encoded', 'Department_Encoded', 'Num_Comorbidities', 'Previous_Admissions']]
    y = df_encoded['Length_of_Stay']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, le_severity, le_department, df

def login_page():
    """Display login page"""
    st.title("🏥 Hospital Length of Stay Predictor")
    
    # Load users
    users_db = load_users()
    
    # Tabs for Login and Signup
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        st.markdown("### Login to Your Account")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    is_valid, role = verify_login(username, password, users_db)
                    if is_valid:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = role
                        st.success(f"✅ Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
            
            st.info("""
            **Demo Accounts:**
            
            👨‍⚕️ `doctor1` / `doc123`
            
            👩‍⚕️ `nurse1` / `nurse123`
            
            👤 `admin` / `admin123`
            """)
    
    with tab2:
        st.markdown("### Create New Account")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username", placeholder="Enter username")
                new_password = st.text_input("Choose Password", type="password", placeholder="Min 6 characters")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                role = st.selectbox("Select Role", ["Doctor", "Nurse", "Admin", "Staff"])
                
                signup_submit = st.form_submit_button("Create Account", use_container_width=True)
                
                if signup_submit:
                    # Validation
                    if not new_username or not new_password:
                        st.error("❌ Please fill in all fields")
                    elif new_username in users_db:
                        st.error("❌ Username already exists! Please choose another.")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords don't match!")
                    else:
                        # Add new user
                        users_db[new_username] = {
                            "password": hash_password(new_password),
                            "role": role
                        }
                        
                        # Save to persistent storage
                        if save_users(users_db):
                            st.success(f"✅ Account created successfully! Welcome, {new_username}!")
                            st.balloons()
                            st.info("👉 Go to the **Login** tab to sign in with your new account")
                        else:
                            st.warning("⚠️ Account created but may not persist. Please note your credentials.")

def predict_los(age, severity, department, num_comorbidities, previous_admissions, model, le_severity, le_department):
    """Make prediction for patient length of stay"""
    # Encode inputs
    severity_encoded = le_severity.transform([severity])[0]
    department_encoded = le_department.transform([department])[0]
    
    # Create feature array
    features = np.array([[age, severity_encoded, department_encoded, num_comorbidities, previous_admissions]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    return max(1, int(round(prediction)))

def patient_entry_form():
    """Form for entering new patient data"""
    st.subheader("📋 Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID", placeholder="e.g., P12345")
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        severity = st.selectbox("Condition Severity", ['Mild', 'Moderate', 'Severe', 'Critical'])
        
    with col2:
        department = st.selectbox("Department", ['Emergency', 'Surgery', 'ICU', 'General Ward'])
        num_comorbidities = st.number_input("Number of Comorbidities", min_value=0, max_value=10, value=1)
        previous_admissions = st.number_input("Previous Admissions", min_value=0, max_value=20, value=0)
    
    return patient_id, age, severity, department, num_comorbidities, previous_admissions

def main_dashboard():
    """Main dashboard after login"""
    st.title("🏥 Hospital Length of Stay Prediction System")
    
    # Header with user info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Logged in as:** {st.session_state.username} ({st.session_state.role})")
    with col3:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()
    
    st.markdown("---")
    
    # Load model
    model, le_severity, le_department, training_data = train_model()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 New Prediction", "📊 Patient History", "📈 Model Info"])
    
    with tab1:
        st.markdown("### Predict Patient Length of Stay")
        
        # Patient entry form
        patient_id, age, severity, department, num_comorbidities, previous_admissions = patient_entry_form()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🔮 Predict Length of Stay", type="primary", use_container_width=True):
                if not patient_id:
                    st.error("Please enter a Patient ID")
                else:
                    # Make prediction
                    predicted_days = predict_los(age, severity, department, num_comorbidities, previous_admissions, model, le_severity, le_department)
                    
                    # Calculate estimated discharge date
                    admission_date = datetime.now()
                    discharge_date = admission_date + timedelta(days=predicted_days)
                    
                    # Display results
                    st.success("### Prediction Complete! ✅")
                    
                    # Result cards
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        st.metric("Predicted Length of Stay", f"{predicted_days} days")
                    
                    with res_col2:
                        st.metric("Admission Date", admission_date.strftime("%Y-%m-%d"))
                    
                    with res_col3:
                        st.metric("Est. Discharge Date", discharge_date.strftime("%Y-%m-%d"))
                    
                    # Patient summary
                    st.markdown("---")
                    st.markdown("#### 📋 Patient Summary")
                    summary_df = pd.DataFrame({
                        'Field': ['Patient ID', 'Age', 'Severity', 'Department', 'Comorbidities', 'Previous Admissions'],
                        'Value': [patient_id, age, severity, department, num_comorbidities, previous_admissions]
                    })
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Risk assessment
                    if predicted_days <= 3:
                        risk_level = "🟢 Low Risk - Short Stay Expected"
                    elif predicted_days <= 7:
                        risk_level = "🟡 Moderate Risk - Standard Stay"
                    elif predicted_days <= 14:
                        risk_level = "🟠 High Risk - Extended Stay"
                    else:
                        risk_level = "🔴 Critical Risk - Long-term Care Needed"
                    
                    st.info(f"**Risk Assessment:** {risk_level}")
    
    with tab2:
        st.subheader("📊 Recent Predictions")
        
        # Show sample patient data
        sample_data = training_data.sample(10).sort_values('Length_of_Stay', ascending=False)
        sample_data['Patient_ID'] = [f"P{random.randint(10000, 99999)}" for _ in range(10)]
        
        display_cols = ['Patient_ID', 'Age', 'Severity', 'Department', 'Num_Comorbidities', 'Previous_Admissions', 'Length_of_Stay']
        st.dataframe(sample_data[display_cols], use_container_width=True, hide_index=True)
        
        # Statistics
        st.markdown("---")
        st.subheader("📈 Statistics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Avg Length of Stay", f"{training_data['Length_of_Stay'].mean():.1f} days")
        with stat_col2:
            st.metric("Min Stay", f"{training_data['Length_of_Stay'].min()} days")
        with stat_col3:
            st.metric("Max Stay", f"{training_data['Length_of_Stay'].max()} days")
        with stat_col4:
            st.metric("Total Patients", len(training_data))
    
    with tab3:
        st.subheader("🤖 Model Information")
        
        st.write("**Algorithm:** Random Forest Regressor")
        st.write("**Training Samples:** 500 patients")
        st.write("**Features Used:**")
        st.markdown("""
        - Patient Age
        - Condition Severity (Mild, Moderate, Severe, Critical)
        - Department (Emergency, Surgery, ICU, General Ward)
        - Number of Comorbidities
        - Previous Hospital Admissions
        """)
        
        st.markdown("---")
        st.subheader("📊 Feature Importance")
        
        # Feature importance
        feature_names = ['Age', 'Severity', 'Department', 'Comorbidities', 'Previous Admissions']
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))
        
        st.info("💡 **Note:** This is a demonstration model using synthetic data. In production, this would be trained on real historical patient data.")

def main():
    """Main app entry point"""
    st.set_page_config(
        page_title="Hospital LOS Predictor",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Route to appropriate page
    if not st.session_state.logged_in:
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()
