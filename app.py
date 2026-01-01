import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import hashlib
import json

# Simple user database (in production, use a real database)
# Password: all passwords are hashed
USERS_DB = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin"
    },
    "user1": {
        "password": hashlib.sha256("user123".encode()).hexdigest(),
        "role": "user"
    }
}

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_login(username, password):
    """Verify user credentials"""
    if username in USERS_DB:
        if USERS_DB[username]["password"] == hash_password(password):
            return True, USERS_DB[username]["role"]
    return False, None

def login_page():
    """Display login page"""
    st.title("🔐 ML Demo - Login")
    st.markdown("### Please login to access the ML model")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            is_valid, role = verify_login(username, password)
            if is_valid:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = role
                st.success(f"Welcome {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    st.info("**Demo Credentials:**\n\n- Username: `admin` | Password: `admin123`\n\n- Username: `user1` | Password: `user123`")

def signup_page():
    """Simple signup page (for demo purposes)"""
    st.title("📝 Sign Up")
    st.markdown("### Create a new account")
    
    with st.form("signup_form"):
        new_username = st.text_input("Choose Username")
        new_password = st.text_input("Choose Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            if new_username in USERS_DB:
                st.error("Username already exists!")
            elif new_password != confirm_password:
                st.error("Passwords don't match!")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters!")
            else:
                # In production, save to database
                USERS_DB[new_username] = {
                    "password": hash_password(new_password),
                    "role": "user"
                }
                st.success("Account created! Please login.")
                st.balloons()

@st.cache_data
def load_data():
    """Load Iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return df, iris

@st.cache_resource
def train_model():
    """Train ML model"""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test

def ml_dashboard():
    """Main ML dashboard after login"""
    st.title("🌸 Iris Species Classifier")
    st.markdown(f"**Logged in as:** {st.session_state.username} ({st.session_state.role})")
    
    if st.button("Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.rerun()
    
    st.markdown("---")
    
    # Load data and model
    df, iris = load_data()
    model, accuracy, X_test, y_test = train_model()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Explorer", "🤖 Model Prediction", "📈 Model Performance"])
    
    with tab1:
        st.subheader("Iris Dataset")
        st.dataframe(df.head(10))
        
        st.subheader("Dataset Statistics")
        st.write(df.describe())
        
        st.subheader("Species Distribution")
        species_counts = df['species_name'].value_counts()
        st.bar_chart(species_counts)
    
    with tab2:
        st.subheader("Make a Prediction")
        st.write("Enter flower measurements to predict the species:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        
        with col2:
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
            petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)
        
        if st.button("🔮 Predict Species", type="primary"):
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            species_names = ['Setosa', 'Versicolor', 'Virginica']
            predicted_species = species_names[prediction[0]]
            
            st.success(f"### Predicted Species: **{predicted_species}** 🌸")
            
            st.subheader("Prediction Confidence")
            proba_df = pd.DataFrame({
                'Species': species_names,
                'Probability': prediction_proba[0]
            })
            st.bar_chart(proba_df.set_index('Species'))
    
    with tab3:
        st.subheader("Model Performance")
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        
        st.write("**Model Details:**")
        st.write(f"- Algorithm: Random Forest Classifier")
        st.write(f"- Number of Trees: 100")
        st.write(f"- Training/Test Split: 80/20")
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': iris.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(feature_importance.set_index('Feature'))

def main():
    """Main app"""
    st.set_page_config(
        page_title="ML Demo with Auth",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Show appropriate page
    if not st.session_state.logged_in:
        page = st.sidebar.radio("Navigation", ["Login", "Sign Up"])
        
        if page == "Login":
            login_page()
        else:
            signup_page()
    else:
        ml_dashboard()

if __name__ == "__main__":
    main()
