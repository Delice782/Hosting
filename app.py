import streamlit as st
import sqlite3
import hashlib

# --- DATABASE SETUP ---
# This creates a file named 'users.db' in your folder automatically
def create_usertable():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')
    conn.commit()
    conn.close()

def add_userdata(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data

# Security: Hash passwords so they aren't stored as plain text
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# --- MAIN APP INTERFACE ---
def main():
    st.set_page_config(page_title="Capstone Project", layout="centered")
    st.title("🔒 Capstone Team Portal")
    
    create_usertable()

    # Initialize session state so the app "remembers" you are logged in
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        # sidebar menu for Login or Signup
        menu = ["Login", "Sign Up"]
        choice = st.sidebar.selectbox("Access Menu", menu)

        if choice == "Login":
            st.subheader("Login to your Account")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            
            if st.button("Login"):
                hashed_pswd = make_hashes(password)
                result = login_user(username, hashed_pswd)
                if result:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

        elif choice == "Sign Up":
            st.subheader("Create a New Account")
            new_user = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type='password')

            if st.button("Register"):
                if new_user and new_password:
                    add_userdata(new_user, make_hashes(new_password))
                    st.success("Account created successfully!")
                    st.info("You can now go to the Login tab.")
                else:
                    st.warning("Please fill out both fields.")
    else:
        # --- LOGGED IN VIEW ---
        st.sidebar.success(f"Logged in as: {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()
            
        st.write("---")
        st.header("Capstone Workspace")
        st.write("This content is only visible to registered users.")
        # You can add your data collection forms or project charts here!

if __name__ == '__main__':
    main()
