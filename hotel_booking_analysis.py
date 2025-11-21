import streamlit as st
# Step 1: Important Libraries Import Banner
import pandas as pd

# Step 2: Read the dataset
data = pd.read_csv("hotel_bookings.csv")

# Step 3: Display basic information
print("✅ Data loaded successfully!")
print("Shape of the dataset:", data.shape)
print("Columns in dataset:", data.columns)
print("Missing values per column:")
print(data.isnull().sum())
# Step 4: Handle missing values
data['children'].fillna(0, inplace=True)
data['country'].fillna('Unknown', inplace=True)
data['agent'].fillna(0, inplace=True)
data['company'].fillna(0, inplace=True)

print("\n✅ Missing values handled successfully!")
print("Now missing values per column:")
print(data.isnull().sum().sum(), "missing values remain in dataset.")
# Step 5: Basic Analysis

# 1️⃣ Count bookings by hotel type
print("\n🏨 Number of bookings by hotel type:")
print(data['hotel'].value_counts())

# 2️⃣ Average daily rate (ADR) comparison
print("\n💰 Average Daily Rate by hotel type:")
print(data.groupby('hotel')['adr'].mean())

# 3️⃣ Most common market segment
print("\n📈 Most common market segment:")
print(data['market_segment'].value_counts().head())

# 4️⃣ Cancellation rate
cancellation_rate = data['is_canceled'].mean() * 100
print(f"\n🚫 Overall cancellation rate: {cancellation_rate:.2f}%")

# 5️⃣ Average stay duration
data['total_stay'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
print("\n🕒 Average stay duration (in nights):", data['total_stay'].mean())
# Step 6: Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 🎨 Set a clean style
sns.set(style="whitegrid")

# 1️⃣ Bookings by Hotel Type
plt.figure(figsize=(6,4))
sns.countplot(x='hotel', data=data, palette='coolwarm')
plt.title('Number of Bookings by Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('Number of Bookings')
plt.show()

# 2️⃣ Average Daily Rate by Hotel Type
plt.figure(figsize=(6,4))
sns.barplot(x='hotel', y='adr', data=data, palette='viridis', estimator='mean')
plt.title('Average Daily Rate by Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('Average Daily Rate')
plt.show()

# 3️⃣ Most Common Market Segments
plt.figure(figsize=(8,4))
top_segments = data['market_segment'].value_counts().head(5)
sns.barplot(x=top_segments.index, y=top_segments.values, palette='magma')
plt.title('Top 5 Market Segments')
plt.xlabel('Market Segment')
plt.ylabel('Number of Bookings')
plt.show()

# 4️⃣ Cancellation Rate Visualization
plt.figure(figsize=(4,4))
cancel_counts = data['is_canceled'].value_counts()
plt.pie(cancel_counts, labels=['Not Canceled', 'Canceled'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
plt.title('Overall Cancellation Rate')
plt.show()

# 5️⃣ Average Stay Duration Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['total_stay'], bins=30, kde=True, color='teal')
plt.title('Distribution of Stay Duration (in Nights)')
plt.xlabel('Total Stay (Nights)')
plt.ylabel('Number of Bookings')
plt.show()
# Step 6: Machine Learning Model to Predict Cancellations
print("\n🤖 Step 6: Machine Learning Model to Predict Cancellations")

# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1️⃣ Select features and target
features = [
    'lead_time', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes', 'total_of_special_requests'
]
X = data[features]
y = data['is_canceled']

# 2️⃣ Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Train the Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 4️⃣ Predict on test data
y_pred = model.predict(X_test)

# 5️⃣ Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# 6️⃣ Confusion Matrix (Numerical)
cm = confusion_matrix(y_test, y_pred)
print("\n🌀 Confusion Matrix (Numerical):\n", cm)

# 7️⃣ Confusion Matrix Display (Visual)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Canceled", "Canceled"])
disp.plot(cmap="Purples", values_format='d')
plt.title("Confusion Matrix for Booking Cancellation Prediction")
plt.show()

# 8️⃣ Classification Report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))
# Step 7: Feature Importance
import pandas as pd
import matplotlib.pyplot as plt

feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n🌟 Feature Importance:\n", feature_importance)

plt.figure(figsize=(8, 5))
feature_importance.head(10).plot(kind='bar', color='teal')
plt.title("Top 10 Important Features for Booking Cancellations")
plt.ylabel("Importance Score")
plt.show()
# Step 7: Insights & Conclusion
print("\n📊 Step 7: Insights & Conclusion")

# 1️⃣ Display Model Performance Summary
print(f"\n🎯 Final Model Accuracy: {accuracy * 100:.2f}%")

# 2️⃣ Interpret the confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"\n✅ True Negatives (Not Canceled predicted correctly): {tn}")
print(f"🚫 False Positives (Predicted Canceled but were Not Canceled): {fp}")
print(f"❌ False Negatives (Predicted Not Canceled but were Canceled): {fn}")
print(f"✔ True Positives (Canceled predicted correctly): {tp}")

# 3️⃣ Calculate and display model precision and recall manually for clarity
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(f"\n📈 Precision: {precision:.2f}")
print(f"📉 Recall: {recall:.2f}")

# 4️⃣ Insights based on model performance
print("\n💡 Insights:")
print("- The model achieves high accuracy, indicating strong performance in predicting booking cancellations.")
print("- Precision and recall values show a balanced capability between detecting cancellations and avoiding false alarms.")
print("- Hotels can use this prediction to identify bookings likely to be canceled and take preventive actions (like flexible pricing or rebooking options).")

# 5️⃣ Conclusion
print("\n📚 Conclusion:")
print("This project successfully analyzed hotel booking data, cleaned missing values, explored key insights, and built a Random Forest model")
print("to predict booking cancellations with good accuracy. The visualizations and metrics demonstrate how data-driven insights can help")
print("improve hotel operations, reduce losses, and enhance customer satisfaction.")
# ==========================================================
# Hotel Booking Analysis Dashboard (Matplotlib Version)
# Developed by Beema R | November 2025
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ------------------- Streamlit Page Config -------------------
st.set_page_config(
    page_title="Hotel Booking Analysis Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------- Title -------------------
st.markdown("<h1>🏨 Hotel Booking Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p>End-to-end analysis & cancellation prediction — Random Forest</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------- Load Dataset -------------------
@st.cache_data(show_spinner=False)
def load_data(path="hotel_bookings.csv"):
    df = pd.read_csv(path)

    # Handle missing values
    df['children'].fillna(0, inplace=True)
    df['country'].fillna('Unknown', inplace=True)
    df['agent'].fillna(0, inplace=True)
    df['company'].fillna(0, inplace=True)

    # Add total stay column
    df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    return df

data = load_data("hotel_bookings.csv")

st.subheader("📋 Dataset Overview")
st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
st.dataframe(data.head(3))
st.markdown("---")

# ------------------- Matplotlib Style -------------------
plt.style.use('dark_background')

# ------------------- Data Visualizations -------------------
st.header("📊 Data Visualizations (Matplotlib)")

# 1️⃣ Number of Bookings by Hotel Type
st.subheader("1. Number of Bookings by Hotel Type")
fig1, ax1 = plt.subplots()
data['hotel'].value_counts().plot(kind='bar', ax=ax1, color=['#03a9f4', '#0288d1'])
ax1.set_title("Number of Bookings by Hotel Type")
ax1.set_xlabel("Hotel Type")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# 2️⃣ Average ADR by Hotel Type
st.subheader("2. Average Daily Rate (ADR) by Hotel Type")
fig2, ax2 = plt.subplots()
data.groupby('hotel')['adr'].mean().plot(kind='bar', ax=ax2, color=['#4fc3f7', '#81d4fa'])
ax2.set_title("Average Daily Rate by Hotel Type")
ax2.set_xlabel("Hotel Type")
ax2.set_ylabel("Average ADR")
st.pyplot(fig2)

# 3️⃣ Most Common Market Segments
st.subheader("3. Most Common Market Segments")
fig3, ax3 = plt.subplots()
data['market_segment'].value_counts().nlargest(5).plot(kind='bar', ax=ax3, color='#03a9f4')
ax3.set_title("Top 5 Market Segments")
ax3.set_xlabel("Market Segment")
ax3.set_ylabel("Count")
st.pyplot(fig3)

# 4️⃣ Distribution of Stay Duration
st.subheader("4. Distribution of Stay Duration (Total Nights)")
fig4, ax4 = plt.subplots()
ax4.hist(data['total_stay'], bins=20, color='#03a9f4', edgecolor='white')
ax4.set_title("Distribution of Stay Duration (Total Nights)")
ax4.set_xlabel("Total Nights")
ax4.set_ylabel("Frequency")
st.pyplot(fig4)

# 5️⃣ Cancellations Distribution
st.subheader("5. Cancellations Distribution")
fig5, ax5 = plt.subplots()
data['is_canceled'].value_counts().plot(kind='bar', ax=ax5, color=['#4fc3f7', '#0288d1'])
ax5.set_title("Canceled vs Not Canceled")
ax5.set_xticklabels(["Not Canceled", "Canceled"], rotation=0)
ax5.set_ylabel("Count")
st.pyplot(fig5)

# 6️⃣ Average ADR by Arrival Month
st.subheader("6. Average ADR by Arrival Month")
fig6, ax6 = plt.subplots()
data.groupby('arrival_date_month')['adr'].mean().plot(kind='bar', ax=ax6, color='#03a9f4')
ax6.set_title("Average ADR by Arrival Month")
ax6.set_xlabel("Arrival Month")
ax6.set_ylabel("Average ADR")
st.pyplot(fig6)

# 7️⃣ Distribution of Special Requests
st.subheader("7. Distribution of Total Special Requests")
fig7, ax7 = plt.subplots()
ax7.hist(
    data['total_of_special_requests'],
    bins=range(0, data['total_of_special_requests'].max() + 2),
    color='#81d4fa',
    edgecolor='white'
)
ax7.set_title("Distribution of Total Special Requests")
ax7.set_xlabel("Number of Special Requests")
ax7.set_ylabel("Frequency")
st.pyplot(fig7)

st.markdown("---")

# ------------------- Machine Learning Model -------------------
st.header("🤖 Machine Learning Model (Random Forest)")

features = [
    'lead_time', 'adr', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'adults', 'children', 'babies',
    'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'total_of_special_requests'
]

X = data[features].fillna(0)
y = data['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)

with st.spinner("Training Random Forest model..."):
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred, digits=3)

# ------------------- Model Results -------------------
st.subheader("📈 Model Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy*100:.2f}%")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# ------------------- Feature Importance -------------------
st.subheader("🌟 Feature Importance")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig_imp, ax_imp = plt.subplots()
ax_imp.barh(importance['Feature'], importance['Importance'], color='#03a9f4')
ax_imp.invert_yaxis()
ax_imp.set_title("Feature Importance (Random Forest)")
ax_imp.set_xlabel("Importance")
st.pyplot(fig_imp)

# ------------------- Confusion Matrix -------------------
st.subheader("🧮 Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
im = ax_cm.imshow(cm, cmap='Blues')
ax_cm.set_title("Confusion Matrix (Actual vs Predicted)")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[i])):
        ax_cm.text(j, i, cm[i][j], ha='center', va='center', color='white')

st.pyplot(fig_cm)

# ------------------- Classification Report -------------------
st.subheader("📋 Classification Report")
st.text(clf_report)

# ------------------- Insights -------------------
st.header("💡 Insights & Conclusion")
st.write("""
The Random Forest model achieved strong predictive performance.

*Lead time, **ADR, and **Special Requests* were highly important.

Hotels can use this model to reduce cancellations and improve efficiency.
""")

# ------------------- Prediction Section -------------------
st.markdown("---")
st.header("🔮 Booking Cancellation Prediction")
st.write("Use the form below to simulate a booking and predict cancellation.")

with st.form("prediction_form"):
    st.subheader("📋 Enter Booking Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        lead_time = st.number_input("Lead Time (days)", 0, 500, 50)
        adr = st.number_input("ADR", 0.0, 600.0, 100.0)
        stays_in_weekend_nights = st.number_input("Weekend Nights", 0, 20, 2)
        stays_in_week_nights = st.number_input("Week Nights", 0, 50, 3)

    with col2:
        adults = st.number_input("Adults", 0, 10, 2)
        children = st.number_input("Children", 0, 10, 0)
        babies = st.number_input("Babies", 0, 5, 0)
        booking_changes = st.number_input("Booking Changes", 0, 10, 0)

    with col3:
        previous_cancellations = st.number_input("Previous Cancellations", 0, 20, 0)
        previous_bookings_not_canceled = st.number_input("Previous Non-Cancelled", 0, 20, 1)
        total_of_special_requests = st.number_input("Special Requests", 0, 10, 1)

    submitted = st.form_submit_button("🚀 Predict Booking Status")

if submitted:
    input_data = pd.DataFrame({
        'lead_time': [lead_time],
        'adr': [adr],
        'stays_in_weekend_nights': [stays_in_weekend_nights],
        'stays_in_week_nights': [stays_in_week_nights],
        'adults': [adults],
        'children': [children],
        'babies': [babies],
        'previous_cancellations': [previous_cancellations],
        'previous_bookings_not_canceled': [previous_bookings_not_canceled],
        'booking_changes': [booking_changes],
        'total_of_special_requests': [total_of_special_requests]
    })

    prediction = model.predict(input_data)[0]
    prediction_prob = model.predict_proba(input_data)[0][prediction]

    st.markdown("---")

    if prediction == 1:
        st.error(f"❌ Booking Canceled | Confidence: {prediction_prob*100:.2f}%")
    else:
        st.success(f"✅ Booking Confirmed | Confidence: {prediction_prob*100:.2f}%")

st.success("✅ Project successfully completed with excellent results!")

# ------------------- Footer -------------------
st.markdown("<p>Developed by <b>Beema R</b> | November 2025</p>", unsafe_allow_html=True)