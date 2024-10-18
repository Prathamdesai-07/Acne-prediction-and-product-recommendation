import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

model = joblib.load(r"C:/Users/NAINITA/OneDrive/Desktop/DMDW/random_forest_recommendation_model.pkl")

# Load the CSV file
df = pd.read_csv(r'C:/Users/NAINITA/OneDrive/Desktop/DMDW/cosmetics.csv')

# Data Cleaning: Drop rows with missing values
df.dropna(inplace=True)

# Feature Encoding: Convert skin types into binary format
df_encoded = pd.get_dummies(df, columns=['Dry', 'Normal', 'Oily', 'Sensitive'], drop_first=True)

# Define Features and Target Variable
X = df_encoded.drop(['Label', 'Brand', 'Name', 'Price', 'Rank', 'Ingredients'], axis=1)  # Adjust as needed
y = df_encoded[['Dry_1', 'Normal_1', 'Oily_1', 'Sensitive_1']].idxmax(axis=1)  # Get the skin type with the highest value

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to recommend products based on user input
def recommend_products(user_input):
    # Convert user input string to a list of integers
    input_values = list(map(int, user_input.split(',')))

    # Create a DataFrame for prediction using the user's input
    user_df = pd.DataFrame([input_values], columns=X.columns)  # Use X.columns to ensure all features are included
    
    # Predict skin type using the trained model
    predicted_skin_type = model.predict(user_df)[0]

    # Create a mask for filtering products based on predicted skin type
    mask = (df_encoded['Dry_1'] == (predicted_skin_type == 'Dry_1')) & \
           (df_encoded['Normal_1'] == (predicted_skin_type == 'Normal_1')) & \
           (df_encoded['Oily_1'] == (predicted_skin_type == 'Oily_1')) & \
           (df_encoded['Sensitive_1'] == (predicted_skin_type == 'Sensitive_1'))

    recommended = df[mask]

    return recommended[['Brand', 'Name', 'Price', 'Ingredients']]


def validation(user_input):
    # Validate user input length
    expected_length = len(X.columns)
    if len(user_input.split(',')) == expected_length:
        recommendations = recommend_products(user_input)
        if not recommendations.empty:
            return(recommendations)
        else:
            return("No products match your criteria.")
    else:
        return(f"Invalid input. Please enter {expected_length} binary values corresponding to all features.")