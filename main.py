import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_excel(r'Zomato(Thrice_upon_time).xlsx')


# Cuisine encoding dictionary
cuisine_encoding = {
    'Afghan': 0,
    'American': 1,
    'Andhra': 2,
    'Arabian': 3,
    'Asian': 4,
    'Awadhi': 5,
    'BBQ': 6,
    'Bakery': 7,
    'Bangladeshi': 8,
    'Bar Food': 9,
    'Bengali': 10,
    'Beverages': 11,
    'Bihari': 12,
    'Biryani': 13,
    'Brazilian': 14,
    'British': 15,
    'Bubble Tea': 16,
    'Burger': 17,
    'Burmese': 18,
    'Cafe': 19,
    'Cantonese': 20,
    'Chettinad': 21,
    'Chinese': 22,
    'Coffee': 23,
    'Continental': 24,
    'Desserts': 25,
    'Drinks Only': 26,
    'European': 27,
    'Fast Food': 28,
    'Finger Food': 29,
    'French': 30,
    'Frozen Yogurt': 31,
    'German': 32,
    'Goan': 33,
    'Grilled Chicken': 34,
    'Gujarati': 35,
    'Healthy Food': 36,
    'Hyderabadi': 37,
    'Ice Cream': 38,
    'Iranian': 39,
    'Italian': 40,
    'Japanese': 41,
    'Juices': 42,
    'Kashmiri': 43,
    'Kathiyawadi': 44,
    'Kebab': 45,
    'Kerala': 46,
    'Konkan': 47,
    'Korean': 48,
    'Lebanese': 49,
    'Lucknowi': 50,
    'Maharashtrian': 51,
    'Malwani': 52,
    'Mandi': 53,
    'Mangalorean': 54,
    'Mediterranean': 55,
    'Mexican': 56,
    'Middle Eastern': 57,
    'Mishti': 58,
    'Mithai': 59,
    'Modern Indian': 60,
    'Momos': 61,
    'Mughlai': 62,
    'Naga': 63,
    'Nepalese': 64,
    'North Eastern': 65,
    'North Indian': 66,
    'Odia': 67,
    'Oriental': 68,
    'Paan': 69,
    'Pancake': 70,
    'Pasta': 71,
    'Pizza': 72,
    'Rajasthani': 73,
    'Roast Chicken': 74,
    'Rolls': 75,
    'Salad': 76,
    'Sandwich': 77,
    'Seafood': 78,
    'Shake': 79,
    'Shawarma': 80,
    'Sichuan': 81,
    'Sindhi': 82,
    'Singaporean': 83,
    'South Indian': 84,
    'Steak': 85,
    'Street Food': 86,
    'Sushi': 87,
    'Tamil': 88,
    'Tea': 89,
    'Thai': 90,
    'Tibetan': 91,
    'Turkish': 92,
    'Vietnamese': 93,
    'Waffle': 94,
    'Wraps': 95
}

# Location encoding dictionary
location_encoding = {
    'Ahmedabad': 0,
    'Bangalore': 1,
    'Chandigarh': 2,
    'Chennai': 3,
    'Delhi': 4,
    'Hyderabad': 5,
    'Jaipur': 6,
    'Kolkata': 7,
    'Lucknow':8,
    'Mumbai': 9,
    'Pune': 10
}

# Create and train the model
X_train, X_test, y_train, y_test = train_test_split(df[['Location', 'Price', 'Cuisine']], df['Rating'], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Title
st.title("Zomato Restaurant Rating Predictor")

# Sidebar for user input
st.sidebar.header("User Input")

# Input for city
city_str = st.sidebar.selectbox("Select City:", location_encoding.keys())

# Input for cuisine
cuisine_str = st.sidebar.selectbox("Select Cuisine:", cuisine_encoding.keys())

# Input for price
price = st.sidebar.number_input("Enter the price:", min_value=0.0)

# Button to get predictions
if st.sidebar.button("Get Predictions"):
    # Map the city string to its encoded value
    if city_str in location_encoding:
        city_code = location_encoding[city_str]
    else:
        st.error('City not found in the encoding.')
        st.stop()

    # Map the cuisine string to its encoded value
    if cuisine_str in cuisine_encoding:
        cuisine_code = cuisine_encoding[cuisine_str]
    else:
        st.error('Cuisine not found in the encoding.')
        st.stop()

    # Filter the data based on user input
    df_filtered = df[(df['Location'] == city_code) & (df['Cuisine'] == cuisine_code) & (df['Price'] <= price)]

    # Check if any restaurants were found
    if df_filtered.empty:
        st.warning('No matching restaurants found in this city for the given cuisine and price.')
    else:
        df_filtered = df_filtered.sort_values(by='Rating', ascending=False)
        best_restaurant = df_filtered.iloc[0]

        # Display the location as a string
        city_str = [k for k, v in location_encoding.items() if v == best_restaurant['Location']][0]

        # Display the cuisine as a string
        cuisine_str = [k for k, v in cuisine_encoding.items() if v == best_restaurant['Cuisine']][0]

        st.success(f'Best restaurant: {best_restaurant ["Hotel_name"]} in {city_str} with {cuisine_str} cuisine')

        new_data = [[best_restaurant['Location'], best_restaurant['Price'], best_restaurant['Cuisine']]]
        predicted_rating = model.predict(new_data)

        st.info(f'Predicted rating for the best restaurant: {predicted_rating[0]:.2f}')

# Display the mean squared error
st.sidebar.header("Model Evaluation")
st.sidebar.info(f'Mean Squared Error: {mse:.2f}')

# Show data sample
#st.sidebar.header("Data Sample")
#st.sidebar.dataframe(df.head())
