import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# Concatenate all samples collected and display the first five.

df0 = pd.read_csv("clothing_items.csv", sep=",")
df1 = pd.read_csv("clothing_items_new_500.csv", sep=",")
df2 = pd.read_csv("clothing_items_new2_1000.csv", sep=",")

df = pd.concat([df0, df1, df2], ignore_index=True)
df.head()

# Print a concise summary of the dataframe
df.info()

# General descriptive statistics.
df.describe()

df.columns

# Functions for visualization

# Bar Chart
def plot_bar_chart(data, column):
    value_counts = data[column].value_counts()
    colors = ['#FFB6C1', '#87CEEB', '#FFD700', '#98FB98', '#CD5C5C']
    plt.figure(figsize=(10, 6))
    plt.bar(value_counts.index, value_counts.values, color=colors)
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.title(f'Distribution by columna "{column}"')
    plt.xticks(rotation=45)
    plt.show()

# Pie chart
def plot_pie_chart(data, column):
    value_counts = data[column].value_counts()
    plt.figure(figsize=(10, 6))
    plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title(f'Distribution by {column}')
    plt.show()

plot_bar_chart(df, 'type')

plot_bar_chart(df, 'brand')

plot_bar_chart(df, 'material')

plot_bar_chart(df, 'style')

plot_bar_chart(df, 'color')

plot_bar_chart(df, 'state')

df.nunique()

df.isnull().sum()

print("Analysis of TYPE feature")
df['type'].value_counts()

print("Analysis of BRAND feature")
df['brand'].value_counts()

print("Analysis of MATERIAL feature")
df['material'].value_counts()

print("Analysis of STYLE feature")
df['style'].value_counts()

print("Analysis of COLOR feature")
df['color'].value_counts()

print("Analysis of STATE feature")
df['state'].value_counts()

"""## Data Preprocessing 🧰

Since we have several categorical variables in this dataset, we need to convert them into numeric form so that we can use them in our regression model
"""

df_encoded = pd.get_dummies(df, columns=['type', 'brand', 'material', 'style', 'color', 'state'])
df_encoded.info()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded.drop('price', axis=1),
    df_encoded['price'],
    shuffle=True)

# Scaling the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled

"""## Regression Model 🔢

### Multiple Linear Regression
"""

# Create a linear regression model
lr = LinearRegression()
# Fit the model to the training data
lr.fit(X_train_scaled, y_train)

# Evaluate the model on the testing data
y_pred_linear = lr.predict(X_test_scaled)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print('Linear regression MAE:', mae_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print('Linear Regression R2 Score:', r2_linear)

"""### Decision Tree"""

# Create a decision tree model
dt = DecisionTreeRegressor()
# Fit the model to the training data
dt.fit(X_train_scaled, y_train)

# Evaluate the model on the testing data
y_pred_tree = dt.predict(X_test_scaled)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
print('Decision tree MAE:', mae_tree)
r2_tree = r2_score(y_test, y_pred_tree)
print('Decision Tree R2 Score:', r2_tree)

"""### Random Forest"""

# Create a random forest model
rf = RandomForestRegressor()
# Fit the model to the training data
rf.fit(X_train_scaled, y_train)

# Evaluate the model on the testing data
y_pred_rf = rf.predict(X_test_scaled)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print('Random forest MSE:', mae_rf)
r2_forest = r2_score(y_test, y_pred_rf)
print('Random Forest R2 Score:', r2_forest)

"""## Results 📑"""

model_names = ['Decission tree', 'Linear regression', 'Random forest']

r2_scores = [r2_tree, r2_linear, r2_forest]
mae_scores = [mae_tree, mae_linear, mae_rf]

plt.figure(figsize=(10, 6))
plt.bar(model_names, r2_scores)
plt.xlabel('Modelos')
plt.ylabel('R2')
plt.title('Comparision of R2 between  modelos')
plt.ylim(0, 1)  #Set the y-axis range between 0 and 1 for R²
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(model_names, mae_scores)
plt.xlabel('Modelos')
plt.ylabel('MAE')
plt.title('Comparision of MAE b/w modelos')
plt.show()

def predict_price(brand, style, material, color, state):
    # Create a dataframe for the new item
    new_item = pd.DataFrame({
        'brand': [brand],
        'style': [style],
        'material': [material],
        'color': [color],
        'state': [state]
    })

    # Encode the new item in the same way as the training data
    new_item_encoded = pd.get_dummies(new_item, drop_first=True)

    # Ensure the new item has the same columns as the training data
    for col in X_train.columns:
        if col not in new_item_encoded.columns:
            new_item_encoded[col] = 0

    new_item_encoded = new_item_encoded[X_train.columns]  # Ensure column order matches

    # Scale the new item
    new_item_scaled = scaler.transform(new_item_encoded)

    # Predict the price
    predicted_price = rf.predict(new_item_scaled)
    return predicted_price[0]

# Function to validate user input
def validate_input(prompt, valid_options=None):
    while True:
        user_input = input(prompt)
        if valid_options and user_input not in valid_options:
            print(f"Invalid input. Please choose from the following: {', '.join(valid_options)}")
        else:
            return user_input

# List of valid options for user input validation
valid_brands = df['brand'].unique().tolist()
valid_styles = df['style'].unique().tolist()
valid_materials = df['material'].unique().tolist()
valid_colors = df['color'].unique().tolist()
valid_states = ['new', 'used']

# Function to take user input and predict price
def get_user_input():
    while True:
        # Prompt the user for input
        brand = validate_input("Enter the brand (or type 'exit' to quit): ", valid_brands)
        if brand.lower() == 'exit':
            break
        style = validate_input("Enter the style: ", valid_styles)
        material = validate_input("Enter the material: ", valid_materials)
        color = validate_input("Enter the color: ", valid_colors)
        state = validate_input("Enter the state (new/used): ", valid_states)

        # Predict the price
        price = predict_price(brand, style, material, color, state)
        print(f"The predicted price for the given clothing item is: ${price:.2f}\n")

        # Ask the user if they want to predict another item
        continue_prediction = input("Do you want to predict the price for another clothing item? (yes/no): ")
        if continue_prediction.lower() != 'yes':
            break

# Uncomment the following line to test user input functionality
get_user_input()

