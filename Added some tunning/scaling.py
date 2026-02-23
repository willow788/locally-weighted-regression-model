#here we are going to scale the dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

#inintialising the scaler
scaler = StandardScaler()

#fitting the scaler
df = pd.read_csv('housing.csv')

df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())

#enncoding ocean_proximity catagorcial variable to numerical
from sklearn.preprocessing import OneHotEncoder

#inintialising the encoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

#ftting and transforming the data
encoded_data = encoder.fit_transform(df[['ocean_proximity']]) 

#creating a new dataframe
#and concatenating it with the original dataset
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['ocean_proximity']))
df = pd.concat([df, encoded_df], axis=1)

#dropping the original ocean_proximity column
df.drop('ocean_proximity', axis=1, inplace=True)



#spliting
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fitting the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)
#transforming the test data
X_test_scaled = scaler.transform(X_test)

#converting back to dataframes
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

