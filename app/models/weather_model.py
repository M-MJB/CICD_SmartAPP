import pickle
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures, DropDuplicateFeatures
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Load the breast cancer dataset
df=pd.read_csv(r'app/models/Weather_Data.csv')

categories = ['Clear','Cloudy','Snow','Rain','Drizzle','Fog','Thunderstorms','Haze']

df.Weather.replace(to_replace='Mainly Clear',value='Clear',inplace=True)
df.Weather.replace(to_replace='Mostly Cloudy',value='Cloudy',inplace=True)
df.Weather.replace(to_replace=['Snow Showers','Snow,Blowing Snow','Rain,Snow','Freezing Drizzle,Snow','Freezing Rain,Snow Grains','Snow,Ice Pellets','Moderate Snow','Rain,Snow,Ice Pellets','Drizzle,Snow','Rain Showers,Snow Showers','Moderate Snow,Blowing Snow','Snow Pellets','Rain,Snow Grains'],value='Snow',inplace=True)
df.Weather.replace(to_replace=['Rain Showers','Freezing Rain','Rain,Ice Pellets'],value='Rain',inplace=True)
df.Weather.replace(to_replace='Freezing Drizzle',value='Drizzle',inplace=True)
df.Weather.replace(to_replace=['Rain,Fog','Drizzle,Fog','Snow,Fog','Drizzle,Snow,Fog','Freezing Drizzle,Fog','Freezing Fog','Snow Showers,Fog','Freezing Rain,Fog','Thunderstorms,Rain Showers,Fog','Rain Showers,Fog','Thunderstorms,Moderate Rain Showers,Fog','Rain,Snow,Fog','Moderate Rain,Fog','Freezing Rain,Ice Pellets,Fog','Drizzle,Ice Pellets,Fog','Thunderstorms,Rain,Fog'],value='Fog',inplace=True)
df.Weather.replace(to_replace=['Thunderstorms,Rain Showers','Thunderstorms,Rain','Thunderstorms,Heavy Rain Showers'],value='Thunderstorms',inplace=True)
df.Weather.replace(to_replace=['Snow,Haze','Rain,Haze','Freezing Drizzle,Haze','Freezing Rain,Haze',],value='Haze',inplace=True)

df['Date/Time'] = pd.to_datetime(df['Date/Time'],errors='coerce')
df['Year'] = df['Date/Time'].dt.year
df['Month'] = df['Date/Time'].dt.month
df['Day'] = df['Date/Time'].dt.day
df['Hour'] = df['Date/Time'].dt.hour
df['Minute'] = df['Date/Time'].dt.minute
df['Second'] = df['Date/Time'].dt.second
df['Days in Month'] = df['Date/Time'].dt.daysinmonth
df['Week'] = df['Date/Time'].dt.weekday
df['is_leap_year'] = df['Date/Time'].dt.is_leap_year
df['is_month_start'] = df['Date/Time'].dt.is_month_start
df['is_month_end'] = df['Date/Time'].dt.is_month_end
df['is_year_start'] = df['Date/Time'].dt.is_year_start
df['is_year_end'] = df['Date/Time'].dt.is_year_end
df['is_quarter_start'] = df['Date/Time'].dt.is_quarter_start
df['is_quarter_end'] = df['Date/Time'].dt.is_quarter_end
df['Weekday'] = df['Date/Time'].dt.weekday


df.drop('Date/Time',axis=1,inplace=True)

df['is_leap_year'] = df['is_leap_year'].astype(int)
df['is_month_end'] = df['is_month_end'].astype(int)
df['is_month_start'] = df['is_month_start'].astype(int)
df['is_quarter_end'] = df['is_quarter_end'].astype(int)
df['is_quarter_start'] = df['is_quarter_start'].astype(int)
df['is_year_end'] = df['is_year_end'].astype(int)
df['is_year_start'] = df['is_year_start'].astype(int)


skewed_cols = ['Press_kPa','Visibility_km','Wind Speed_km/h']

def remove_outliers(data,col):
    lower_limit, upper_limit = data[col].quantile([0.25,0.75])
    IQR = upper_limit - lower_limit
    lower_whisker = lower_limit - 1.5 * IQR
    upper_whisker = upper_limit + 1.5 * IQR
    return np.where(data[col]<lower_whisker,lower_whisker,np.where(data[col]>upper_whisker,upper_whisker,data[col]))

for col in skewed_cols:
    df[col] = remove_outliers(df,col)


X = df.drop('Weather',axis=1)
y = df.Weather

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,shuffle=True)

print("Shape of the training set:",X_train.shape)
print("Shape of the testing set:",X_test.shape)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

label_mapping = {label:idx for idx, label in enumerate(encoder.classes_)}
print(label_mapping)


print("Shape of the training set:",X_train.shape)
print("Shape of the testing set:",X_test.shape)


final_selected_features = ['Temp_C','Press_kPa','Rel Hum_%','Wind Speed_km/h','Visibility_km','Hour']

final_X_train = X_train[final_selected_features]
final_X_test = X_test[final_selected_features]



scaler = StandardScaler()
features = final_X_train.columns
final_X_train = scaler.fit_transform(final_X_train)
final_X_train = pd.DataFrame(final_X_train,columns=features)
final_X_test = scaler.transform(final_X_test)
final_X_test = pd.DataFrame(final_X_test,columns=features)
final_X_train.head()



# Create and train the model

model =SGDClassifier()
model.fit(final_X_train,y_train)
y_pred = model.predict(final_X_test)
acc = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred,average='macro')
recall = recall_score(y_test,y_pred,average='macro')
f1 = f1_score(y_test,y_pred,average='macro')
print(acc)



model_path = os.path.join('app\models', 'model.pkl')
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)