import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


penguin_data = pd.read_csv("data/penguins_size.csv")
penguin_data.dropna(subset="body_mass_g", axis=0, inplace=True)
# penguin_data.describe()

X_full = penguin_data.drop(labels="body_mass_g", axis=1)
y = penguin_data['body_mass_g'].copy()

X_train_full, X_valid_full = train_test_split(X_full, test_size=0.2, random_state=0)
y_train, y_valid = train_test_split(y, test_size=0.2, random_state=0)

categorical_cols = [col for col in X_train_full.columns if X_train_full[col].dtypes == 'object' and X_train_full[col].nunique() < 10 ]
# categorical_cols

numerical_cols = [col for col in X_train_full.columns if X_train_full[col].dtypes in ['int64', 'float64'] ]
# numerical_cols

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


numerical_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# categorical_cols.pop(categorical_cols.index('sex'))
# categorical_cols

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

from sklearn.ensemble import RandomForestRegressor

results = dict()
# for rs in range(101):
rs = 53
nest = 64
# for nest in range(1, 301, 1):
mss = 4
maxd = 13
# for maxd in range(1, 101, 1):
model = RandomForestRegressor(random_state=rs, n_estimators=nest, criterion="absolute_error", min_samples_split=mss, max_depth=maxd)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_valid)


# results[maxd] = mean_absolute_error(y_valid, y_pred)

# maxdmin = min(results, key=results.get)
# print("for nest: ", maxdmin, ", MSE: ", results[maxdmin])

print("MAE: ", "{:.2f}".format(mean_absolute_error(y_valid, y_pred)))

# Coeficienti si valori
# print("Regression coefficients: \n", clf.named_steps['model'].regrcoef_)
print("Mean squared error: %.2f" % mean_squared_error(y_valid, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_valid, y_pred))

# # Dreapta de regresie
fig1 = plt.figure("Figure 1")
plt.scatter(X_valid['flipper_length_mm'], y_valid, color="black")
plt.plot(X_valid['flipper_length_mm'], y_pred, color="blue", linewidth=1)
plt.title('Regression line', fontsize=20)
plt.xlabel('Flipper length [mm]')
plt.ylabel('Body mass [g]')
plt.xticks(())
plt.yticks(())




# #matricea de corelare
# corr =  df[['culmen_length_mm', 'flipper_length_mm','body_mass_g']].corr()  
# print('Pearson correlation coefficient matrix for each independent variable: \n', corr)  
  
 
# masking = np.zeros_like(corr, dtype = bool)  
# np.fill_diagonal(masking, val = True)  
  

# figure, axis = plt.subplots(figsize = (8, 4))  
  

  
# # Harta de culoare a matricii de corelare
# sns.heatmap(corr, mask = masking, cmap = 'hsv', vmin = 0, vmax = 1, center = 1, linewidths = 1)  
# figure.suptitle('Heatmap visualizing Pearson Correlation Coefficient Matrix', fontsize = 14)  
# axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  


plt.show()