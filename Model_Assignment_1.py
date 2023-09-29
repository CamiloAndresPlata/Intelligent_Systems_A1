# %% Imports

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from pyfume.Clustering import Clusterer
from pyfume.EstimateAntecendentSet import AntecedentEstimator
from pyfume.EstimateConsequentParameters import ConsequentEstimator
from pyfume.SimpfulModelBuilder import SugenoFISBuilder
from pyfume.Tester import SugenoFISTester
from numpy import copy
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef
from numpy import clip, column_stack, argmax
from pandas import DataFrame

# %% Loading dataset and preprocessing

data = load_wine()
X = data.data
X = minmax_scale(X)
y = data.target
var_names = data.feature_names
var_names = [var_names[i].title().replace('/','_') for i in range(0, len(var_names))]

y_0_vs_all = copy(y)
y_0_vs_all[y_0_vs_all==0] = -1
y_0_vs_all[y_0_vs_all!=-1] = 0
y_0_vs_all[y_0_vs_all==-1] = 1

y_1_vs_all = copy(y)
y_1_vs_all[y_1_vs_all==1] = -1
y_1_vs_all[y_1_vs_all!=-1] = 0
y_1_vs_all[y_1_vs_all==-1] = 1

y_2_vs_all = copy(y)
y_2_vs_all[y_2_vs_all==2] = -1
y_2_vs_all[y_2_vs_all!=-1] = 0
y_2_vs_all[y_2_vs_all==-1] = 1

#%% train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
_, _, y_train_0_vs_all, _ = train_test_split(X, y_0_vs_all, test_size=0.3, random_state=42)
_, _, y_train_1_vs_all, _ = train_test_split(X, y_1_vs_all, test_size=0.3, random_state=42)
_, _, y_train_2_vs_all, _ = train_test_split(X, y_2_vs_all, test_size=0.3, random_state=42)


# %% 0 vs all model training

# Clustering the input-output space (fuzzy c-means)
cl = Clusterer(x_train=X_train, y_train=y_train_0_vs_all, nr_clus=5)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimation of membership functions parameters (Gaussian shape fit)
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimation of consequent parameters (least mean squares estimation)
ce = ConsequentEstimator(X_train, y_train_0_vs_all, part_matrix)
conseq_params = ce.suglms()
# First order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_0_vs_all = modbuilder.get_model()

# %% 1 vs all model training

# Clustering the input-output space (fuzzy c-means)
cl = Clusterer(x_train=X_train, y_train=y_train_1_vs_all, nr_clus=5)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimation of membership functions parameters (Gaussian shape fit)
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimation of consequent parameters (least mean squares estimation)
ce = ConsequentEstimator(X_train, y_train_1_vs_all, part_matrix)
conseq_params = ce.suglms()
# First order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_1_vs_all = modbuilder.get_model()


# %% 2 vs all model training

# Clustering the input-output space (fuzzy c-means)
cl = Clusterer(x_train=X_train, y_train=y_train_2_vs_all, nr_clus=5)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimation of membership functions parameters (Gaussian shape fit)
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimation of consequent parameters (least mean squares estimation)
ce = ConsequentEstimator(X_train, y_train_2_vs_all, part_matrix)
conseq_params = ce.suglms()
# First order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_2_vs_all = modbuilder.get_model()

# %% Class probabilities predictions for each ova model

# Class probabilities predictions for 0 vs all
modtester = SugenoFISTester(model_0_vs_all, X_test, var_names)
y_pred_probs_0_vs_all = clip(modtester.predict()[0], 0, 1)

# Class probabilities predictions for 1 vs all
modtester = SugenoFISTester(model_1_vs_all, X_test, var_names)
y_pred_probs_1_vs_all = clip(modtester.predict()[0], 0, 1)

# Class probabilities predictions for 2 vs all
modtester = SugenoFISTester(model_2_vs_all, X_test, var_names)
y_pred_probs_2_vs_all = clip(modtester.predict()[0], 0, 1)

# %% Class predictions

y_pred_probs = column_stack((y_pred_probs_0_vs_all, y_pred_probs_1_vs_all, y_pred_probs_2_vs_all))
y_pred_probs = y_pred_probs / y_pred_probs.sum(axis=1, keepdims=1)
y_pred = argmax(y_pred_probs, axis=1)

# %% Classification performance metrics

acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa Score: {:.3f}".format(kappa))
matthews_coef = matthews_corrcoef(y_test, y_pred)
print("matthews correlation coefficient: {:.3f}".format(matthews_coef))

# %% saving y_pred to a csv file

# Convert the array to a DataFrame
pred = DataFrame(y_pred)
# Save the DataFrame to a CSV file
pred.to_csv('y_prediction.csv', index=False, header=False)