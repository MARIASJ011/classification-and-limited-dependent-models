# Codes
R 
# Core libraries 
install.packages("tidyverse") 
install.packages("caret") 
install.packages("pROC") 
install.packages("ResourceSelection")  # Hosmer-Lemeshow test 
install.packages("car")    
install.packages("rpart") 
install.packages("rpart.plot") 
library(tidyverse) 
library(caret) 
library(pROC) 
library(ResourceSelection) 
library(car) 
            # For VIF 
data <- read.csv("C:\\Users\\Maria\\Downloads\\heart - heart.csv") 
data$target <- as.factor(data$target) 
logit_model <- glm(target ~ ., data = data, family = binomial) 
summary(logit_model) 
16 
vif(logit_model) 
hoslem.test(data$target, fitted(logit_model)) 
predicted_class <- ifelse(predict(logit_model, type = "response") > 0.5, 1, 0) 
confusionMatrix(as.factor(predicted_class), data$target) 
roc_curve <- roc(as.numeric(data$target), predict(logit_model, type = "response")) 
plot(roc_curve, col = "blue", main = "ROC Curve - Logistic Regression") 
auc(roc_curve) 
library(caret) 
nsso <- read.csv("C:\\Users\\Maria\\OneDrive\\Desktop\\NSSO68.csv") 
nsso$non_veg <- ifelse(rowSums(nsso[, c('eggsno_q', 'fishprawn_q', 'goatmeat_q', 
'beef_q', 'pork_q','chicken_q', 'othrbirds_q')]) > 0, 1, 0) 
nsso$non_veg <- as.factor(nsso$non_veg) 
# Convert necessary predictors to factor 
nsso$Region <- as.factor(nsso$Region) 
probit_model <- glm(non_veg ~ HH_type + Religion + Social_Group + Regular_salary_earner 
+ 
Region + Meals_At_Home + Education + Age + Sex + Possess_ration_card, 
17 
data = nsso, family = binomial(link = "probit")) 
summary(probit_model) 
library(rpart) 
library(rpart.plot) 
library(caret) 
library(pROC) 
# Load dataset 
data <- read.csv("C:\\Users\\Maria\\Downloads\\heart - heart.csv") 
data$target <- as.factor(data$target) 
# Fit Decision Tree 
tree_model <- rpart(target ~ ., data = data, method = "class") 
rpart.plot(tree_model) 
# Predict classes 
tree_preds <- predict(tree_model, type = "class") 
confusionMatrix(tree_preds, data$target) 
# Predict probabilities for ROC 
tree_probs <- predict(tree_model, type = "prob")[,2] 
roc_tree <- roc(as.numeric(data$target), tree_probs) 
18 
plot(roc_tree, col = "green", main = "ROC Curve - Decision Tree") 
auc(roc_tree) 
# 1. Load the dataset 
nsso <- read.csv("C:\\Users\\Maria\\OneDrive\\Desktop\\NSSO68.csv") 
# 2. Create a censored dependent variable (total non-veg quantity) 
nsso$nonveg_total_qty <- rowSums(nsso[, c('eggsno_q', 'fishprawn_q', 'goatmeat_q', 
'beef_q', 'pork_q', 'chicken_q', 'othrbirds_q')], 
na.rm = TRUE) 
# 3. Install and load AER package for Tobit regression 
if (!require("AER")) install.packages("AER") 
library(AER) 
# 4. Clean the data (remove rows with NA in outcome) 
nsso_clean <- nsso[!is.na(nsso$nonveg_total_qty), ] 
# 5. Fit the Tobit regression model (censoring from below at 0) 
tobit_model <- tobit(nonveg_total_qty ~ HH_type + Religion + Social_Group +  
Age + Sex + Region + Possess_ration_card, 
left = 0, data = nsso_clean) 
# 6. Display the model summary 
summary(tobit_model) 
19 
PYTHON: 
import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
# Load and prepare data 
heart = pd.read_csv("heart - heart.csv") 
X = heart.drop(columns="target") 
y = heart["target"] 
# Scale data 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
# Fit logistic model 
model = LogisticRegression(max_iter=1000) 
model.fit(X_scaled, y) 
# Predict and evaluate 
y_pred = model.predict(X_scaled) 
y_prob = model.predict_proba(X_scaled)[:, 1] 
print("Confusion Matrix:\n", confusion_matrix(y, y_pred)) 
20 
# ROC and AUC 
fpr, tpr, _ = roc_curve(y, y_prob) 
plt.plot(fpr, tpr, label=f"AUC: {roc_auc_score(y, y_prob):.3f}", color="blue") 
plt.xlabel("FPR") 
plt.ylabel("TPR") 
plt.title("ROC Curve - Logistic Regression") 
plt.legend() 
plt.show() 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
tree_model = DecisionTreeClassifier(max_depth=4) 
tree_model.fit(X_scaled, y) 
# Prediction & Evaluation 
y_tree_pred = tree_model.predict(X_scaled) 
y_tree_prob = tree_model.predict_proba(X_scaled)[:, 1] 
print("Confusion Matrix (Tree):\n", confusion_matrix(y, y_tree_pred)) 
# ROC 
fpr_tree, tpr_tree, _ = roc_curve(y, y_tree_prob) 
plt.plot(fpr_tree, tpr_tree, label=f"AUC: {roc_auc_score(y, y_tree_prob):.3f}", 
color="green") 
plt.title("ROC Curve - Decision Tree") 
plt.xlabel("FPR") 
plt.ylabel("TPR") 
21 
plt.legend() 
plt.show() 
import statsmodels.api as sm 
from statsmodels.discrete.discrete_model import Probit 
# Load data 
nsso = pd.read_csv("NSSO68.csv") 
# Binary non-veg column 
nonveg_items = ['eggsno_q', 'fishprawn_q', 'goatmeat_q', 'beef_q', 'pork_q', 'chicken_q', 
'othrbirds_q'] 
nsso['non_veg'] = (nsso[nonveg_items].sum(axis=1) > 0).astype(int) 
# Predictors 
predictors = ['HH_type', 'Religion', 'Social_Group', 'Regular_salary_earner', 
'Region', 'Meals_At_Home', 'Education', 'Age', 'Sex', 'Possess_ration_card'] 
# Clean and prepare 
nsso_clean = nsso.dropna(subset=predictors + ['non_veg']) 
X = pd.get_dummies(nsso_clean[predictors], drop_first=True) 
X = sm.add_constant(X) 
y = nsso_clean['non_veg'] 
# Fit Probit model 
probit_model = Probit(y, X).fit() 
print(probit_model.summary()) 
from statsmodels.miscmodels.ordinal_model import OrderedModel  # Note: PyTobit needs 
custom setup 
import statsmodels.formula.api as smf 
# Total non-veg consumption 
22 
nsso['nonveg_total_qty'] = nsso[nonveg_items].sum(axis=1) 
nsso_tobit = nsso.dropna(subset=['nonveg_total_qty'] + predictors) 
# Create dummies 
X_tobit = pd.get_dummies(nsso_tobit[predictors], drop_first=True) 
X_tobit = sm.add_constant(X_tobit) 
y_tobit = nsso_tobit['nonveg_total_qty'] 
# Tobit approximation using censored regression 
from statsmodels.base.model import GenericLikelihoodModel 
class Tobit(GenericLikelihoodModel): 
def loglike(self, params): 
endog = self.endog 
exog = self.exog 
beta = params[:-1] 
sigma = params[-1] 
XB = np.dot(exog, beta) 
cens = endog == 0 
llf = np.where(cens, 
np.log(sm.distributions.norm.cdf((0 - XB) / sigma)), 
np.log(sm.distributions.norm.pdf((endog - XB) / sigma)) - np.log(sigma)) 
return llf.sum() 
X_np = X_tobit.values 
y_np = y_tobit.values 
params_init = np.append(np.zeros(X_np.shape[1]), 1) 
tobit_mod = Tobit(y_np, X_np) 
tobit_res = tobit_mod.fit(start_params=params_init, maxiter=200, disp=0) 
print(tobit_res.summary()) 
