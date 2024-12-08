# Data Preprocessing

import pandas as pd
import boto3
from io import StringIO
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest


# Initialize a session using Amazon S3
s3 = boto3.client('s3')
 
# Define your bucket and object path (file key)
bucket_name = 'x23180013cac-data'
file_key = 'cloud_access_control_dataset.csv'
 
# Get the CSV file from S3
response = s3.get_object(Bucket=bucket_name, Key=file_key)
 
# Read the CSV data directly into a pandas DataFrame
access_control = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

print("Class Distribution of Security_Score:")
print(access_control['Security_Score'].value_counts(normalize=True))
access_control.shape
access_control.info()

"""### Data Selection"""

access_control.isnull().sum()
access_control.duplicated().sum()
access_control.describe()

"""###Categorical Encoding"""

categorical_columns = access_control.select_dtypes(include=['object']).columns

for col in access_control.select_dtypes(include=['object']).columns:
    access_control[col] = LabelEncoder().fit_transform(access_control[col])

for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=access_control, x=col)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

# One-Hot Encoding for categorical variables
access_control_encoded = pd.get_dummies(
    access_control, columns=categorical_columns, drop_first=True)

# unique values of encoded columns
for col in categorical_columns:
    print(f"\nUnique values in {col} before encoding:\n",
          access_control[col].unique())

for col in access_control_encoded.columns:
    if col.startswith(tuple(categorical_columns)):
        plt.figure(figsize=(10, 5))
        sns.countplot(data=access_control_encoded, x=col)
        plt.title(f"Distribution of {col} after One-Hot Encoding")
        plt.xticks(rotation=45)
        plt.show()

"""### Feature Correlation"""

corr_matrix = access_control_encoded.corr().abs()

# Visualize the correlation matrix
plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.show()

# Create an upper triangle matrix to remove redundant correlations
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Set the correlation threshold (default is 0.9)
threshold = 0.9

# Find columns to drop based on the threshold
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

# Drop the highly correlated features
access_control_cleaned = access_control_encoded.drop(columns=to_drop)

print("Columns dropped due to high correlation:")
print(to_drop)

print("\nColumns retained after dropping highly correlated features:")
print(access_control_cleaned.columns)

correlation_with_target = corr_matrix['Security_Score'].sort_values(ascending=False)

print("Top 10 features positively correlated with Security_Score:")
print(correlation_with_target.head(10))

print("\nTop 10 features negatively correlated with Security_Score:")
print(correlation_with_target.tail(10))

"""# Training: Split Data into Train and Test Sets"""

# Training: Split the data into training and testing sets
X = access_control_cleaned.drop(columns=['Security_Score'])  # Feature matrix
y = access_control_cleaned['Security_Score']  # Target variable

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# Standardization: Scaling Features"""

# Standardize the features (Scaling)
from sklearn.preprocessing import StandardScaler

# Initialize scaler and scale the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""#  Regularization (Lasso): Remove Unimportant Features"""

# Regularization using LassoCV to select important features
from sklearn.linear_model import LassoCV
import pandas as pd

# LassoCV with 5-fold cross-validation
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

# Get non-zero coefficient features
lasso_coefficients = pd.Series(lasso_cv.coef_, index=X.columns)
important_features = lasso_coefficients[lasso_coefficients != 0].index

# Create a new dataset with only the important features
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

print(f"Selected important features: {list(important_features)}")

"""# Generalization (XGBoost Feature Selection and Tuning)

### Hyperparameter Tuning with RandomizedSearchCV
"""

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# XGBoost hyperparameter tuning using RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

le = LabelEncoder()
y_train = le.fit_transform(y_train) # Fit label encoder on y_train and transform y_train
y_test = le.transform(y_test)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, n_iter=10, cv=2, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train_selected, y_train)

# Use the best estimator
best_xgb = random_search.best_estimator_

# Extract feature importances
xgb_feature_importances = pd.Series(best_xgb.feature_importances_, index=important_features)

# Drop low-importance features
importance_threshold = 0.01
low_importance_features = xgb_feature_importances[xgb_feature_importances < importance_threshold].index

# Drop low-importance features from the dataset
X_train_final = X_train_selected.drop(columns=low_importance_features)
X_test_final = X_test_selected.drop(columns=low_importance_features)

print(f"Removed low-importance features: {list(low_importance_features)}")

"""# Train and Evaluate Models: Comparing Models

### Model Definitions
"""

# Define the models to compare
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': best_xgb,  # tuned XGBoost model
}

"""### Train, Evaluate and Generate Performance Metrics"""

import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize dictionaries to store MTTD and MTTR results
mttd_times = {}
mttr_times = {}

# Train and evaluate each model
for name, model in models.items():
    # Measure training time (MTTR)
    start_train = time.time()
    model.fit(X_train_final, y_train)
    end_train = time.time()
    mttr_times[name] = end_train - start_train
    print(f"\nModel: {name}")
    print(f"Mean Time to Respond (MTTR): {mttr_times[name]:.4f} seconds")

    # Measure prediction time per sample (MTTD)
    start_detect = time.time()
    y_pred = model.predict(X_test_final)
    end_detect = time.time()
    mttd_times[name] = (end_detect - start_detect) / len(X_test_final)
    print(f"Mean Time to Detect (MTTD): {mttd_times[name]:.6f} seconds per sample")

    # Calculate and print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Display MTTD and MTTR results for all models
print("\nCybersecurity Metrics Summary:")
for name in models.keys():
    print(f"{name} - Mean Time to Detect (MTTD): {mttd_times[name]:.6f} seconds per sample")
    print(f"{name} - Mean Time to Respond (MTTR): {mttr_times[name]:.4f} seconds")

"""# Predicitive System

### Predict on Training Data (Seen Data)
"""

# Predict on training data (seen data)
from sklearn.metrics import mean_squared_error

y_train_pred = best_xgb.predict(X_train_final)
mse = mean_squared_error(y_train, y_train_pred)
print(f"Mean Squared Error on the train set: {mse}")

print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Training Balanced Accuracy: {balanced_accuracy_score(y_train, y_train_pred):.4f}")
print(f"Training Precision: {precision_score(y_train, y_train_pred, average='weighted'):.4f}")
print(f"Training Recall: {recall_score(y_train, y_train_pred, average='weighted'):.4f}")
print(f"Training F1 Score: {f1_score(y_train, y_train_pred, average='weighted'):.4f}")

# Confusion Matrix for training data
cm_train = confusion_matrix(y_train, y_train_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=best_xgb.classes_, yticklabels=best_xgb.classes_)
plt.title('Confusion Matrix for XGBoost on Training Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""### Predict on Test Data (Unseen Data)"""

# Predict on test data (unseen data)
y_test_pred = best_xgb.predict(X_test_final)
mse = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error on the test set: {mse}")

print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Test Balanced Accuracy: {balanced_accuracy_score(y_test, y_test_pred):.4f}")
print(f"Test Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"Test Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"Test F1 Score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

# Confusion Matrix for test data
cm_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=best_xgb.classes_, yticklabels=best_xgb.classes_)
plt.title('Confusion Matrix for XGBoost on Test Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""### Entering New Unseen Data Manually (For Prediction)"""

original_features = scaler.feature_names_in_

new_data = {
    'User_Identity_Management': True,
    'Security_Policies': True,
    'Privileged_Access_Management': True,
    'User_Session_Management': True,
    'Time_Based_Access': False,
    'User_Behavior_Analytics': True,
    'Network_Security_Controls': True,
    'Access_Control_Lists': True,
    'Encryption_Policies': True,
    'Logging_and_Monitoring': True,
    'Security_Groups': False,
    'Identity_Federation': True,
    'Least_Privilege_Principle': True,
    'Access_Control_Propagation': True,
    'API_Access_Control': True,
    'Cloud_Workload_Identity': True,
    'Audit_Trails': True,
    'Access_Revocation': False,
    'Cross_Region_Access': False,
    'DLP_Policies': True,
    'Multi_Tenancy_Security': True,
    'Cloud_Orchestration_Layer_Security': True,
    'Token_based_Access_Control': True,
    'Granular_Access_Control': True,
    'Cloud_Native_Directory_Services': False,
    'Access_to_Logs_and_Monitoring_Tools': True,
    'Custom_Access_Control_Policies': True,
    'Zero_Trust_Architecture': True,
    'VPC_Controls': True,
    'Segmentation_of_Duties': False,
    'Instance_Metadata_Service_Access': False,
    'Shared_Responsibility_Model': True,
    'Cloud_Storage_Access_Policies': True,
    'API_Gateway_Security': True,
    'Dynamic_Access_Management': True,
    'Account_Lockout_Policies': True,
    'Access_to_Sensitive_Compute_Resources': True,
    'Penetration_Testing_and_Vulnerability_Assessments': True
}

new_data_df = pd.DataFrame([new_data])

# Ensure new_data_df contains only the features that were used during training
new_data_df = new_data_df[[f for f in original_features if f in new_data_df.columns]]

# Identify if any features used during training are missing in the new data
missing_features = set(original_features) - set(new_data_df.columns)

# Add missing features with default value (e.g., 0)
for feature in missing_features:
    new_data_df[feature] = 0

# Reorder the columns in new_data_df to match the original feature order during training
new_data_df = new_data_df[original_features]

# Now you can scale the data using the same scaler
new_data_scaled = scaler.transform(new_data_df)
new_data_selected = new_data_df[important_features]
new_data_final = new_data_selected.drop(columns=low_importance_features)


# Predict the security score using the trained XGBoost model
predicted_security_score = best_xgb.predict(new_data_final)

# Output the predicted security score
print(f"Predicted Security Score: {predicted_security_score[0]}")

"""# Remediation to Improve Security Score"""

def required_remediations(input_df, original_features):
    protocol_suggestions = {
        'User_Identity_Management': "Adopt robust user identity management practices.",
        'Security_Policies': "Implement comprehensive security policies to guide access control.",
        'Privileged_Access_Management': "Implement privileged access management strategies.",
        'User_Session_Management': "Enhance user session management to control active sessions effectively.",
        'Time_Based_Access': "Adopt time-based access control for additional security.",
        'User_Behavior_Analytics': "Use user behavior analytics to detect and respond to anomalies.",
        'Network_Security_Controls': "Enhance network security controls, such as firewalls.",
        'Access_Control_Lists': "Ensure proper configuration of access control lists (ACLs).",
        'Encryption_Policies': "Strengthen encryption policies to secure data in transit and at rest.",
        'Logging_and_Monitoring': "Implement robust logging and monitoring practices for better audit trails.",
        'Security_Groups': "Review and optimize security group settings for improved isolation.",
        'Identity_Federation': "Adopt identity federation to enable seamless cross-domain access.",
        'Least_Privilege_Principle': "Ensure the principle of least privilege is adhered to for user roles.",
        'Access_Control_Propagation': "Ensure proper propagation of access control policies across cloud services.",
        'API_Access_Control': "Implement strict API access control to secure data and services.",
        'Cloud_Workload_Identity': "Adopt workload identity solutions for cloud resources.",
        'Audit_Trails': "Maintain detailed audit trails to monitor access and changes.",
        'Access_Revocation': "Improve processes for quick and effective access revocation.",
        'Cross_Region_Access': "Control and monitor cross-region access to resources.",
        'DLP_Policies': "Enforce data loss prevention (DLP) policies to protect sensitive data.",
        'Multi_Tenancy_Security': "Enhance security measures for multi-tenant environments.",
        'Cloud_Orchestration_Layer_Security': "Secure the cloud orchestration layer for better control.",
        'Token_based_Access_Control': "Use token-based access control for secure session management.",
        'Granular_Access_Control': "Implement granular access control for precise permissions.",
        'Cloud_Native_Directory_Services': "Adopt cloud-native directory services for better identity management.",
        'Access_to_Logs_and_Monitoring_Tools': "Ensure controlled access to logs and monitoring tools.",
        'Custom_Access_Control_Policies': "Develop custom access control policies for unique needs.",
        'Zero_Trust_Architecture': "Adopt a zero-trust architecture for better access control.",
        'VPC_Controls': "Implement VPC (Virtual Private Cloud) controls to enhance network security.",
        'Segmentation_of_Duties': "Ensure proper segmentation of duties to prevent conflicts of interest.",
        'Instance_Metadata_Service_Access': "Restrict access to instance metadata service for enhanced security.",
        'Shared_Responsibility_Model': "Clarify shared responsibility model roles to manage risks.",
        'Cloud_Storage_Access_Policies': "Strengthen access policies for cloud storage.",
        'API_Gateway_Security': "Secure API gateways to control access to backend services.",
        'Dynamic_Access_Management': "Implement dynamic access management based on risk levels.",
        'Account_Lockout_Policies': "Adopt account lockout policies to prevent brute force attacks.",
        'Access_to_Sensitive_Compute_Resources': "Control access to sensitive compute resources.",
        'Penetration_Testing_and_Vulnerability_Assessments': "Conduct regular penetration testing and vulnerability assessments."
    }

    for index, row in input_df.iterrows():
        suggestions = []
        for feature, suggestion in protocol_suggestions.items():
            # Check if the protocol is absent or not implemented (assuming 0 means not implemented)
            if feature in original_features and feature in row.index and row[feature] == 0:
                suggestions.append(feature+" : "+suggestion)

        if suggestions:
            print(f"By implementing, enabling, or enhancing the features listed below for the given access control data, a higher security score can be achieved.")
            for idx, suggestion in enumerate(suggestions, start=1):
                print(f"{idx}. {suggestion}")
        else:
            print(f"All key protocols for input at index {index} are well-configured. No significant improvements needed.")

required_remediations(new_data_df, original_features)