import joblib
import numpy as np

def get_input(feature_name):
    """
    Safely get user input for a feature.
    :param feature_name: Name of the feature for which input is required
    :return: Tuple of value and null status for the feature
    """
    while True:
        try:
            value = float(input(f"Enter value of {feature_name}: "))
            is_null = input(f"Is it null for {feature_name}? (yes/no): ").lower() == 'yes'
            return value, float(is_null)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def predict_cancer(input_data):
    """
    Predicts cancer diagnosis based on input data.
    :param input_data: User input data
    :return: Diagnosis result
    """
    clf = joblib.load("finalized_modelLayer1.sav")
    x_test = np.array(input_data).reshape(1, -1)
    x1 = clf.predict(x_test)
    x_test2 = np.concatenate((x_test, np.array([[x1[0]]])), axis=1)

    clf2 = joblib.load("finalized_modelLayer2.sav")
    x2 = clf2.predict(x_test2)
    return "CANCER" if x2[0] == 1 else "NO CANCER"

# List of all different inputs
feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
                 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 
                 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
                 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
                 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
                 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
                 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

input_data = []
for feature in feature_names:
    value, is_null = get_input(feature)
    input_data.extend([value, is_null])

diagnosis = predict_cancer(input_data)
print(f"\n\t\t\t\t\t\tRESULTS ARE: {diagnosis}")
