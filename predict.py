import joblib
import shap

with open('Model/sscalerf.pkl', 'rb') as f:
    scaler = joblib.load(f)

with open('Data/ex_data.pkl', 'rb') as g:
    ex_data = joblib.load(g)




def get_prediction(data,model):
    ans = ['Full', 'No']
    X = scaler.transform(data)
    res = model.predict(X)
    return X, ans[int(res)]


def explain_model_prediction(data,nn,features):
    # Calculate Shap values
    #data = pd.DataFrame(data=data, columns=features.tolist(),index=range(1))
    nne = shap.KernelExplainer(nn.predict,ex_data)
    shap_values = nne.shap_values(data)
    #p = shap.force_plot(nne.expected_value, shap_values[0], data)
    p = shap.force_plot(nne.expected_value, shap_values[0][:], data, feature_names =features.tolist(), plot_cmap="PkYg")
    return p, shap_values
