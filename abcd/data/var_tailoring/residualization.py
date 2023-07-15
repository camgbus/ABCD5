"""Residualize cofounding effects.
"""
import numpy as np
import statsmodels.api as sm

def residualize_age_sex(df, var, new_var_name=None, age_var="interview_age", 
                        sex_var="kbi_sex_assigned_at_birth"):
    
    if not new_var_name:
        new_var_name = var
        
    mean_age = np.mean(df[age_var])
    reference_sex = 1.0 # 1.0 is ,, 2.0 is f
        
    endog = df[var]  # endog: The variable we want to model
    
    # Intercept/constant: the mean value of the response variable when all predictor variables 
    # in the model are equal to zero
    df["Intercept"] = 1
    
    exog = df[["Intercept", sex_var, age_var]] # exog:  the covariates alias explanatory variables
    
    # Generalized linear model
    md = sm.GLM(endog, exog, family=sm.families.Gaussian())

    # Fit the regression model
    md = md.fit()
    print(md.summary())

    # Calculate residual variables that end in 'prime'
    df[new_var_name] = df[var] - (md.params[1] * (df[sex_var] - reference_sex) + md.params[2] * (df[age_var] - mean_age))
    
    return df
