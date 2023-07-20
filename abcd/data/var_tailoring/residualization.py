"""Residualize cofounding effects.
"""
import numpy as np
import statsmodels.api as sm

def residualize(df, var, new_var_name=None, covs = ["interview_age", "kbi_sex_assigned_at_birth"],
                references = {"kbi_sex_assigned_at_birth": 1.0}, verbose=False):
                
    if not new_var_name:
        new_var_name = var
    
    # If a reference is not given, replace by the mean
    for cov in covs:
        if cov not in references:
            references[cov] = np.mean(df[cov])
            
    # Endog: The variable we want to model
    endog = df[var] 
    
    # Intercept/constant: the mean value of the response variable when all predictor variables 
    # in the model are equal to zero
    df["Intercept"] = 1
    
    # Exog:  the covariates alias explanatory variables
    exog = df[["Intercept"] + covs] 
    
    # Generalized linear model
    md = sm.GLM(endog, exog, family=sm.families.Gaussian())

    # Fit the regression model
    md = md.fit()
    if verbose:
        print(md.summary())

    # Calculate residual variables
    diff = sum(md.params[cov_ix+1] * (df[cov] - references[cov]) for cov_ix, cov in enumerate(covs))
    df[new_var_name] = df[var] - diff
    
    return df