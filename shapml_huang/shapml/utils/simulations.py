import smote_variants as sv
import pandas as pd
import logging
import numpy as np

def generate_sythetic_minority_class_data(df, target, oversampler_name='OUPS', desired_rows=5000, verbose=False, random_state=0, **kwargs):
    """
    oversampler_name (str):  one of: ['OUPS', 'NT_SMOTE', 'ROSE', 'NDO_sampling', 'Borderline_SMOTE1', 'SMOTE', 'Borderline_SMOTE2', 'SMOTE_OUT', 'SN_SMOTE', 'Selected_SMOTE', 'distance_SMOTE', 'Gaussian_SMOTE', 'Random_SMOTE', 'SL_graph_SMOTE', 'CURE_SMOTE']
    kwargs : kwargs for SMOTE object
    """
    logging.getLogger(sv.__name__).setLevel(logging.CRITICAL)
    imputedDF = df.copy()
    discrete_columns = list(imputedDF.columns[np.all(imputedDF.applymap(lambda x: float(x).is_integer() or np.isnan(float(x))), axis=0)])
    for col in discrete_columns:
        imputedDF[col] = imputedDF[col].astype(int)
    imputedDF = imputedDF.fillna(imputedDF.median())
    X = imputedDF.drop(columns=target); y=imputedDF[target]
    get_oversampler_name = lambda x: str(x).split('.')[-1].split("'")[0]
    oversamplers= [sv.OUPS, sv.NT_SMOTE, sv.ROSE, sv.NDO_sampling, sv.Borderline_SMOTE1,
                  sv.SMOTE, sv.Borderline_SMOTE2, sv.SMOTE_OUT, sv.SN_SMOTE, 
                   sv.Selected_SMOTE, sv.distance_SMOTE, sv.Gaussian_SMOTE,
                  sv.Random_SMOTE, sv.SL_graph_SMOTE, sv.CURE_SMOTE]
    oversampler_d = {get_oversampler_name(m):m for m in oversamplers}
    try: 
        assert oversampler_name in oversampler_d
    except AssertionError:
        print(f"oversampler_name must be one of: {list(oversampler_d.keys())}")

    if verbose:
        print(f"Generating synthetic data using {oversampler_name} to create dataset with {desired_rows} rows")
    n_rows = 0   
    r=0
    while n_rows < desired_rows:
        oversampler= oversampler_d[oversampler_name](random_state=random_state*1000+r, proportion = 1, **kwargs)
        if verbose>1:
            print("Updated params: ", oversampler.get_params()) 
        X_samp, y_samp= oversampler.sample(X.values, y.values)
        if np.all(X_samp[:X.shape[0], :] == X.iloc[:X.shape[0],:].values):
            if verbose>1:
                print('original data was in the beginning')
#             X_samp, y_samp=X_samp[-(X_samp.shape[0]-X.shape[0]):, :], y_samp[-(X_samp.shape[0]-X.shape[0]):] # original data is put in the beginning
            X_samp, y_samp=X_samp[X.shape[0]:, :], y_samp[X.shape[0]:] # original data is put in the beginning
        elif np.all(X_samp[X.shape[0], :] == X.iloc[0,:].values):
            import pdb; pdb.set_trace()
            if verbose>1:
                print('original data is at the end the end')
            X_samp, y_samp=X_samp[:-X.shape[0], :], y_samp[:-y.shape[0]] # original data is put in the beginning
        else:
            import pdb; pdb.set_trace()
        if r ==0:
            X_synth, y_synth=X_samp, y_samp
        else:
            X_synth, y_synth= np.concatenate([X_synth,X_samp]), np.concatenate([y_synth,y_samp])
        if verbose>1: 
            print('Synthetic: ', X_synth.shape, y_synth.mean())
        n_rows = X_synth.shape[0]
        r+=1
    syntheticDF = pd.DataFrame(np.concatenate([X_synth[:desired_rows, :], y_synth[:desired_rows].reshape(-1,1)],axis=1), columns = imputedDF.columns)
    try:
        assert syntheticDF.duplicated().sum()==0
    except:
        import pdb; pdb.set_trace()
    
    
    if verbose>1: 
        print('Orig:      ', X.shape, y.mean())
        print('Synthetic: ', X_synth.shape, y_synth.mean())
        print("Duplicated: ", syntheticDF.duplicated().sum())
    return syntheticDF

def generate_synthetic_binary_classification_df(df, target, oversampler_name='OUPS', desired_rows=5000, verbose=False, random_state=0, **kwargs):
    """
    oversampler_name (str):  one of: ['OUPS', 'NT_SMOTE', 'ROSE', 'NDO_sampling', 'Borderline_SMOTE1', 'SMOTE', 'Borderline_SMOTE2', 'SMOTE_OUT', 'SN_SMOTE', 'Selected_SMOTE', 'distance_SMOTE', 'Gaussian_SMOTE', 'Random_SMOTE', 'SL_graph_SMOTE', 'CURE_SMOTE']
    kwargs : kwargs for SMOTE object
    """
    proportion_pos = np.round(df[target].mean(),2)
    if proportion_pos < .5: 
        majority_class=0
    else:
        majority_class=1
    minorityDF = generate_sythetic_minority_class_data(df, target=target, oversampler_name=oversampler_name, desired_rows=int(10*desired_rows/2), verbose=False, random_state=random_state, **kwargs)
    interim_df = pd.concat([minorityDF, df[df[target]==majority_class]])
    majorityDF = generate_sythetic_minority_class_data(interim_df, target=target, oversampler_name=oversampler_name, desired_rows=int(10*desired_rows/2), verbose=False, random_state=random_state, **kwargs)
    syntheticDF=pd.concat([minorityDF, majorityDF])
    syntheticDF = syntheticDF.sample(n=desired_rows, weights=syntheticDF[target].apply(lambda x: proportion_pos if x == 1 else 1-proportion_pos), replace=False).reset_index(drop=True)
    if verbose:
        print("Final synthetic data has {} rows with {:.2f}% positive class".format(desired_rows, syntheticDF[target].mean()*100))
    return syntheticDF