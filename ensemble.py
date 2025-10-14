import pandas as pd 

def compute_ensemble(df_preds, models = None): 


    if models is not None: 
        df_preds = df_preds.loc[df_preds.model_id.isin(models)]
        
    list_dfs = []
    
    for col in ['lower_95', 'lower_90', 'lower_80', 'lower_50', 'pred',
           'upper_50', 'upper_80', 'upper_90', 'upper_95']:

        try: 
            list_dfs.append(pd.DataFrame(df_preds.pivot(index = ['date', 'state', 'valid_test'],
                                                    columns = 'model_id', values = col).median(axis =1)).rename(columns = {0: col}))
        except: 
            list_dfs.append(pd.DataFrame(df_preds.pivot(index = ['date', 'state'],
                                                    columns = 'model_id', values = col).median(axis =1)).rename(columns = {0: col}))


    df_median_ens = pd.concat(list_dfs ,axis =1).reset_index()
    
    quantile_order = (
    (df_median_ens['lower_95'] <= df_median_ens['lower_90']) &
    (df_median_ens['lower_90'] <= df_median_ens['lower_80']) &
    (df_median_ens['lower_80'] <= df_median_ens['lower_50']) &
    (df_median_ens['lower_50'] <= df_median_ens['pred']) &
    (df_median_ens['pred']     <= df_median_ens['upper_50']) &
    (df_median_ens['upper_50'] <= df_median_ens['upper_80']) &
    (df_median_ens['upper_80'] <= df_median_ens['upper_90']) &
    (df_median_ens['upper_90'] <= df_median_ens['upper_95'])
    )

    if ~quantile_order.all(): 
        raise Exception("The ensemble includes quantile values that violate monotonicity.")

    return df_median_ens
    