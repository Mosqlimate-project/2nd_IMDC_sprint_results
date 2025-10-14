import numpy as np
from mosqlient.scoring import compute_wis

def compute_metrics(model, df_w, df_preds_):
    '''
    Function to compute the WIS normalized by the
    total the cases notified in the period.

    Parameters: 
    `model`: int or str. name of the model; 
    `df_w`: pd.DataFrame. dataframe with the cases. It must contain the columns: date, uf, casos
    `df_preds`: pd.DataFrame. `dataframe with the predictions. It must contain the columns: 
               date, state, lower_95, lower_90, lower_80, lower_50, pred,
               upper_50, upper_80, upper_90, upper_95. 
    `state`: str. state to compute the score;

    '''
        
    df_preds_model = df_preds_.loc[(df_preds_.model_id == model)].reset_index(drop = True)
        
    df_preds_to_score = df_w.merge(df_preds_model, left_on = ['date', 'uf'], right_on = ['date', 'state'])

    wis = np.sum(compute_wis( 
                        df_preds_to_score[['date',  'lower_95', 'lower_90', 'lower_80', 'lower_50',
                           'pred', 'upper_50', 'upper_80', 'upper_90', 'upper_95']],
                        observed_value = df_preds_to_score['casos'].values)) 
    
    wis_norm = wis/np.sum(df_preds_to_score['casos'].values)

    return wis_norm


def compute_ss(df_piv, col1, col_ref ):

    ss =  1- (df_piv[col1].values/df_piv[col_ref].values)

    return ss