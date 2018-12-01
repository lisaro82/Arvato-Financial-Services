import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import progressbar
from sklearn.decomposition import PCA

#------------------------------------------------------------------------------------------------------------------------------------
def utl_pcaResults(p_pca, p_columns, p_ShowTop = 10):
    '''
    INPUT:
        p_pca            - trained PCA object
        p_columns        - feature names
        p_ShowTop        - number of weights to be displayed
    OUTPUT:
        Returns the calculated weights for all dimensions, and the report for the most important ones.
    '''
    
    def getDimWeight(p_results, p_dimensionNo, p_ShowTop):
        '''
        INPUT:
            p_results        - (pandas dataframe) daraframe containing the calculated weigths for all dimensions 
            p_dimensionNo    - dimension number for which the weights will be extracted
            p_ShowTop        - number of weights to be displayed
        OUTPUT:
            Returns the weights for the given dimension.
        '''
        v_df = pd.DataFrame(p_results.loc[p_results.index[p_dimensionNo - 1]])
        v_df['weight'] = v_df[v_df.columns.values[0]].apply(np.abs)  
        v_df.loc['00_Component Variance', 'weight'] = 1
        v_df.loc['00_Cumulated Variance', 'weight'] = 1
        return pd.DataFrame(v_df.sort_values('weight', ascending = False)[v_df.columns.values[0]]).head(p_ShowTop)

    v_idx = ['Dimension {}'.format(idx + 1) for idx in range(len(p_pca.components_))]
    
    # PCA components
    v_comp = pd.DataFrame(np.round(p_pca.components_, 4), columns = p_columns)
    v_comp.index = v_idx
    
    # PCA explained variance
    v_ratios = p_pca.explained_variance_ratio_.reshape(len(p_pca.components_), 1)
    v_variance = pd.DataFrame(np.round(v_ratios, 4), columns = ['00_Component Variance'])
    v_variance.index = v_idx
    
    v_results = v_comp.merge(v_variance, left_index = True, right_index = True)
    v_idx = v_results.index.values
    for idx in range(len(v_idx)):
        if idx == 0:
            v_results.loc[v_idx[idx], '00_Cumulated Variance'] = v_results.loc[v_idx[idx], '00_Component Variance']
        else:
            v_results.loc[v_idx[idx], '00_Cumulated Variance'] = v_results.loc[v_idx[idx], '00_Component Variance'] \
                                                                   + v_results.loc[v_idx[idx - 1], '00_Cumulated Variance']
    
    v_n_components = 7 if len(p_pca.components_) > 6 else len(p_pca.components_)
    v_dim = getDimWeight(v_results, 1, p_ShowTop)
    for idx in range(2, v_n_components):
        v_dim = v_dim.merge(getDimWeight(v_results, idx, p_ShowTop), how = 'outer', left_index = True, right_index = True)
        
    return v_results, v_dim


#------------------------------------------------------------------------------------------------------------------------------------
def utl_applyPCA(p_data, p_n_components = None, p_ShowWeights = False, p_ShowTop = 10, p_figHeight = 20):
    '''
    INPUT:
        p_data           - (pandas dataframe) the dataframe on which the PCA will be applied
        p_n_components   - (int, float, None or string) number of components to keep. Parameter will be passed to PCA()
        p_ShowWeights    - flag indicating if the heatmap report for the dimensions weights should be showed
        p_ShowTop        - number of weights to be displayed
    OUTPUT:
        Returns the trained PCA.
    '''
    # Apply PCA to the data.
    v_pca = PCA(p_n_components)
    X_pca = v_pca.fit_transform(p_data)
    
    # Investigate the variance accounted for by each principal component.
    v_results, v_dim = utl_pcaResults(v_pca, p_data.columns.values, p_ShowTop)
    v_display = v_results['00_Cumulated Variance'].copy().reset_index()
    fig, ax = plt.subplots(figsize = (16, 6))
    plt.plot( v_display.index.values + 1, 
              v_display['00_Cumulated Variance'], 
              marker = 'o', markersize = 10 )
    plt.grid(True)
    plt.show()
    
    # Display the weight for the dimensions
    if p_ShowWeights:
        v_display = ( v_dim.reset_index()
                           .sort_values('index', ascending = True) 
                           .set_index('index') ) 

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, p_figHeight))
        mask = np.zeros(v_display.shape)
        mask[v_display < 0] = 1
        sns.heatmap( v_display, annot = True, cmap = "Blues", vmin = 0.05, vmax = 0.15, 
                     linewidths=.5, ax = ax1, robust = True, cbar = False, mask = mask )

        mask = np.zeros(v_display.shape)
        mask[v_display > 0] = 1
        sns.heatmap( v_display.apply(np.abs), annot = True, cmap = "Reds", vmin = 0.05, vmax = 0.15, 
                     linewidths=.5, ax = ax2, robust=True, cbar = False, mask = mask )
        ax1.xaxis.tick_top()
        ax2.xaxis.tick_top()
        ax2.set_yticks([])
        plt.show()
    return v_pca