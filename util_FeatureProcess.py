import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import progressbar

from util_Generic import printmd, printHeader, formatLabel
from util_Generic import utl_toInteger, utl_isInteger
from util_Generic import utl_reduceMemoryUsage

#------------------------------------------------------------------------------------------------------------------------------------
def utl_splitMixedColumnValues(p_column, p_values, p_valueUnknown):
    def extendKeys(p_keys, p_valueUnknown = p_valueUnknown):        
        for key, value in p_values.items():
            if 'unknown' in p_values[key]['00_Meaning']:
                p_values[key]['01_Columns'] = {key: p_valueUnknown for key in p_keys.keys()}
                continue            
            for nkey in p_keys.keys():
                if nkey not in p_values[key]['01_Columns'].keys():
                    p_values[key]['01_Columns'][nkey] = 0
        return
    
    
    #-------------------------------------------------------------------------------------------------------------------
    if p_column == 'PRAEGENDE_JUGENDJAHRE':  
        v_keys = {}
        for key, value in p_values.items():
            if 'unknown' in p_values[key]['00_Meaning']: continue
            v_meaning = ' |*| '.join([item for item in p_values[key]['00_Meaning'].split('ies - ')])
            v_meaning = [item for item in v_meaning.split(' (')]
            v_tmp = [item for item in v_meaning[1].split(',')]
            v_meaning[1] = v_tmp[0]
            v_meaning.append(v_tmp[1].replace(')', ''))
            v_meaning = ' |*| '.join([str(item).strip() for item in v_meaning]).split(' |*| ')
            p_values[key]['01_Columns'] = { f'_Value_{idx}': v_meaning[idx] for idx in range(len(v_meaning)) }   
            for key in p_values[key]['01_Columns'].keys():
                if not key in v_keys.keys():
                    v_keys[key] = p_valueUnknown
                    
        extendKeys(v_keys)            
            
    #-------------------------------------------------------------------------------------------------------------------
    elif p_column == 'ALTER_HH':  
        v_keys = {} 
        for key, value in p_values.items():
            if 'unknown' in p_values[key]['00_Meaning']: continue                
            v_meaning = [ item.replace('01.01.', '').replace('31.12.', '').strip() 
                             for item in p_values[key]['00_Meaning'].split(' bis ') ] 
            
            v_meaning[0] = (int(v_meaning[0]) - 1895 + 1) / 10
            v_meaning[1] = (int(v_meaning[1]) - 1895 + 1) / 10
            
            p_values[key]['01_Columns'] = { f'_Value_{idx}': v_meaning[idx] for idx in range(len(v_meaning)) }
            for key in p_values[key]['01_Columns'].keys():
                if not key in v_keys.keys():
                    v_keys[key] = p_valueUnknown
                    
        extendKeys(v_keys)  
    
    #-------------------------------------------------------------------------------------------------------------------
    elif p_column == 'CAMEO_DEUINTL_2015':   
        v_keys = {}
        for key, value in p_values.items():
            if 'unknown' in p_values[key]['00_Meaning']: continue                
            v_meaning = [ item.strip() 
                             for item in p_values[key]['00_Meaning'].upper()
                                                                    .replace('  ', ' ')
                                                                    .replace('WITH', '&')
                                                                    .split('-') ]               
            v_tmp = [item for item in '-'.join(v_meaning[1:]).split('&')]
            v_tmp.append(v_meaning[0])
            v_tmp = [ item.strip() for item in v_tmp ]
            
            if len(v_tmp) == 3: v_meaning = [ v_tmp[0] + ' - ' + v_tmp[1], v_tmp[2] ]
            else: v_meaning = [ v_tmp[0], v_tmp[1] ]

            p_values[key]['01_Columns'] = { f'_Value_{idx}': v_meaning[idx] for idx in range(len(v_meaning)) }
        
        extendKeys(v_keys)
    
    #-------------------------------------------------------------------------------------------------------------------
    elif p_column in [ 'D19_BANKEN_DATUM',  'D19_BANKEN_ONLINE_DATUM',  'D19_BANKEN_OFFLINE_DATUM', 
                       'D19_GESAMT_DATUM',  'D19_GESAMT_ONLINE_DATUM',  'D19_GESAMT_OFFLINE_DATUM',
                       'D19_TELKO_DATUM',   'D19_TELKO_ONLINE_DATUM',   'D19_TELKO_OFFLINE_DATUM',
                       'D19_VERSAND_DATUM', 'D19_VERSAND_ONLINE_DATUM', 'D19_VERSAND_OFFLINE_DATUM', ]:    
        v_keys = {}
        v_no_transactions = None
        for key, value in p_values.items():
            if p_values[key]['00_Meaning'].upper().strip() == 'NO TRANSACTIONS KNOWN': 
                v_no_transactions = key
                p_values[v_no_transactions]['01_Columns'] = {}
                continue      
            
            p_values[key]['01_Columns'] = {}            
            v_meaning = p_values[key]['00_Meaning'].upper().strip()
            
            v_key = '_Value_INCREASE'
            v_keys[v_key] = None
            for item in [ ( 'SLIGHTLY INCREASED',  1 ), 
                          ( 'INCREASED',           2 ), 
                          ( 'HIGHEST',             5 ), 
                          ( 'VERY HIGH',           4 ), 
                          ( 'HIGH',                3 ) ]:
                if item[0] in v_meaning:
                    p_values[key]['01_Columns'][v_key] = item[1]
                    break
                    
            v_key = '_Value_ACTIVITY_WITHIN_MONTHS'
            v_keys[v_key] = None
            if p_column in ['D19_BANKEN_ONLINE_DATUM', 'D19_BANKEN_OFFLINE_DATUM']:
                for item in [ ( 'LAST 12 MONTHS',         1 ), 
                              ( 'ELDER THAN 12 MONTHS',   2 ), 
                              ( 'ELDER THAN 18 MONTHS',   3 ), 
                              ( 'ELDER THAN 24 MONTHS',   4 ), 
                              ( 'ELDER THAN 36 MONTHS',   5 ) ]:
                    if item[0] in v_meaning:
                        p_values[key]['01_Columns'][v_key] = item[1]
                        break
            else:
                for item in [ ( 'LAST 12 MONTHS',       1 ), 
                              ( 'ELDER THAN 1 YEAR',    2 ), 
                              ( 'ELDER THAN 1,5 YEARS', 3 ), 
                              ( 'ELDER THAN 2 YEARS',   4 ), 
                              ( 'ELDER THAN 3 YEARS',   5 ) ]:
                    if item[0] in v_meaning:
                        p_values[key]['01_Columns'][v_key] = item[1]
                        break
                    
        extendKeys(v_keys, 0)
    
    #-------------------------------------------------------------------------------------------------------------------
    elif p_column in [ 'LP_LEBENSPHASE_GROB', 'LP_LEBENSPHASE_FEIN', 
                       'LP_STATUS_GROB', 'LP_STATUS_FEIN',
                       'LP_FAMILIE_GROB', 'LP_FAMILIE_FEIN' ]:   
        def createKey(p_key, p_valuesList, p_meaning):
            if not p_key in v_keys.keys(): v_keys[p_key] = []
            
            p_values[key]['01_Columns'][p_key] = '__NONE'
            
            for item in p_valuesList:                                
                if item in p_meaning:
                    if p_values[key]['01_Columns'][p_key] == '__NONE':
                        p_values[key]['01_Columns'][p_key] = item
                    else:
                        p_values[key]['01_Columns'][p_key] += ' - ' + item
                    p_meaning = p_meaning.replace(f'{item}S', '*')
                    p_meaning = p_meaning.replace(item, '*')
            
            v_keys[p_key].append(p_values[key]['01_Columns'][p_key])
            
            return p_meaning
        
        v_keys = {}         
        for key, value in p_values.items():              
            p_values[key]['01_Columns'] = {}            
            v_meaning = ( p_values[key]['00_Meaning'].upper()
                                                     .replace('FAMILIES', 'FAMILY')
                                                     .replace('HOMEOWNER', 'HOUSEOWNER')
                                                     .replace('TITLE HOLDER-HOUSEHOLDS', 'TITLE HOLDER HOUSEOWNER')
                                                     .replace('VILLAGERS', 'LOW-INCOME VILLAGERS')
                                                     .replace('MULITPERSON', 'MULTIPERSON').strip() )
            
            v_meaning = createKey( p_key        = '_Value_INCOME', 
                                   p_valuesList = [ 'LOW-INCOME', 'AVERAGE EARNER', 'HIGH-INCOME', 'INDEPENDANT', 'HOUSEOWNER', 
                                                    'TOP EARNER' ],
                                   p_meaning    = v_meaning ) 
            v_meaning = v_meaning.replace('EARNERS', '').replace('EARNER', '')
            
            v_meaning = createKey( p_key        = '_Value_AGE', 
                                   p_valuesList = [ 'YOUNGER AGE', 'MIDDLE AGE', 'HIGHER AGE', 'ADVANCED AGE', 
                                                    'RETIREMENT AGE' ],
                                   p_meaning    = v_meaning )  
            
            v_meaning = createKey( p_key        = '_Value_FAMILY', 
                                   p_valuesList = [ 'SINGLE', 'COUPLE', 'PARENT', 'FAMILY', 'MULTIPERSON HOUSEHOLD', 'PERSON' ],
                                   p_meaning    = v_meaning )  
            
            v_meaning = createKey( p_key        = '_Value_GENERATIONAL', 
                                   p_valuesList = [ 'TWO-GENERATIONAL', 'MULTI-GENERATIONAL', 'SHARED FLAT' ],
                                   p_meaning    = v_meaning )   
            
            v_meaning = createKey( p_key        = '_Value_CHILD', 
                                   p_valuesList = [ 'CHILD OF FULL AGE', 'TEENAGER', 'YOUNG' ],
                                   p_meaning    = v_meaning )     
            
            v_key = '_Value_OTHER'
            if not v_key in v_keys.keys(): v_keys[v_key] = []
            p_values[key]['01_Columns'][v_key] = ( v_meaning.replace('*', '').replace('-', '')
                                                            .replace(' OF ', '').replace(' AT ', '').replace(' AND ', '')
                                                            .replace(' FROM ', '')
                                                            .replace('  ', '')
                                                            .strip() )
            if p_values[key]['01_Columns'][v_key] == '':
                p_values[key]['01_Columns'][v_key] = '__NONE'
            v_keys[v_key].append(p_values[key]['01_Columns'][v_key])
        
        v_tmp = {key: ' | '.join(sorted(set(value))) for key, value in v_keys.items()}
        
        v_keys = {}
        for key, value in v_tmp.items():
            if value != '__NONE':
                v_keys[key] = len(value)
            else:
                for ikey in p_values.keys():
                    p_values[ikey]['01_Columns'].pop(key, None)
        
        v_keys = {key: 0 for key in v_keys.keys() if v_keys[key] > 0}            
        extendKeys(v_keys, 0) 
    
    return p_values


#------------------------------------------------------------------------------------------------------------------------------------
def utl_cleanOnDefinition(p_label, p_data, p_featuresDef, p_showHeader = False):
    if p_showHeader:
        printHeader(f'Clean dataframe {formatLabel(p_label)} based on definition.')
    
    v_data = p_data.copy()    
    
    # Rename the column in order to match the name in the definition file
    v_data = v_data.rename(columns = {'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
    
    printmd(f'Re-encode columns values based on definition.')
    with progressbar.ProgressBar(max_value = len(p_featuresDef.keys())) as bar:
        v_count = 0
        for columnKey in p_featuresDef.keys():
            if columnKey in v_data.columns:
                # Replace value XX from column CAMEO_DEUINTL_2015 to unknown
                if columnKey == 'CAMEO_DEUINTL_2015':
                    v_data[columnKey] = v_data[columnKey].replace({'XX': -1})
                # Replace value X from column CAMEO_DEUG_2015 to unknown
                elif columnKey == 'CAMEO_DEUG_2015':
                    v_data[columnKey] = v_data[columnKey].replace({'X': -1})
                
                # Calculate the new map to be used for the values which needs to be re-encoded
                v_map = { str(key): str(value['Value Converted'])
                              for key, value in p_featuresDef[columnKey]['Values'].items() }                
                v_map = { key:value for key, value in v_map.items() if key != value}
                
                # If we have data in the map, than we replace the values
                if len(v_map.keys()) > 0:
                    try:
                        v_data[columnKey] = v_data[columnKey].astype(np.float16)
                        v_map = {np.float16(key): np.float16(value) for key, value in v_map.items()}
                    except:
                        raise
                    v_data[columnKey] = v_data[columnKey].replace(v_map)
                    
                # If columns exists in the definition for a split operation, do the split
                if 'Split' in p_featuresDef[columnKey].keys():
                    v_columns = {key : value['01_Columns'] for key, value in p_featuresDef[columnKey]['Split'].items()}
                    v_columns = pd.DataFrame(v_columns).T
                    v_columns.columns = [f'{columnKey}{item}' for item in v_columns.columns]
                    v_columns.reset_index(inplace = True)                    
                    try:
                        v_data[columnKey] = v_data[columnKey].astype(np.float16)
                        v_columns['index'] = v_columns['index'].astype(v_data[columnKey].dtype)
                        v_data = v_data.merge(v_columns, how = 'left', left_on = columnKey, right_on = 'index' )
                        v_data.drop([columnKey, 'index'], axis = 1, inplace = True)
                    except: 
                        print(f' *** ERROR *** Could not convert column {columnKey} to float.')
                        print(f'     - Column type: {v_data[columnKey].dtype}.')
                        print(f'     - Column values:    {v_data[columnKey].value_counts().index.tolist()}.')
                        print(f'     - Converted values: {v_columns["index"].tolist()}.')
                        
            v_count += 1
            bar.update(v_count) 
    
    printmd(f'Transform object columns to numeric.')
    with progressbar.ProgressBar(max_value = len(v_data.select_dtypes(include=['object']).columns)) as bar:
        v_count = 0                         
        for column in v_data.select_dtypes(include=['object']).columns:
            try:
                v_data[column] = v_data[column].astype(np.float16)
            except: None                         
            v_count += 1
            bar.update(v_count) 
    
    utl_reduceMemoryUsage(p_label, v_data)    
    return v_data


#------------------------------------------------------------------------------------------------------------------------------------
def utl_processColumns(p_data, p_column):
    
    # Column 'EINGEFUEGT_AM' contains an encoding of type timestamp, so we will convert it to a timestamp and extract 
    # its components.
    if p_column == 'EINGEFUEGT_AM':
        p_data[p_column] = pd.to_datetime(p_data[p_column])
        utl_SplitDate(p_data, p_column)
        p_data.drop(p_column, axis = 1, inplace = True)
    
    elif p_column in [ 'LP_LEBENSPHASE_GROB_Value_INCOME', 'LP_LEBENSPHASE_FEIN_Value_INCOME', 
                       'LP_STATUS_GROB_Value_INCOME', 'LP_STATUS_FEIN_Value_INCOME',
                       'LP_FAMILIE_GROB_Value_INCOME', 'LP_FAMILIE_FEIN_Value_INCOME' ]:
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  '__NONE':                           0,
                                                                  'LOW-INCOME':                       1, 
                                                                  'AVERAGE EARNER':                   2, 
                                                                  'LOW-INCOME - AVERAGE EARNER':      3,
                                                                  'HIGH-INCOME':                      4, 
                                                                  'INDEPENDANT':                      5, 
                                                                  'HOUSEOWNER':                       6, 
                                                                  'TOP EARNER':                       7 }).astype(int)
    
    elif p_column in [ 'LP_LEBENSPHASE_GROB_Value_AGE', 'LP_LEBENSPHASE_FEIN_Value_AGE', 
                       'LP_STATUS_GROB_Value_AGE', 'LP_STATUS_FEIN_Value_AGE',
                       'LP_FAMILIE_GROB_Value_AGE', 'LP_FAMILIE_FEIN_Value_AGE' ]:
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  '__NONE':                           0,
                                                                  'YOUNGER AGE':                      1, 
                                                                  'MIDDLE AGE':                       2, 
                                                                  'HIGHER AGE':                       3, 
                                                                  'ADVANCED AGE':                     4, 
                                                                  'RETIREMENT AGE':                   5 }).astype(int)
    
    elif p_column in [ 'LP_LEBENSPHASE_GROB_Value_FAMILY', 'LP_LEBENSPHASE_FEIN_Value_FAMILY', 
                       'LP_STATUS_GROB_Value_FAMILY', 'LP_STATUS_FEIN_Value_FAMILY',
                       'LP_FAMILIE_GROB_Value_FAMILY', 'LP_FAMILIE_FEIN_Value_FAMILY' ]:
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  'SINGLE':                           1, 
                                                                  'SINGLE - PERSON':                  2,
                                                                  'SINGLE - PARENT':                  3, 
                                                                  'COUPLE':                           4, 
                                                                  'SINGLE - COUPLE':                  5,
                                                                  'FAMILY':                           6,
                                                                  'SINGLE - FAMILY':                  7,
                                                                  'MULTIPERSON HOUSEHOLD':            8,
                                                                  'MULTIPERSON HOUSEHOLD - PERSON':   9 }).astype(int)
    
    elif p_column in [ 'LP_LEBENSPHASE_GROB_Value_CHILD', 'LP_LEBENSPHASE_FEIN_Value_CHILD', 
                       'LP_STATUS_GROB_Value_CHILD', 'LP_STATUS_FEIN_Value_CHILD',
                       'LP_FAMILIE_GROB_Value_CHILD', 'LP_FAMILIE_FEIN_Value_CHILD' ]:
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  '__NONE':                           0,
                                                                  'YOUNG':                            1, 
                                                                  'TEENAGER':                         2, 
                                                                  'CHILD OF FULL AGE':                3 }).astype(int)
    
    elif p_column == 'LP_FAMILIE_FEIN_Value_GENERATIONAL':
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  '__NONE':                           0,
                                                                  'SHARED FLAT':                      1,
                                                                  'TWO-GENERATIONAL':                 2,
                                                                  'MULTI-GENERATIONAL':               3 }).astype(int)
    
    elif p_column == 'LP_LEBENSPHASE_FEIN_Value_OTHER':
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  '__NONE':                           0,
                                                                  'WEALTHY':                          1 }).astype(int)
    
    elif p_column == 'LP_STATUS_FEIN_Value_OTHER':
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  '__NONE':                           0,
                                                                  'TYPICAL':                          1,
                                                                  'ORIENTATIONSEEKING':               2,
                                                                  'VILLAGERS':                        3,
                                                                  'ASPIRING':                         4,
                                                                  'WORKERS':                          5,
                                                                  'MINIMALISTIC':                     6,
                                                                  'NEW':                              7,
                                                                  'TITLE HOLDER':                     8 }).astype(int)

    elif p_column == 'PRAEGENDE_JUGENDJAHRE_Value_0':
        p_data[p_column] = p_data[p_column].replace({ 40:   1, 
                                                      50:   2, 
                                                      60:   3, 
                                                      70:   4, 
                                                      80:   5, 
                                                      90:   6 }).astype(int)
    
    # Column 'PRAEGENDE_JUGENDJAHRE_Value_2' contains only 2 values, so we will re-encode it to 0 and 1.
    elif p_column == 'PRAEGENDE_JUGENDJAHRE_Value_2':
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  'Mainstream':                       1, 
                                                                  'Avantgarde':                       2 }).astype(np.float16)
    
    # Column 'OST_WEST_KZ' contains only 2 values, so we will re-encode it to 0 and 1.
    elif p_column in [ 'OST_WEST_KZ', 'PRAEGENDE_JUGENDJAHRE_Value_3']:
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  'O':                                1, 
                                                                  'W':                                2, 
                                                                  'O+W':                              3 }).astype(int)
    
    elif p_column == 'CAMEO_DEUINTL_2015_Value_0':
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  'PRE-FAMILY COUPLES - SINGLES':     1, 
                                                                  'YOUNG COUPLES - CHILDREN':         2,
                                                                  'FAMILIES - SCHOOL AGE CHILDREN':   3, 
                                                                  'OLDER FAMILIES - MATURE COUPLES':  4, 
                                                                  'ELDERS IN RETIREMENT':             5 }).astype(int)
    
    elif p_column == 'CAMEO_DEUINTL_2015_Value_1':
        p_data[p_column] = p_data[p_column].astype(str).replace({ 'nan':                             -2, 
                                                                  '-1.0':                            -1,
                                                                  'POORER HOUSEHOLDS':                1,
                                                                  'LESS AFFLUENT HOUSEHOLDS':         2, 
                                                                  'COMFORTABLE HOUSEHOLDS':           3, 
                                                                  'PROSPEROUS HOUSEHOLDS':            4, 
                                                                  'WEALTHY HOUSEHOLDS':               5 }).astype(int)
                    
    else:
        raise ValueError('Unknown column name.')
    return


#------------------------------------------------------------------------------------------------------------------------------------
def utl_cleanDataFrame( p_label, p_data, p_featuresDef, p_categoricalColumns, p_categoricalToProcess, 
                        p_dummyEncode, p_highlyCorrelated, p_pcaMissingUnknown, p_colMissing, p_colUnknownVals,
                        p_ignoreScale = [] ):
    '''
    INPUT:
        p_data           - (pandas dataframe) the dataframe to be cleaned
        p_featuresDef    - (pandas dataframe) containing the definition for the values
        p_colMissing     - the list of columns for which we should calculate the flag if they are missing or not (must be the 
                           same list as the one used for training the PCA)
        p_pcaMissing     - trained PCA to be used to calculate the reduced missing values profile
        p_ShowTop        - number of weights to be displayed
    OUTPUT:
        Returns the trained PCA.
    '''
    
    #------------------------------------------------------------------------------------------------------------------------
    def replaceDummy(p_data, p_column, p_dummyEncode):
        if p_column not in p_data.columns: 
            for column in p_dummyEncode[p_column]:
                p_data[column] = 0
            return p_data

        p_data[p_column] = p_data[p_column].str.replace('/', '-').str.replace(' ', '_')
        p_data = pd.get_dummies(p_data, columns = [p_column])

        v_cols = [item for item in p_data.columns if (p_column in item and item not in p_dummyEncode[p_column])]
        if len(v_cols) > 0:
            p_data.drop(v_cols, axis = 1, inplace = True)

        for column in [item for item in p_dummyEncode[p_column] if item not in p_data.columns]:
            p_data[column] = 0

        return p_data
    
    #------------------------------------------------------------------------------------------------------------------------
    printHeader(f'START Clean dataframe {formatLabel(p_label)}.')
    
    
    #------------------------------------------------------------------------------------------------------------------------
    # Clean dataframe based on definition
    printmd(f'Clean dataframe {formatLabel(p_label)} based on definition.')
    v_data = utl_cleanOnDefinition( p_label       = p_label,
                                    p_data        = p_data, 
                                    p_featuresDef = p_featuresDef )
    
    
    #------------------------------------------------------------------------------------------------------------------------
    # Re-encode object columns
    for column in ['EINGEFUEGT_AM']:
        printmd(f'Re-encode column {formatLabel(column)}.')
        utl_processColumns(v_data, column)
        
        
    #------------------------------------------------------------------------------------------------------------------------
    # Re-encode columns having year format
    for column in [ 'GEBURTSJAHR', 'MIN_GEBAEUDEJAHR', 'EINGEZOGENAM_HH_JAHR', 'EINGEFUEGT_AM_Year' ]:  
        printmd(f'Re-encode column having year format {formatLabel(column)}.')
        v_values = v_data[column].value_counts(dropna = True).index.tolist()
        try:
            v_values.remove(0)
        except: None
        v_min = min(v_values) - 1
        v_idx = v_data[v_data[column].fillna(0) != 0].index
        v_data.loc[v_idx, column] = v_data.loc[v_idx, column].apply(lambda x: (int(x) - v_min) / 10 )
        v_data[column] = v_data[column].fillna(-1)
        
        
    #------------------------------------------------------------------------------------------------------------------------
    # Transform the type for object columns that can be encoded to numeric
    for column in v_data.select_dtypes(include=['object']).columns:
        try:
            v_data[column] = v_data[column].astype(np.float16)
        except: None
    
    
    #------------------------------------------------------------------------------------------------------------------------
    # Calculate the reduced dimension flags for MISSING and UNKNOWN values
    printmd(f'Calculate the reduced dimension flags for {formatLabel("MISSING")} and {formatLabel("UNKNOWN")} values.')
    v_missing = v_data[['LNR']].copy()
    for column in v_data.columns:
        v_missing[column] = v_data[column].isnull().astype(int)    
    v_missing.drop('LNR', axis = 1, inplace = True)
    v_missing.columns = [f'_isMissing_{item}' for item in v_missing.columns]   
    v_missing = v_missing[p_colMissing]
    
    v_unknown = v_data[['LNR']].copy()
    for column in p_colUnknownVals.keys():
        v_value = p_colUnknownVals[column]
        v_unknown[column] = v_data[column].fillna(-999).apply(lambda x: 1 if x == v_value else 0 ) 
    v_unknown.drop('LNR', axis = 1, inplace = True)
    v_unknown.columns = [f'_isUnknown_{item}' for item in v_unknown.columns]
    
    v_missingUnknown = pd.concat([v_missing, v_unknown], axis = 1)
    v_missingUnknown = pd.DataFrame(p_pcaMissingUnknown.transform(v_missingUnknown))    
    v_missingUnknown.columns = [f'_isMissingUnknown_{item + 1}' for item in v_missingUnknown.columns]
         
        
    #------------------------------------------------------------------------------------------------------------------------
    # Concatenate the calculated missing / unknonw values    
    printmd(f'Concatenate the calculated missing / unknonw with reduced dimension.')
    v_data = pd.concat([v_data, v_missingUnknown], axis = 1)
    
    
    #------------------------------------------------------------------------------------------------------------------------
    # Re-encode some categorical columns
    printmd('Re-encode some categorical columns.')    
    for column in [ 'LP_FAMILIE_FEIN_Value_CHILD', 
                    'LP_FAMILIE_FEIN_Value_GENERATIONAL', 
                    'LP_LEBENSPHASE_FEIN_Value_AGE',
                    'LP_LEBENSPHASE_FEIN_Value_OTHER',
                    'LP_LEBENSPHASE_GROB_Value_AGE',
                    'LP_LEBENSPHASE_GROB_Value_INCOME',
                    'LP_STATUS_FEIN_Value_OTHER' ]:
        v_data[column] = v_data[column].replace({'__NONE': np.NaN})
    
    for column in p_categoricalToProcess:
        utl_processColumns(v_data, column)
        
    #------------------------------------------------------------------------------------------------------------------------
    # Dummy encode the object columns not re-encoded above.
    printmd('Dummy encode the object columns not re-encoded above.')
    for column in list(p_dummyEncode.keys()):
        v_data = replaceDummy( p_data        = v_data, 
                               p_column      = column, 
                               p_dummyEncode = p_dummyEncode )
    
    
    #------------------------------------------------------------------------------------------------------------------------
    # Fill in categorical columns with -2
    printmd('Fill in categorical columns with -2.')
    for column in p_categoricalColumns:
        v_data[column] = v_data[column].fillna(-2).astype(int)
    
    
    #------------------------------------------------------------------------------------------------------------------------
    # Drop features identified as highly correlated 
    printmd('Drop features identified as highly correlated.')
    v_data.drop(p_highlyCorrelated, axis = 1, inplace = True)
    
    
    #------------------------------------------------------------------------------------------------------------------------
    # Calculate columns to be imputed later
    printmd('Calculate columns to be imputed later.')
    v_missing = pd.DataFrame({'Null Value': 0}, index = ['LNR'])
    for column in v_data.columns:
        v_missing.loc[column, 'Null Value'] = v_data[column].isnull().sum()
    v_missing = v_missing[v_missing['Null Value'] > 0]
    v_missing = v_missing.index.tolist()
    
    
    #------------------------------------------------------------------------------------------------------------------------
    # Calculate columns to be scaled later
    printmd('Calculate columns to be scaled later.')
    v_dummyEncode = []
    for value in p_dummyEncode.values():
        v_dummyEncode.extend(value)
    v_scaleColumns = sorted([ item for item in v_data.drop(p_ignoreScale, axis = 1).columns 
                                  if ( item not in p_categoricalColumns
                                       and 'isMissingUnknown' not in item
                                       and 'isMissingUnknown' not in item
                                       and item not in v_dummyEncode )  ])
    
    #------------------------------------------------------------------------------------------------------------------------
    printHeader(f'END Clean dataframe {formatLabel(p_label)}.')

    return v_data, v_missing, v_scaleColumns