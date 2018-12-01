import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import progressbar

#------------------------------------------------------------------------------------------------------------------------------------
def printmd(p_string):
    display(Markdown(p_string))
    return

    
#------------------------------------------------------------------------------------------------------------------------------------
def printHeader(p_header):
    print('\n********************************************************************************************')
    printmd(p_header)
    return


#------------------------------------------------------------------------------------------------------------------------------------
def formatLabel(p_label):
    return f'**<span style="color: blue">{p_label}</span>**'


#------------------------------------------------------------------------------------------------------------------------------------
def utl_toInteger(p_value):
    try:
        v_value = int(p_value)
        return v_value
    except:
        return p_value
    
    
#------------------------------------------------------------------------------------------------------------------------------------
def utl_isInteger(p_value):
    try:
        v_value = int(p_value)
        return True
    except:
        return False
    
    
#------------------------------------------------------------------------------------------------------------------------------------
def utl_reduceMemoryUsage(p_label, p_data):
    printHeader(f'Reduce memory usage for dataframe {formatLabel(p_label)}.')
    
    v_memoryUsage_start = p_data.memory_usage().sum() / 1024**2 
    print(f'Initial memory usage is {round(v_memoryUsage_start, 2)} MBs.')
    print(p_data.info())
    
    with progressbar.ProgressBar(max_value = len(p_data.columns)) as bar:
        v_count = 0
        for column in p_data.columns:
            if p_data[column].dtype != object: # Exclude object types            
                if p_data[column].isnull().sum() == 0:
                    # Calculate min / max in order to decide on the datatype to be used
                    v_min = p_data[column].min()
                    v_max = p_data[column].max()

                    # Make Integer/unsigned Integer datatypes
                    if v_min >= 0:
                        if v_max < 255:
                            p_data[column] = p_data[column].astype(np.uint8)
                        elif v_max < 65535:
                            p_data[column] = p_data[column].astype(np.uint16)
                        elif v_max < 4294967295:
                            p_data[column] = p_data[column].astype(np.uint32)
                        else:
                            p_data[column] = p_data[column].astype(np.uint64)
                    else:
                        if v_min > np.iinfo(np.int8).min and v_max < np.iinfo(np.int8).max:
                            p_data[column] = p_data[column].astype(np.int8)
                        elif v_min > np.iinfo(np.int16).min and v_max < np.iinfo(np.int16).max:
                            p_data[column] = p_data[column].astype(np.int16)
                        elif v_min > np.iinfo(np.int32).min and v_max < np.iinfo(np.int32).max:
                            p_data[column] = p_data[column].astype(np.int32)
                        elif v_min > np.iinfo(np.int64).min and v_max < np.iinfo(np.int64).max:
                            p_data[column] = p_data[column].astype(np.int64)    
                else:
                    try:
                        p_data[column] = p_data[column].astype(np.float16)
                    except:
                        p_data[column] = p_data[column].astype(np.float32)
                        
            v_count += 1
            bar.update(v_count)
                                        
    v_memoryUsage_end = p_data.memory_usage().sum() / 1024**2 
    print(f'\nFinal memory usage is {round(v_memoryUsage_end, 2)} MBs.')
    print(f'This is {round(100 * v_memoryUsage_end / v_memoryUsage_start, 2)}% of the initial size.\n')
    print(p_data.info())    
    return