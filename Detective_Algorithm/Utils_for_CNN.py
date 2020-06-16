"""
 -------------------------------------------------------------------------------------------
 Created by: Veronika Ulanova
 On: 09/06/2020
 -------------------------------------------------------------------------------------------
 Python file with all the relevant functions for CNN_blocking_predctions.ipynb analysis and making plots with
 Making_Nice_Plots.ipynb. 
 
"""

import xarray
import numpy as np
import scipy
import glob
import pandas as pd
import matplotlib.pyplot as plt

def check_blocking_accuracy(pred_labels, actual_labels):
    """ Returns the statistics on predicted blocked days compared to the traning labels.
    
    Calculates the number of predicted blocked days that matches the actual blocked days (True Positives),
    the total number of actual blocked days as defined by the GTD (Positives) and the predicted blocked
    days accuracy (True Positives/Positives).
    
    Parameters: 
    pred_labels (array or list): predicted labels outputed by the algorithm (e.g. CNN)
    actual_labels (array or list): of actual labels (e.g. GTD) 
        
    Returns:
    int: number of true positives
    int: number of positives
    int: number of true positives/number of positives
    
    """
    check_blocking = pd.DataFrame(np.array(pred_labels), columns=['predicted_label'])
    check_blocking['actual_label'] = actual_labels
    check_blocking['matching'] = (check_blocking['actual_label'] == check_blocking['predicted_label'])\
                                    & (check_blocking['actual_label'] == 1)
    num_blocked = check_blocking['actual_label'].sum()
    return (check_blocking['matching'].sum(), num_blocked , check_blocking['matching'].sum()/num_blocked) 

def multiply_data(arr,multiple):
    """Formats existing labels for a different type of data.
    
    When multiple snapshots in a day of Z500* contours are used for tarining, the array of labels used for 
    training needs to be formatted. This function multiplies the labels for each day by the number of 
    snapshots taken in a day (i.e. multiplier).
    
    Parameters:
    arr (array or list): arrays of labels to be multiplied (e.g. GTD)
    multiplier (int): the multiplier (e.g. for 3hrs snapshots, 24/3 = 8)
    
    Returns:
    array: label array of shape (len(arr), multiplier)
    
    """
    combined_8timesnaps = arr.reshape(time_domain_size,1)
    combined_modif =[]
    for i in range(len(combined_8timesnaps)):
        li = list(combined_8timesnaps[i])
        combined_modif.append(li*multiple)
    return np.array(combined_modif)

def windowing_geopotential_new(data_arr, window):
    """Implementation of window method on data (see report for more details)
    
    Stacks several individual contours along y-axis (latitude) to get 1 training sample that is several days
    long (i.e. a 7-day window of data).
    
    Parameters:
    data_arr (array) - the data to be stacked up (e.g. dimensions (T,x,y))
    window (int) - the number of days you would like to stack together
    
    Returns:
    array: array of data 
    """
    even_more_new_arr =[]
    for i in range(len(data_arr)-(window-1)):
        more_new_arr =[]
        for j in range(len(data_arr[0])):
            new_arr_top = []
            added_dayz = []
            for m in range(0,window):
                added_dayz += list(data_arr[i+m][j])
            new_arr_top.append(added_dayz)
            more_new_arr+=new_arr_top
        even_more_new_arr.append(more_new_arr)
    return np.array(even_more_new_arr)

def format_labeling(label_arr, window,day_of_label, num_of_years, number_of_days_in_a_year):
    """ When windowing method is implemented the label array has to be formatted such that
    a number of labels are removed from the beginning of each summer period. That number is
    defined by the size of the window and the day in that window used for labeling.
    
    Parameters:
    label_arr (list or array): the list of labels used for training (e.g. GTD)
    window (int): the size of the window period (number of days to be stacked together)
    day_of_label (int): the day in a window sample which decided the label of a the window
    sample
    num_of_years (int): number of years the data is available for (40 for this project, data from ERA5)
    number_of_days_in_a_year (int): number of days per year that we want to investigate over (e.g. summer
    period only for this project, 92-day per period)
    
    Returns:
    array: array of formatted labels
    
    E.g. If we have an array of 3680 labels (40 92-day summer period) and we want to convert
    this to 7-day samples labeled on the middle day (i.e. day 4), this will result in 4 days
    being removed from the beggining of each summer period in label array, resulting an array 
    of 3440 7-day window labels.
    """
    split = label_arr.reshape(num_of_years, number_of_days_in_a_year)
    formated=[]
    for h in split:
        formated.append(h[window-day_of_label:number_of_days_in_a_year-(window-day_of_label)])
    return (np.array(formated).reshape(num_of_years*(number_of_days_in_a_year-2*(window-day_of_label))))
