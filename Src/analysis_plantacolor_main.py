# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:51:49 2024

@author: JBI
"""

"""
###############################################
##Title             : analysis_plantacolor_main.py
##Description       : Main script for TerraLight project
##Author            : John Bigeon   @ Github
##Date              : 20240323
##Version           : Test with
##Usage             : Python, OpenCV, pandas, matplotlib
                    pip install imutils
##Script_version    : 0.0.1 (not_release)
##Output            :
##Notes             :
###############################################
"""
###############################################
### Package
###############################################
import numpy as np
import time
import os
import re
import cv2
import logging
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy import interpolate
from matplotlib.gridspec import GridSpec
import Lib_local

###############################################
### Function
###############################################
def versioning():
    ver = f"#Plantacolor: {Lib_local.__version__}"
    return ver


# Define helper functions to modularize key operations
def load_and_preprocess_data(data_log_list, data_typ_list):
    df_all = pd.DataFrame()
    for data_log, data_typ in zip(data_log_list, data_typ_list):
        df = pd.read_csv(data_log, delimiter='; ', engine='python')
        df = df.drop(df.columns[0], axis=1) # Drop the timestamp of the logging
        df.insert(0, 'Type', data_typ)
        df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all


def get_unique_values(df_all):
    wvl_rng = df_all['Wvl'].unique().tolist()
    qval_rng = df_all['Qval'].unique().tolist()
    typ_rng = df_all['Type'].unique().tolist()
    return wvl_rng, qval_rng, typ_rng


def process_images(df_all, debug=False, display=False):
    img_res_wvl = []
    typ_list = []
    wvl_list = []
    img_stack = []
    bck_stack = []

    # Sort df_all by 'Wvl' to process in wavelength order
    df_all = df_all.sort_values(by='Wvl', kind='mergesort')
        
    qval_iter = max(df_all['Qval']+1)
    
    # Loop through df_all in increments of qval_iter
    for i in range(0, len(df_all), qval_iter):
        df_batch = df_all.iloc[i:i+qval_iter]  # Take a batch of qval_iter rows
        if len(df_batch) < qval_iter:
            break  # End loop if fewer than expected

        # Check for background type within the current batch
        for _, row in df_batch.iterrows():
            
            typ_tmp = df_batch.iloc[0]['Type']
            wvl_tmp = df_batch.iloc[0]['Wvl']
            
            ### Communications
            print(5*'#')
            print(f'Image processing for type={typ_tmp}, wvl={wvl_tmp}')
            
            img = cv2.imread(row['Filename'], cv2.IMREAD_GRAYSCALE)
            if 'background' in row['Type']:
                bck_stack.append(img)
                # print(f"{row['Filename']}: considered as background")
            else:
                img_stack.append(img)
                # print(f"{row['Filename']}: considered as image")

            if debug:
                plt.imshow(img, cmap='gray')
                plt.title(f'Img raw: {row["Type"]}, {row["Wvl"]}nm')
                plt.colorbar()
                plt.show(block=False)
                plt.close()
        
            # Apply background processing if a background image was found
            if len(img_stack) == qval_iter:
                bck_sum_qval = np.sum(bck_stack, axis=0).astype('float64')
                img_sum_qval = np.sum(img_stack, axis=0).astype('float64')
    
                img_res = img_sum_qval / bck_sum_qval

                # Get min/max of both images
                vmin = min(np.min(bck_sum_qval), np.min(img_sum_qval))
                vmax = max(np.max(bck_sum_qval), np.max(img_sum_qval))
                
                if debug:
                    fig = plt.figure(figsize=(10, 5))
                    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
                    ax1 = fig.add_subplot(gs[0, 0])
                    im1 = ax1.imshow(bck_sum_qval, vmin=vmin, vmax=vmax, cmap='viridis')
                    
                    ax2 = fig.add_subplot(gs[0, 1])
                    im2 = ax2.imshow(img_sum_qval, vmin=vmin, vmax=vmax, cmap='viridis')
                    
                    cbar = fig.colorbar(im1, cax=fig.add_subplot(gs[0, 2]))
                    cbar.set_label('Color Scale')
                    plt.title(f'Bck vs Img for type={typ_tmp}, wvl={wvl_tmp}')
                    plt.show(block=False)
                    plt.close()

                # Append
                typ_list.append(typ_tmp)
                wvl_list.append(wvl_tmp)
                img_res_wvl.append(img_res)

                # When the processing is done: re-init the stacks
                img_stack = []
                bck_stack = []
    
                if debug:
                    plt.imshow(img_res, cmap='gray')
                    plt.title(f'Processed Image: {df_batch.iloc[0]["Type"]}, {df_batch.iloc[0]["Wvl"]}nm')
                    plt.colorbar()
                    plt.show(block=False)
                    plt.close()
                    
                if display:
                    plt.imshow(img_res, cmap='gray')
                    plt.title(f'Img res: {df_batch.iloc[0]["Type"]}, {df_batch.iloc[0]["Wvl"]}nm')
                    plt.colorbar()
                    mask = np.ma.masked_where(img_res <= 1, img_res)
                    plt.imshow(mask, cmap='Reds', alpha=1.0)
                    plt.show(block=False)
                    plt.close()
    
    # Export to the dataframe
    res_dict = {'Type': typ_list, 'Wvl (nm)': wvl_list, 'Img res': img_res_wvl} 
    res_df = pd.DataFrame(res_dict)
    
    img_res_wvl = np.array(img_res_wvl)    
    
    return img_res_wvl, res_df


def find_contours(img):
    # Rescale image
    img_res_scaled = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    threshold_value = 0.2 * 255  # 20% of 255 after scaling
    
    #_, img_th = cv2.threshold(img, 0.1*np.max(img), 0.9*np.max(img), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(img_res_scaled, threshold_value, 255, cv2.THRESH_BINARY)

    thresh_uint8 = thresh.astype(np.uint8)
    
    contours, _ = cv2.findContours(thresh_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on the area threshold
    img_area = img.shape[0] * img.shape[1]
    area_threshold = 0.9 * img_area  # 10% of the total image area
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < area_threshold]
    
    if contours:
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (np.nan, np.nan)

        cont_area = cv2.contourArea(contour)
        
        # Create a mask and find contour pixel intensities
        mask = np.zeros_like(img_res_scaled)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        cont_pixels = img[mask == 255]
        cont_mean = np.mean(cont_pixels) if cont_pixels.size > 0 else np.nan
        
        img_contours = cv2.cvtColor(img_res_scaled, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, [contour], -1, (0, 255, 0), 1)
        label = f"Perimeter: {cv2.arcLength(contour, True)}"
        cv2.putText(img_contours, label, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return img_contours, cont_area, cont_mean, cx, cy     
        

def analyze_contours(res_df, debug=False, display=False):
    cont_area_list, cx_list, cy_list = [], [], []
    wvl_list = []
    typ_list = []
    cont_area_list = []
    cont_mean_list = []
    
    os.makedirs('Results', exist_ok=True)
    os.makedirs('Debug', exist_ok=True)
    
    for i in range(0, len(res_df)):
        typ_tmp = res_df.iloc[i]['Type']
        wvl_tmp = res_df.iloc[i]['Wvl (nm)']
        img_tmp = res_df.iloc[i]['Img res']

        img_contours, cont_area, cont_mean, cx, cy = find_contours(img_tmp)
        # Display contours for debugging
        if display:
            fig_name = f"Debug//analyse_contour_typ_{typ_tmp}_wvl_{wvl_tmp}.png"

            plt.imshow(img_contours)
            plt.title(f"Contour at {wvl_tmp}, mean {cont_mean}")
            plt.axis('off')
            plt.colorbar()
            plt.savefig(fig_name, dpi=500, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=True, bbox_inches='tight', pad_inches=0.1, metadata=None)
            plt.show(block=False)
            plt.close()

        # Store and Update response dictionary
        typ_list.append(typ_tmp)
        wvl_list.append(wvl_tmp)
        cont_area_list.append(cont_area)
        cont_mean_list.append(cont_mean)
        cx_list.append(cx)
        cy_list.append(cy)
        print(f"Type={typ_tmp}, Wavelength={wvl_tmp},cx={cx}, cy={cy}, contour area={cont_area}, contour mean={cont_mean}")

    # Export to the dataframe
    res_df['Cx (pixels)'] = cx_list
    res_df['Cy (pixels)'] = cy_list
    res_df['Contour area'] = cont_area_list
    res_df['Contour resp. (mean)'] = cont_mean_list
    
    return cx_list, cy_list, cont_area_list, cont_mean_list, res_df


def plot_slice(res_df, params, datum):
    os.makedirs('Results', exist_ok=True)
    params_exp = params.replace(' ', '_') # Cosmetic: remove space ...
    params_exp = params.replace('.', '_') # Cosmetic: remove space ...
    fig_name = f'Results//response_{params_exp}_slice_{datum:%Y-%m-%d_%H%M%S}.png'
    
    ### The dataframe is easy to use but we will be limited for lmfit
    # sns.lineplot(data=res_df, x="Wvl (nm)", y=params, hue="Type", marker="o")
    
    for typ in res_df['Type'].unique():
        subset = res_df[res_df['Type'] == typ]
        wvl = subset['Wvl (nm)']
        resp = subset[params]
        
        # Fit it
        xnew = np.arange(min(wvl), max(wvl), 0.1)
        f = interpolate.interp1d(wvl, resp, kind='cubic')

        # Plot it
        plt.scatter(wvl, resp, marker='o', s=7, label=typ)
        plt.plot(xnew, f(xnew))
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(f'{params}')
    plt.legend()
    plt.savefig(fig_name, dpi=500, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=True, bbox_inches='tight', pad_inches=0.1, metadata=None)
    plt.show(block=False)
    plt.close()
 
    
def plot_matrix(res_df, params, datum):
    os.makedirs('Results', exist_ok=True)
    params_exp = params.replace(' ', '_') # Cosmetic: remove space ...
    params_exp = params.replace('.', '_') # Cosmetic: remove space ...

    fig_name = f'Results//response_{params_exp}_matrix_{datum:%Y-%m-%d_%H%M%S}.png'

    # Extract what we need via pivot strategy
    heatmap_data = res_df.pivot(index='Type', columns='Wvl (nm)', values=params)
    
    # Show me what you got
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
    
    plt.xticks(ticks=np.arange(len(heatmap_data.columns))[::2], labels=heatmap_data.columns[::2], rotation=45)
    plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=heatmap_data.index)

    plt.colorbar(label='Reflectance')
    plt.title(f'{params} Matrix')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Type Index')
    plt.savefig(fig_name, dpi=500, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=True, bbox_inches='tight', pad_inches=0.1, metadata=None)
    plt.show(block=False)
    plt.close()  
    

def load_logbook(path):
    df = pd.read_csv(log_book, delimiter=',', engine='python')
    
    data_log_list, data_typ_list = df['#Log_file'], df['#Type']
    
    return data_log_list, data_typ_list


def argparse_to_bool(str) -> bool:
    if str == 'True':
        return True
    else:
        return False


### Main
def main_v01(data_path, debug, display):
    datum = datetime.datetime.now()
    
    # Get params
    data_log_list, data_typ_list = load_logbook(data_path)

    # Argparse to boolean: TODO Fix that
    debug, display = argparse_to_bool(debug), argparse_to_bool(display)

    # Load and preprocess data
    df_all = load_and_preprocess_data(data_log_list, data_typ_list)

    # Process images and metadata
    sorted_img_res_wvl, response_df = process_images(df_all, debug=debug, display=display)

    # Analyze contours and calculate response dictionary
    cx_list, cy_list, cont_area_list, cont_mean_list, res_df = analyze_contours(response_df, debug=debug, display=display)

    # Show me what you got
    plot_slice(res_df, 'Contour area', datum)
    plot_matrix(res_df, 'Contour area', datum)
    plot_slice(res_df, 'Contour resp. (mean)', datum)
    plot_matrix(res_df, 'Contour resp. (mean)', datum)

###############################################
### MAIN
###############################################  
if __name__ == '__main__':
    # Parse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--log_book", type=str, default='my_logbook.csv',   help='(Logbook of the experiments. Default=my_logbook.csv')
    parser.add_argument("--debug", type=str,    default='False',            help='Display images loaded. Default=False')
    parser.add_argument("--display", type=str,  default='False',            help="Display analysis. Default=False")
    
    # Get it
    args        = parser.parse_args()
    log_book    = args.log_book
    debug       = args.debug
    display     = args.display

    # Start the main script
    main_v01(log_book, debug, display)

    print('End of the script %s for Plantacolor' %versioning())
