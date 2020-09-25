# Based on https://pythoninoffice.com/use-python-to-combine-multiple-excel-files/

import os
import pandas as pd
import time

# Get files
cwd = os.getcwd() + '/output'
files = os.listdir(cwd)
print(files)

# Read and combine 'summary' sheets in Excel files
master = pd.DataFrame()

# Loop through files
for file in files:
    if file.endswith('.xlsx'):
        print('Starting file {}'.format(file))
        file_path = cwd + '/' + file
        excel_file = pd.ExcelFile(file_path)
        sheets = excel_file.sheet_names
        # Loop through summary sheets in file
        for sheet in sheets:
            if sheet.startswith('summary_'):
                print('Worksheet: {}'.format(sheet))
                df = excel_file.parse(sheet_name=sheet)
                df['Statistic'] = sheet[8:]
                master = master.append(df)

# Fill down blank cells
print('Filling blank cells.')
for j in range(master.shape[1]-2):
    for i in range(1,master.shape[0]):
        if pd.isnull(master.iloc[i,j]):
            master.iloc[i,j] = master.iloc[i-1,j]

# Save combined file
date = time.strftime("%Y-%m-%d")
master.to_excel('combined_outputs_'+date+'.xlsx')
print('Complete.')