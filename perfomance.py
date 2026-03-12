#This script is for compaing accuracy from all available models
import json
import os




def get_all_meta_data() -> list:
    '''This functions returns a list containing all available meta data files'''

    meta_data = []

    dir = os.listdir()

    for file in dir:
        if file.endswith("json"):
            meta_data.append(file)

    print(f"Total File: {len(dir)}")
    print(f"Json files: {len(meta_data)}")
    return meta_data




def extract_percentage(meta_data_files:list) -> list:
    '''This function returns a list of dictionaries of version:percentage key pare values'''

    meta_data_records = []
    meta_data = {}
    # print("All meta files:")
    # print(meta_data_files)
    for file in meta_data_files:
        with open(file, 'r') as dfile:
            records = json.load(dfile)
            meta_data['version'] = records['version']
            meta_data['accuracy'] = records['metrics']['accuracy']
            meta_data['precision'] = records['metrics']['precision_toxic']
            meta_data['file'] = file

            meta_data_records.append(meta_data)
            meta_data = {}


    return meta_data_records



def transform(meta_data:list):

    results = []
    meta_results = {}
    for data in meta_data:

        perc = data['accuracy'] * 100 /1
        prec = data['precision'] * 100 /1
        normalize = float(f"{perc:.2f}")
        prec_nomalize = float(f"{prec:.2f}")
        meta_results['version'] = data['version']
        meta_results['file'] = data['file']
        meta_results['percentage'] = normalize
        meta_results['precision'] = prec_nomalize

        results.append(meta_results)
        meta_results = {}

    return results


def calculate(data):

    return {
        'min': min(data),
        'max': max(data),
        'avg': float(f"{sum(data) /len(data):.2f}")
    }

files = get_all_meta_data()

meta = extract_percentage(files)

results = transform(meta)

for result in results:
    print(f"Version: {result['version']} | Accuracy: {result['percentage']}% | Precision: {result['precision']}% | File: {result['file']}")
    
percentage = [d['percentage'] for d in results]
precision = [d['precision'] for d in results]

percentage_c = calculate(percentage)
precision_c = calculate(precision)

print("Percentage stats:")
print("-"*6)
print(
    "Min:",percentage_c['min'],'\n'
    "Max:", percentage_c['min'],'\n'
    'Avg:', percentage_c['avg'],'\n'
)
print()
print("Precision stats:")
print("-"*6)
print(
    "Min:",precision_c['min'],'\n'
    "Max:", precision_c['min'],'\n'
    'Avg:', precision_c['avg'], '\n'
)