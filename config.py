from pathlib import Path
def get_config():
    return {
        "batch_size":8,
        "num_epochs":20,
        "lr":10**-3,
        "seq_len":470,
        "d_model":512,
        "datasource": 'Helsinki-NLP/opus-100',
        "lang_src":"en",
        "lang_tgt":"hi",
        "model_folder":"weights",
        "model_basename": "tmodel_",
        "preload":None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"         
    }

def get_weights_file_path(config,epoch):
    model_folder=config['model_folder']
    model_basename=config['model_basename']
    model_filename=f'{model_basename}{epoch}.pt'
    return str(Path('.')/model_folder/model_filename)

from pathlib import Path

# def get_config():
#     return {
#         "batch_size": 8,
#         "num_epochs": 20,
#         "lr": 10**-3,
#         "seq_len": 470,
#         "d_model": 512,
#         "datasource": 'path/to/your/translations.csv',  # Update this path to your CSV file
#         "lang_src": "original",  # Source language column name in your CSV
#         "lang_tgt": "translation",  # Target language column name in your CSV
#         "model_folder": "weights",
#         "model_basename": "tmodel_",
#         "preload": None,
#         "tokenizer_file": "tokenizer_{0}.json",
#         "experiment_name": "runs/tmodel"
#     }

# def get_weights_file_path(config, epoch):
#     model_folder = config['model_folder']  # Corrected key from 'nodel_folder' to 'model_folder'
#     model_basename = config['model_basename']
#     model_filename = f'{model_basename}{epoch}.pt'
#     return str(Path('.') / model_folder / model_filename)
