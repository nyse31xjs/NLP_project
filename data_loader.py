import pandas as pd
import json

# restricted paths to the json files
event_file = "data/Nodes/Event.json"  
hashtag_file = "data/Nodes/Hashtag.json"  
postcategory_file = "data/Nodes/PostCategory.json"  
tweet_file = "data/Nodes/Tweet.json"
user_file = "data/Nodes/User.json"


def load_data(root_path: str, insert_csv: bool = False, insert_pkl: bool = False):
    '''
    Load data from json files and save them in csv or pkl format.
    
    args:
    root_path (str): path to the root directory
    insert_csv (bool): if True, save the data in csv format
    insert_pkl (bool): if True, save the data in pkl format
    '''
    
    dfs = {}

    for file in [event_file, hashtag_file, postcategory_file, tweet_file, user_file]:
        
        file = root_path + file
        file_key = file.split("/")[-1].replace(".json", "")  

        with open(file, "r") as f:
            data = [json.loads(line) for line in f]  

        df = pd.DataFrame([item["n"]["properties"] for item in data])

        dfs[file_key] = df

    if insert_csv:
        for key, df in dfs.items():
            df.to_csv(f"{root_path}data/Nodes/{key}.csv", index=False)
        
    if insert_pkl:
        for key, df in dfs.items():
            df.to_pickle(f"{root_path}data/Nodes/{key}.pkl")
       
def load_pkl(root_path: str):
    '''
    Load data from pkl files.
    
    args:
    root_path (str): path to the root directory
    
    returns:
    dfs (dict): dictionary with the pandas dataframes
    '''
    
    dfs = {}
    for file in [event_file, hashtag_file, postcategory_file, tweet_file, user_file]:
        file = root_path + file
        file_key = file.split("/")[-1].replace(".json", "")  
        df = pd.read_pickle(f"{root_path}data/Nodes/{file_key}.pkl")
        dfs[file_key] = df
    
    return dfs


def main():
    load_data(root_path = "/Users/hugorameil/Desktop/Code/GitHub/NLP_project/", insert_csv=True, insert_pkl=True)
    dfs = load_pkl(root_path = "/Users/hugorameil/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Code/GitHub/NLP_project/")

    
if __name__ == "__main__":
    main()