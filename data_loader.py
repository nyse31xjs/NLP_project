import pandas as pd
import json

# paths to the json files
event_file = "data/Nodes/Event.json"  
hashtag_file = "data/Nodes/Hashtag.json"  
postcategory_file = "data/Nodes/PostCategory.json"  
tweet_file = "data/Nodes/Tweet.json"
user_file = "data/Nodes/User.json"


def load_data(root_path: str, insert: bool = False):
    dfs = {}

    for file in [event_file, hashtag_file, postcategory_file, tweet_file, user_file]:
        
        file = root_path + file
        file_key = file.split("/")[-1].replace(".json", "")  

        with open(file, "r") as f:
            data = [json.loads(line) for line in f]  

        df = pd.DataFrame([item["n"]["properties"] for item in data])

        dfs[file_key] = df

    if insert:
        for key, df in dfs.items():
            df.to_csv(f"{root_path}data/Nodes/{key}.csv", index=False)
        
if __name__ == "__main__":
    load_data(root_path = "/Users/hugorameil/Desktop/Code/GitHub/NLP_project/", insert=True)