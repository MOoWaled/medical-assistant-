import pandas as pd
from pymongo import MongoClient
import os

# call MedicalTextPreprocessor class
# for clean, tokenize, normalize data
from preprocessor import MedicalTextPreprocessor

def prepare_and_export_dataset(db_name="nhs_conditions_db", collection_name="conditions", output_folder="dataset"):
    print("connect to MongoDB...")
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    collection = db[collection_name]
    
    # 1.fetching data from MongoDB
    # get condition and symptoms and other fields from MongoDB
    cursor = collection.find({}, {
        "_id": 0, 
        "Condition_name": 1, 
        "Symptoms": 1, 
        "Causes": 1, 
        "Warnings": 1, 
        "Recommendations": 1
    })
    
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        print("can't find any data in MongoDB.")
        return
        
    print(f"Done fetching data from MongoDB {len(df)}")
    
    # 2. use Preprocessing (Clean, tokenize, normalize)
    print("cleaning ...")
    preprocessor = MedicalTextPreprocessor()
    
    # create new column for cleaned symptoms using Lemmatization
    df['cleaned_Symptoms'] = df['Symptoms'].apply(preprocessor.clean_text)
    df['cleaned_Causes'] = df['Causes'].apply(preprocessor.clean_text)
    df['cleaned_Warnings'] = df['Warnings'].apply(preprocessor.clean_text)
    df['cleaned_Recommendations'] = df['Recommendations'].apply(preprocessor.clean_text)
    
    # save data as Usable Dataset
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "usable_dataset.csv")
    
    # save the dataset for Baseline Model (TF-IDF + Logistic Regression)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Done,dataset is ready to train at : {csv_path}")

if __name__ == "__main__":
    prepare_and_export_dataset()