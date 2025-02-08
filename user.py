import pandas as pd
from collections import Counter
import os

class UserManager:
    def __init__(self, file_path="user_data.csv"):
        self.file_path = file_path
        if os.path.exists(file_path):
            try:
                # Read all columns but only use user_id and preference
                self.data = pd.read_csv(file_path)[["user_id", "preference"]]
                print(f"CSV file loaded successfully with {len(self.data)} records.")
            except Exception as e:
                print(f"Error reading the CSV file: {e}")
                self.data = pd.DataFrame(columns=["user_id", "preference"])
        else:
            print("CSV file does not exist, creating a new one.")
            self.data = pd.DataFrame(columns=["user_id", "preference"])
        self.user_keyword_counts = self._process_user_keywords()

    
    def _process_user_keywords(self, top_n=3):
        user_keywords = {}
        # Group by user_id and aggregate all preferences
        for user_id in self.data['user_id'].unique():
            # Get all preferences for this user
            user_prefs = self.data[self.data['user_id'] == user_id]['preference'].values
            # Combine all preferences and split into individual keywords
            all_keywords = []
            for pref in user_prefs:
                keywords = [k.strip() for k in pref.split(',')]
                all_keywords.extend(keywords)
            # Count keywords and get top N
            keyword_counts = Counter(all_keywords)
            top_keywords = keyword_counts.most_common(top_n)
            user_keywords[user_id] = [keyword for keyword, _ in top_keywords]
        return user_keywords

    
    def get_user_ids(self):
        return self.data['user_id'].unique()
    
    def get_top_keywords(self, user_id):
        return self.user_keyword_counts.get(user_id, [])
    
    def validate_user(self, user_id):
        try:
            user_id = int(user_id)
            return user_id in self.data['user_id'].values
        except ValueError:
            return False

    def create_user(self, user_id, preferences):
        try:
            new_user = pd.DataFrame({
                "user_id": [int(user_id)],
                "preference": [preferences]
            })
            
            # Make sure the columns are in the same order as the original CSV
            new_user = new_user[["user_id", "preference"]]
            
            # Append new user data to the CSV file
            self.data = pd.concat([self.data, new_user], ignore_index=True)
            self.data.to_csv(self.file_path, index=False)

            # Reload data to reflect changes
            self.data = pd.read_csv(self.file_path)
            
            # Update the keyword counts
            self.user_keyword_counts = self._process_user_keywords()
            print("added to file")
            return True
            
        except Exception as e:
            print(f"Error creating user: {e}")
            return False

