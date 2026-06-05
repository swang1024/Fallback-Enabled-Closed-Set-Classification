import pandas as pd
import re

# def process_schools(df):
#     """
#     Process the schools dataframe according to the requirements.
    
#     Parameters:
#     df: DataFrame with columns ['school_id', 'state_code', 'subjects']
    
#     Returns:
#     DataFrame with state-wise count of schools offering each subject
#     """
    
#     # Step 1: Drop schools offering fewer than 3 subjects
#     # Count subjects by splitting the space-separated string
#     df['subject_count'] = df['subjects'].str.split().str.len()
#     df_filtered = df[df['subject_count'] >= 3].copy()
#     df_filtered = df_filtered.drop('subject_count', axis=1)
    
#     # Step 2: Clean state_code to only contain alphanumeric characters
#     df_filtered['state_code'] = df_filtered['state_code'].apply(
#         lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x))
#     )
    
#     # Step 3: For each state, count schools offering English, Maths, Physics, Chemistry
#     # Create boolean columns for each subject
#     df_filtered['offers_english'] = df_filtered['subjects'].str.contains('english', case=False, na=False)
#     df_filtered['offers_maths'] = df_filtered['subjects'].str.contains('maths', case=False, na=False)
#     df_filtered['offers_physics'] = df_filtered['subjects'].str.contains('physics', case=False, na=False)
#     df_filtered['offers_chemistry'] = df_filtered['subjects'].str.contains('chemistry', case=False, na=False)
    
#     # Group by state and count schools offering each subject
#     result = df_filtered.groupby('state_code').agg({
#         'offers_english': 'sum',
#         'offers_maths': 'sum',
#         'offers_physics': 'sum',
#         'offers_chemistry': 'sum'
#     }).reset_index()
    
#     # Rename columns for clarity
#     result.columns = ['state_code', 'English', 'Maths', 'Physics', 'Chemistry']
    
#     # Convert counts to integers
#     result[['English', 'Maths', 'Physics', 'Chemistry']] = result[['English', 'Maths', 'Physics', 'Chemistry']].astype(int)
    
#     return result


# # Example usage:
# # Create sample data
# sample_data = {
#     'school_id': [1, 2, 3, 4, 5, 6, 7, 8],
#     'state_code': ['CA-01', 'CA@02', 'TX#01', 'TX-02', 'CA-03', 'NY!01', 'TX-03', 'CA-04'],
#     'subjects': [
#         'english maths physics chemistry',
#         'english maths',  # Will be dropped (< 3 subjects)
#         'maths physics chemistry biology',
#         'english physics chemistry',
#         'english maths physics',
#         'chemistry biology history',
#         'english maths chemistry physics',
#         'history geography'  # Will be dropped (< 3 subjects)
#     ]
# }

# df = pd.DataFrame(sample_data)

# print("Original DataFrame:")
# print(df)
# print("\n" + "="*60 + "\n")

# result = process_schools(df)

# print("Result:")
# print(result)

# ```

# **Explanation of the solution:**

# 1. **Drop schools with fewer than 3 subjects**: 
#    - Split the `subjects` string by spaces and count the number of subjects
#    - Filter to keep only schools with 3 or more subjects

# 2. **Clean state_code**:
#    - Use regex `re.sub(r'[^a-zA-Z0-9]', '', str(x))` to remove all non-alphanumeric characters
#    - This keeps only letters and numbers

# 3. **Count schools by state and subject**:
#    - Create boolean columns checking if each subject exists in the `subjects` string
#    - Group by `state_code` and sum the boolean columns (True=1, False=0)
#    - This gives the count of schools offering each subject per state

# **Output example:**
# ```
#   state_code  English  Maths  Physics  Chemistry
# 0       CA01        2      2        2          2
# 1       CA03        1      1        1          0
# 2       NY01        0      0        0          1
# 3       TX01        1      1        1          1
# 4       TX02        1      0        1          1
# 5       TX03        1      1        1          1


# Sample dataframe based on the image
data = {
    'school_id': ['A', 'B', 'C', 'D', 'E'],
    'state_code': ['sch_1', 'sch_2', 'sch_3', 'sch_4', 'sch_5'],
    'subjects': ['l@sc_1', 'l)sc_2', 'l@sc_2', 'sC_2i', 'SG-1#@']
}
df = pd.DataFrame(data)

# Step 1: Count number of subjects for each school (space-separated)
df['subject_count'] = df['subjects'].str.split().str.len()

# Drop schools offering fewer than 3 subjects
df = df[df['subject_count'] >= 3].copy()

# Step 2: Clean state_code column - keep only alpha-numeric characters
df['state_code'] = df['state_code'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)

# Step 3: For each state, count schools offering English, Maths, Physics, Chemistry
# First, we need to parse the subjects properly
# Create columns for each subject
df['English'] = df['subjects'].str.lower().str.contains('english', na=False).astype(int)
df['Maths'] = df['subjects'].str.lower().str.contains('maths|math', na=False).astype(int)
df['Physics'] = df['subjects'].str.lower().str.contains('physics', na=False).astype(int)
df['Chemistry'] = df['subjects'].str.lower().str.contains('chemistry', na=False).astype(int)

# Group by state_code and sum
result = df.groupby('state_code')[['English', 'Maths', 'Physics', 'Chemistry']].sum().reset_index()

print(result)