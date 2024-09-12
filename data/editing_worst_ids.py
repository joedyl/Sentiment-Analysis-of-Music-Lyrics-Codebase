## File to run in the terminal to edit our final dataframe to reannotate worst 100 ids

## First step is to find a list of the worst 100 ids, from all 3 separate datasets

import pandas as pd

# Loading in data

joe_final = pd.read_csv('./final_joe.csv')

sinan_final = pd.read_csv('./final_sinan.csv')

sam_final = pd.read_csv('./final_sam.csv')

# Setting number of songs to reannotate
N = 100

sinan_coord = sinan_final[["x_coordinate", "y_coordinate"]]


sam_coord = sam_final[["x_coordinate", "y_coordinate"]]


joe_coord = joe_final[["x_coordinate", "y_coordinate"]]

sinan_coord['rater_id'] = 1
sam_coord['rater_id'] = 2
joe_coord['rater_id'] = 3

sinan_coord['question_id'] = sinan_coord.index + 1
sam_coord['question_id'] = sam_coord.index + 1
joe_coord['question_id'] = joe_coord.index + 1

sn_df = pd.merge(sinan_coord, sam_coord, on='question_id', suffixes=('_sinan', '_sam'))
final_df = pd.merge(sn_df, joe_coord, on='question_id')
final_df.rename(columns={'x_coordinate': 'x_coordinate_joe', 'y_coordinate': 'y_coordinate_joe'}, inplace=True)

# Find distances using custom distance function used in alpha calculation

def custom_distance_func(a, b):
    value = abs(a - b) / 2

    return value 

def compute_distances(row):
    sinan_sam = (custom_distance_func(row['x_coordinate_sinan'], row['x_coordinate_sam']),
                               custom_distance_func(row['y_coordinate_sinan'], row['y_coordinate_sam']))
    sinan_joe = (custom_distance_func(row['x_coordinate_sinan'], row['x_coordinate_joe']),
                               custom_distance_func(row['y_coordinate_sinan'], row['y_coordinate_joe']))
    sam_joe = (custom_distance_func(row['x_coordinate_sam'], row['x_coordinate_joe']),
                             custom_distance_func(row['y_coordinate_sam'], row['y_coordinate_joe']))
    
    total_distance = sinan_sam[0] + sinan_sam[1] + sinan_joe[0] + sinan_joe[1] + sam_joe[0] + sam_joe[1]
    return total_distance

final_df['total_distance'] = final_df.apply(compute_distances, axis=1)

#Get the top N songs with the worst agreement
top_N_worst = final_df.nlargest(N, 'total_distance')

bad_ids = top_N_worst['question_id'].to_list()


## Now bad_ids are found, use this list to index the final dataset and reannotate those songs.


# Sample CSV file path
csv_file_path = "final.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Iterate through the list of IDs
for id_ in bad_ids:
    # Find the row corresponding to the current ID
    row = df[df['question_id'] == id_]

    if not row.empty:
        # Display the lyrics
        print(f"ID: {id_}")
        print("Lyrics:")
        print(row.iloc[0]['lyrics'])
        print(f"Current x: {row.iloc[0]['x_coord']}, Current y: {row.iloc[0]['y_coord']}")
        
        # Get new x and y values from the user
        new_x = input("Enter new x value (or press Enter to keep current): ")
        new_y = input("Enter new y value (or press Enter to keep current): ")
        
        # Update the DataFrame with the new values if provided
        if new_x:
            df.loc[df['question_id'] == id_, 'x_coord'] = new_x
        if new_y:
            df.loc[df['question_id'] == id_, 'y_coord'] = new_y
    else:
        print(f"ID {id_} not found in the CSV.")

# Save the updated DataFrame back to the CSV file
df.to_csv(csv_file_path, index=False)
print('Updated successfully')