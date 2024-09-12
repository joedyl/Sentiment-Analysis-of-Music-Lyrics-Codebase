import pandas as pd
from itertools import combinations
import numpy as np

## File inherited from Nathan Duran - his repo with the initial version of this can be found here:
## https://github.com/NathanDuran/CAMS-Dialogue-Annotation/tree/master/data_processing

def create_coder_cumulative_matrix(data):
    """Gets the sums of labels for all coders in data.

    Example of returned matrix:
          FPP-base SPP-base FPP-pre SPP-pre Pre FPP-insert SPP-insert Insert FPP-post SPP-post Post
    0       11        1       0       1   0          0          0      0        2        0    0
    1        2        2       1       0   1          7          2      0        0        0    0
    2        0        4       1       2   0          2          5      1        0        0    0
    3        0        6       1       1   0          0          4      0        2        1    0
    4        0        0       0       0   0          0          0      0        7        1    7
    5        0        0       0       0   0          0          0      0        2       10    3

    Args:
        data (dict): Dictionary of Dataframes where keys user_id. Dataframe rows are utterances for the set or
                     dialogue and columns are label for coder (1 indicates assigned label).

    Returns:
        matrix (DataFrame): Dataframe rows are utterance index and columns are labels types, items are sum of labels
                            applied to that utterance by all coders.
    """
    # Create an empty dataframe with the same column names
    column_names = data[list(data.keys())[0]].columns
    matrix = pd.DataFrame(columns=column_names)

    # Add all dataframe values together
    for user_name, data in data.items():
        matrix = matrix.add(data, fill_value=0)

    return matrix

def create_coder_sum_matrix(data):
    """Gets the sums of labels for all coders in data.

    Example of returned matrix:
           stat  ireq  chck
        joe    46    44    10
        sam    52    32    16

        joe, sam is coders and stat, ireq, chck is possible different labels
    """
    # Create a matrix with coders as rows and unique labels as columns
    matrix = data.apply(pd.Series.value_counts).fillna(0).astype(int).T

    return matrix


def pariwise_average(user_data, function):
    """Calculates an average of a given function over a set of coders pairs.

    Args:
        user_data (dict): Dictionary of users dataframes.
        function (func): The function to apply to all pairs, i.e. observed/expected agreement.

    Returns:
        average (float): The average of all the coder pairs.
    """
    # Get all the pariwise combinations of users
    user_combinations = list(combinations(user_data.keys(), 2))

    # Apply function and average
    total = 0
    n = 0
    for pair in user_combinations:
        total += function(user_data[pair[0]], user_data[pair[1]])
        n += 1

    return total / n


def observed_agreement_kappa(coder_a, coder_b):
    """Calculates the observed agreement between two coders."""
    num_items = len(coder_a)
    num_labels = len(coder_a.columns)
    total = 0
    for item in range(num_items):
        for label in range(num_labels):
            if coder_a.iloc[item, label] == 1 and coder_b.iloc[item, label] == 1:
                total += 1

    return total / num_items


def expected_agreement_kappa(coder_a, coder_b):
    """Calculates the expected agreement between two coders label distributions."""
    exp_agr = 0
    num_items = len(coder_a)

    # Get the frequency distribution of each coder
    a_lable_freq = coder_a.sum(axis=0)
    b_lable_freq = coder_b.sum(axis=0)

    # Calculate the expected agreement for each label
    for label in range(len(a_lable_freq)):
        exp_agr += (a_lable_freq.iloc[label] / num_items) * (b_lable_freq.iloc[label] / num_items)
    return exp_agr


def multi_kappa(data):
    """Davies and Fleiss (1982) Averages over observed and expected agreements for each coder pair.

        Generalisation of Cohens K for multiple coders. Expected agreement is directly computed from the observed
        distributions of the individual annotators.

        If input user_data is a dict of only two coders this if effectively just Cohen's Kappa - Cohen, J. (1960)
        A coefficient of agreement for nomial scales. Educational and Psychological Measurement.

    Args:
        data (dict): Dictionary of Dataframes where keys user_id. Dataframe rows are utterances for the set or
                          dialogue and columns are label for coder (1 indicates assigned label).

    Returns:
        multi_kappa (float): The Multi-kappa stat for the given set or dialogue and label type.
    """
    # Calculate the pairwise observed agreement
    obs_agr = pariwise_average(data, observed_agreement_kappa)

    # Calculate the pairwise expected agreement
    exp_agr = pariwise_average(data, expected_agreement_kappa)

    return (obs_agr - exp_agr) / (1.0 - exp_agr)


def multi_pi(data):
    """multi-pi = Fleiss, J.L. (1971) Measuring Nominal Scale Agreement Among Many Raters.

       Generalisation of Scots π for multiple coders. Expected agreement is estimated on the basis of the distribution
       of categories found in the observed data, but it is assumed that all coders assign categories on the basis of
       a single (but not necessarily uniform) distribution.

       If input data only has two coders (i.e. max sum of 2) this is effectively Scotts Pi -
       Scott, W.A.. (1955) Reliability of Content Analysis : The Case of Nominal Scale Coding.

    Args:
        data (DataFrame): Dictionary of Dataframes where keys user_id. Dataframe rows are utterances for the set or
                          dialogue and columns are label for coder (1 indicates assigned label).

    Returns:
        multi_pi (float): The Multi-pi stat for the given set or dialogue and label type.
    """
    # Need to get the cumulative labels for all coders in data
    data = create_coder_cumulative_matrix(data)

    # Get the number of annotators
    num_raters = data.sum(axis=1)[0]

    # Get the number of labeled items
    num_items = len(data)

    # Get the number of labels
    num_categories = len(data.columns)

    # Calculate the proportion of assignments
    pj = list(data.sum() / (num_raters * num_items))

    # Calculate expected agreement
    exp_agr = sum([i ** 2 for i in pj])

    # Calculate the extent that raters agree on each item
    pi = []
    for i in range(num_items):
        curr_item = []
        for j in range(num_categories):
            curr_item.append(data.iloc[i, j] ** 2)
        pi.append((1 / (num_raters * (num_raters - 1))) * (sum(curr_item) - num_raters))

    # Calculate observed agreements
    obs_agr = (1 / num_items) * sum(pi)

    return (obs_agr - exp_agr) / (1 - exp_agr)


def bias(data, dist_func):
    """Calculates bias of weighted measures according to Artstein, R. and Poesio, M. (2005) Kappa 3 = Alpha (or Beta)"""

    # Get data as summary matrix
    # This function has changed so this will not work!!!
    sum_matrix = create_coder_sum_matrix(data)

    # Get useful vars
    num_coders = len(sum_matrix)
    labels = list(sum_matrix.columns.values)
    num_items = sum_matrix.iloc[0].sum(axis=0)

    # All values need to be divided by num_items, so easier to do it with the dataframe
    sum_matrix = sum_matrix.apply(lambda x: x / num_items)

    # Calculate bias
    bias_val = 0
    # For each label pair
    for label_a in labels:
        for label_b in labels:
            label_pair = (label_a, label_b)

            # Individual and paired coder probs
            lhs, rhs, = 0, 0
            # For each coder get sum of probs for all label pairs
            for coder in sum_matrix.index.tolist():
                lhs += num_coders * sum_matrix.loc[coder, label_pair[0]] * sum_matrix.loc[coder, label_pair[1]]
            # lhs *= num_coders

            # For each pair of coders get sum of probs for all label pairs
            for coder_a in sum_matrix.index.tolist():
                for coder_b in sum_matrix.index.tolist():
                    rhs += sum_matrix.loc[coder_a, label_pair[0]] * sum_matrix.loc[coder_b, label_pair[1]]

            # 1/num_coders^2 * (lhs - rhs) * distance for label pair
            bias_val += (1 / num_coders**2) * (abs(lhs - rhs) * dist_func(label_pair[0], label_pair[1]))

    bias_val /= num_coders - 1
    return bias_val


def observed_disagreement(items, num_ratings, distance):
    """Observed disagreement for alpha/alpha prime and beta."""
    obs_dis_agr = 0.0
    for item in items:

        item_distances = sum(distance(i, j) for i in item for j in item)
        obs_dis_agr += item_distances / float(len(item) - 1)

    return obs_dis_agr / float(num_ratings)


def expected_disagreement_alpha(items, num_ratings, distance):
    """Coder sums / num_ratings * (num_ratings - 1)."""
    exp_dis_agr = 0.0
    for item_a in items:
        for item_b in items:
            exp_dis_agr += sum(distance(i, j) for i in item_a for j in item_b)

    exp_dis_agr /= float(num_ratings * (num_ratings - 1))
    return exp_dis_agr


def alpha(data, distance):
    """Krippendorff, K. (1980/2004) Content Analysis: An Introduction to its Methodology.

    Weighted agreement from category similarity for multiple coders.
    Assumes single probability distribution for all coders. Expected disagreement is the mean of the distances
    between all the judgment pairs in the data, without regard to items.

    Args:
        data (dict): Dictionary of Dataframes where keys user_id. Dataframe rows are utterances for the set or
                     dialogue and columns are label for coder (1 indicates assigned label).
        distance (func): Function which returns the distance between two labels (0=Min distance, 1=Max distance).

    Returns:
        alpha (float): The Alpha stat for the given set or dialogue and label type.
    """
    # matrix = create_reliability_matrix(data)

    # Convert to 2d array
    items = data.values.tolist()

    # Number of pairable values (num utterances x num coders)
    num_ratings = sum(len(item) for item in items)

    # Calculate observed disagreement
    obs_dis_agr = observed_disagreement(items, num_ratings, distance)

    # Calculate expected disagreement
    exp_dis_agr = expected_disagreement_alpha(items, num_ratings, distance)

    return 1.0 - obs_dis_agr / exp_dis_agr if (obs_dis_agr and exp_dis_agr) else 1.0


def expected_disagreement_alpha_prime(items, num_ratings, distance):
    """Coder sums / num_ratings."""
    exp_dis_agr = 0.0
    for item_a in items:
        item_sum = 0.0
        for item_b in items:
            item_sum += sum(distance(i, j) for i in item_a for j in item_b)

        exp_dis_agr += item_sum / float(num_ratings)

    exp_dis_agr /= float(num_ratings)

    return exp_dis_agr


def alpha_prime(data, distance):
    """Artstein, R. and Poesio, M. (2008) Inter-Coder Agreement for Computational Linguistics.

    Weighted agreement from category similarity for multiple coders. A slight variation of Krippendorff's alpha.
    Assumes single probability distribution for all coders. Expected disagreement is the mean of the distances between
    categories, weighted by these probabilities for all (ordered) category pairs.

    Args:
        data (dict): Dictionary of Dataframes where keys user_id. Dataframe rows are utterances for the set or
                     dialogue and columns are label for coder (1 indicates assigned label).
        distance (func): Function which returns the distance between two labels (0=Min distance, 1=Max distance).

    Returns:
        alpha_prime (float): The Alpha stat for the given set or dialogue and label type.
    """
    # matrix = create_reliability_matrix(data)
    
    # Convert to 2d array
    items = data.values.tolist()

    # Number of pairable values (coders x items)
    num_ratings = sum(len(item) for item in items)

    # Calculate observed disagreement
    obs_dis_agr = observed_disagreement(items, num_ratings, distance)

    # Calculate expected disagreement
    exp_dis_agr = expected_disagreement_alpha_prime(items, num_ratings, distance)

    return 1.0 - obs_dis_agr / exp_dis_agr if (obs_dis_agr and exp_dis_agr) else 1.0


def expected_disagreement_beta(items_matrix, num_items, distance):
    """Coder sums / num_items."""

    # Get a list of the labels so we can get the index
    labels = list(items_matrix.columns.values)

    # All values need to be divided by num_items, so easier to do it with the dataframe
    items_matrix = items_matrix.apply(lambda x: x / num_items)

    # Get the first row and then all other rows to lists
    items = items_matrix.values.tolist()
    first_row = items[0]
    other_rows = [items[i][:] for i in range(1, len(items))]

    # Iterate over first row and multiply by all other values (and distance between labels)
    exp_dis_agr = 0.0
    for curr in range(len(first_row)):
        current_val = first_row[curr]
        current_label = labels[curr]
        for row in range(len(other_rows)):
            for row_col in range(len(other_rows[row])):
                exp_dis_agr += current_val * other_rows[row][row_col] * distance(current_label, labels[row_col])

    return exp_dis_agr


def beta(data, distance):
    """Artstein, R. and Poesio, M. (2005) Kappa 3 = Alpha (or Beta).

    An agreement coefficient that is weighted, applies to multiple coders, and calculates expected agreement using a
    separate probability distribution for each coder.

    Args:
        data (dict): Dictionary of Dataframes where keys user_id. Dataframe rows are utterances for the set or
                     dialogue and columns are label for coder (1 indicates assigned label).
        distance (func): Function which returns the distance between two labels (0=Min distance, 1=Max distance).

    Returns:
        beta (float): The Alpha stat for the given set or dialogue and label type.
    """
    # Convert to 2d array
    items = data.values.tolist()

    # Number of pairable values (coders x items)
    num_ratings = sum(len(item) for item in items)

    # Calculate observed disagreement
    obs_dis_agr = observed_disagreement(items, num_ratings, distance)

    # Calculate expected disagreement
    sum_matrix = create_coder_sum_matrix(data)  # Easier to do with a summary matrix

    exp_dis_agr = expected_disagreement_beta(sum_matrix, len(data), distance)

    return 1.0 - obs_dis_agr / exp_dis_agr if (obs_dis_agr and exp_dis_agr) else 1.0

## Functionality from here onwards designed by us:

def custom_distance_func(a, b):
    value = abs(a - b) / 2
    return value 

## Preparing data into format that the alpha/beta functions like

def prepare_data(joe=pd.DataFrame(), sam=pd.DataFrame(), sinan=pd.DataFrame()):

    # Prepares data instead of the 
    if  sinan.empty:
        x_df = pd.DataFrame({'joe': joe['x_coordinate'],
                         'sam': sam['x_coordinate'],
                         })
        
        y_df = pd.DataFrame({'joe': joe['y_coordinate'],
                         'sam': sam['y_coordinate'],
                         })
        
    if  sam.empty:
        x_df = pd.DataFrame({'joe': joe['x_coordinate'],
                         'sinan': sinan['x_coordinate']})
        y_df = pd.DataFrame({'joe': joe['y_coordinate'],
                         'sinan': sinan['y_coordinate']})
        
    if  joe.empty:
        x_df = pd.DataFrame({
                         'sam': sam['x_coordinate'],
                         'sinan': sinan['x_coordinate']})
        y_df = pd.DataFrame({
                         'sam': sam['y_coordinate'],
                         'sinan': sinan['y_coordinate']})

    if not sam.empty and not joe.empty and not sinan.empty:
        x_df = pd.DataFrame({'joe': joe['x_coordinate'],
                            'sam': sam['x_coordinate'],
                            'sinan': sinan['x_coordinate']})
        
        y_df = pd.DataFrame({'joe': joe['y_coordinate'],
                            'sam': sam['y_coordinate'],
                            'sinan': sinan['y_coordinate']})

    return x_df, y_df

## Loading data

joe_data = pd.read_csv('./final_joe.csv')
sam_data = pd.read_csv('./final_sam.csv')
sinan_data = pd.read_csv('./final_sinan.csv')

x_data, y_data = prepare_data(joe=joe_data, sam=sam_data, sinan=sinan_data)

x_alpha = alpha(x_data, custom_distance_func)

y_alpha = alpha(y_data, custom_distance_func)

x_beta = beta(x_data, custom_distance_func)

y_beta = beta(y_data, custom_distance_func)

print("X alpha: ", x_alpha)

print("Y alpha: ", y_alpha)

print("X beta: ", x_beta)

print("Y beta: ", y_beta)

## Performing pairwise analysis as well as finding alpha and beta for all 3.

def pairwise_analysis():
    print("---------------- SAM & JOE----------------")
    x_data, y_data = prepare_data(joe=joe_data, sam=sam_data)
    x_alpha = alpha(x_data, custom_distance_func)

    y_alpha = alpha(y_data, custom_distance_func)

    x_beta = beta(x_data, custom_distance_func)

    y_beta = beta(y_data, custom_distance_func)

    print("X alpha: ", x_alpha)

    print("Y alpha: ", y_alpha)

    print("X beta: ", x_beta)

    print("Y beta: ", y_beta)

    print("---------------- SINAN & JOE----------------")
    x_data, y_data = prepare_data(joe=joe_data, sinan=sinan_data)
    x_alpha = alpha(x_data, custom_distance_func)

    y_alpha = alpha(y_data, custom_distance_func)

    x_beta = beta(x_data, custom_distance_func)

    y_beta = beta(y_data, custom_distance_func)

    print("X alpha: ", x_alpha)

    print("Y alpha: ", y_alpha)

    print("X beta: ", x_beta)

    print("Y beta: ", y_beta)

    print("---------------- SAM & SINAN----------------")
    x_data, y_data = prepare_data(sinan=sinan_data, sam=sam_data)
    x_alpha = alpha(x_data, custom_distance_func)

    y_alpha = alpha(y_data, custom_distance_func)

    x_beta = beta(x_data, custom_distance_func)

    y_beta = beta(y_data, custom_distance_func)

    print("X alpha: ", x_alpha)

    print("Y alpha: ", y_alpha)

    print("X beta: ", x_beta)

    print("Y beta: ", y_beta)

pairwise_analysis()
