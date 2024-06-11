import numpy as np


def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1-p) * np.log2(1-p)

def split_indices(X, index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have
    that feature = 1 and the right node those that have the feature = 0
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []
    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices


def weighted_entropy(X, y, left_indices, right_indices):
    """
    This function takes the splitted dataset, the indices we chose
    to split and return the weight entropy.
    """
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)

    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy



left_indices, right_indices = split_indices(X_train, 0)
weighted_entropy(X_train, y_train, left_indices, right_indices)


def information_gain(X, y, left_indices, right_indices):
    """
    Here, X has elements in the node and y is their respective classes
    """
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X, y, left_indices, right_indices)
    return h_node - w_entropy







# One hot encoding

df = pd.read_csv('...')

cat_variables = ['Sex',
    'ChestPainType',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]


df = pd.get_dummies(data = df,
                    prefix = cat_variables,
                    columns = cat_variables)


features = [x for x in df.columns if x not in 'HeartDisease'] ## Removing our target variable


from skleearn.tree import DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier(min_samples_split = 50,
                                             max_depth = 3,
                                             random_state = RANDOM_STATE).fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators = 100,
                                             max_depth = 16,
                                             min_samples_split = 10).fit(X_train, y_train)

from xgboost import XGBClassifier

# One interesting thing about the XGBoost is that during fitting, it can take in an evaluation dataset of the form (X_val,y_val).
#
# On each iteration, it measures the cost (or evaluation metric) on the evaluation datasets.
# Once the cost (or metric) stops decreasing for a number of rounds (called early_stopping_rounds), the training will stop.
# More iterations lead to more estimators, and more estimators can result in overfitting.
# By stopping once the validation metric no longer improves, we can limit the number of estimators created, and reduce overfitting.


# We can then set a large number of estimators, because we can stop if the cost function stops decreasing.
#
# Note some of the .fit() parameters:
#
# eval_set = [(X_train_eval,y_train_eval)]:Here we must pass a list to the eval_set, because you can have several different tuples ov eval sets.
# early_stopping_rounds: This parameter helps to stop the model training if its evaluation metric is no longer improving on the validation set. It's set to 10.
# The model keeps track of the round with the best performance (lowest evaluation metric). For example, let's say round 16 has the lowest evaluation metric so far.
# Each successive round's evaluation metric is compared to the best metric. If the model goes 10 rounds where none have a better metric than the best one, then the model stops training.
# The model is returned at its last state when training terminated, not its state during the best round. For example, if the model stops at round 26, but the best round was 16, the model's training state at round 26 is returned, not round 16.
# Note that this is different from returning the model's "best" state (from when the evaluation metric was the lowest).



xgb_model = XGBClassifier(n_estimators = 500,
                          learning_rate = 0.1,
                          verbosity = 1,
                          random_state = RANDOM_STATE)
xgb_model.fit(X_train_fit, y_train_fit,
              eval_set = [(X_train_eval, y_train_eval)],
              early_stopping_rounds = 10)


# Even though we initialized the model to allow up to 500 estimators, the algorithm only fit 26 estimators (over 26 rounds of training).
#
# To see why, let's look for the round of training that had the best performance (lowest evaluation metric). You can either view the validation log loss metrics that were output above, or view the model's .best_iteration attribute:

xgb_model.best_iteration


# The best round of training was round 16, with a log loss of 4.3948.
#
# For 10 rounds of training after that (from round 17 to 26), the log loss was higher than this.
# Since we set early_stopping_rounds to 10, then by the 10th round where the log loss doesn't improve upon the best one, training stops.
# You can try out different values of early_stopping_rounds to verify this. If you set it to 20, for instance, the model stops training at round 36 (16 + 20).
