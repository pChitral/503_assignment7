import pandas as pd

# Read the following link and complete this homework. https://www.codemag.com/Article/1711091/Implementing-Machine-Learning-Using-Python-and-Scikit-learn

# Make sure to install scikit-learn and Pandas

def step1():
    """
    # Step 1: Getting the Titanic Dataset
    Return a dataframe containing the Titantic dataset from the following URL
    # URL: https://gist.githubusercontent.com/mkzia/aa4f293661dba857b8c4459c0095ac95/raw/8075037f6f7689a1786405c1bc8ea9471d3aa9c3/train.csv

    """
    # BEGIN SOLUTION
    return pd.read_csv("https://gist.githubusercontent.com/mkzia/aa4f293661dba857b8c4459c0095ac95/raw/8075037f6f7689a1786405c1bc8ea9471d3aa9c3/train.csv")
    # END SOLUTION
    # return df


def step2(df):
    """
    # Step 2: Clean data
    Modify df to drop the following columns:
    PassengerId
    Name
    Ticket
    Cabin
    Hint: Just pass all the columns to the .drop() method as an array
    return dataframe
    """
    # BEGIN SOLUTION

    df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace = True)
    return df
    # END SOLUTION
    # return df


def step3(df):
    """
    # Step 3: Drop NaNs and reindex
    You want to reindex so your index does not have missing values after you drop the NaNs. Remember, index is used 
    to access a row. Notice how many rows you dropped!
    Modify df to drop NaNs and reindex
    return dataframe
    """
    # BEGIN SOLUTION
 
    return df.dropna().reset_index(drop = True)
    # END SOLUTION
    # return df


def step4(df):
    """
    # Step 4: Encoding the Non-Numeric Fields
    Encode text fields to numbers
    Modify df to encode Sex and Embarked to encoded values.
    return dataframe
    """
    # BEGIN SOLUTION
    encode_sex_hasho = {'male': 1, 'female': 0}
    encode_embarked_hasho = {'S': 2, 'C': 0, 'Q': 1}
    df = df.replace({'Sex':encode_sex_hasho, 'Embarked':encode_embarked_hasho})
    # END SOLUTION
    return df


def step5(df):
    """
    # Step 5: Making Fields Categorical
    Turn values that are not continues values into categorical values
    Modify df to make Pclass, Sex, Embarked, and Survived a categorical field
    return dataframe
    """
    # BEGIN SOLUTION
    categorical_fields = ['Pclass', 'Sex', 'Embarked', 'Survived']
    df[categorical_fields] = df[categorical_fields].apply(lambda x: x.astype('category'))

    return df
    # END SOLUTION
    # return df categorical_fields


def step6(df):
    """
    1. Split dataframe into feature and label
    2. Do train and test split; USE: random_state = 1
    4. Use LogisticRegression() for classification
    3. Return accuracy and confusion matrix

    Use  metrics.confusion_matrix to calculate the confusion matrix
    # https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # IMPORTANT !!!! 
    # https://stackoverflow.com/questions/56078203/why-scikit-learn-confusion-matrix-is-reversed

    From the confusion matrix get TN, FP, FN, TP

    return --> accuracy, TN, FP, FN, TP; 
    Hint: round accuracy to 4 decimal places

    """
    # BEGIN SOLUTION
    
    from sklearn import metrics
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split


    data = df.drop("Survived", axis = 1)

    train_df, test_df, train_label, test_label = train_test_split(data, df["Survived"], test_size = 0.25, random_state = 1, stratify = df["Survived"])
    
    log_fit = linear_model.LogisticRegression().fit(X = train_df, y = train_label)
    log_pred = log_fit.predict(test_df)

    accuracy = round(log_fit.score(X = test_df, y = test_label), 4)

    conf_m = metrics.confusion_matrix(y_true = test_label, y_pred = log_pred)

    TN, FN, TP, FP = conf_m[0][0], conf_m[1][0], conf_m[1][1], conf_m[0][1]


    return accuracy, TN, FP, FN, TP
    # END SOLUTION
    # return accuracy, TN, FP, FN, TP
