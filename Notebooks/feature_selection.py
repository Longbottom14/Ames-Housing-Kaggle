"""
* Feature Selection
    -----------------------------------------------------
    NOTE:
    You needn't restrict development to only these top features, 
    but you do now have a good place to start. 
    Combining these top features with other related features, 
    especially those you've identified as creating interactions, 
    is a good strategy for coming up with a highly informative set of features to train your model on.
    """

"""
Tips on Discovering New Features
    ------------------------------------------------------

    * Understand the features. 

    * Refer to your dataset's data documentation, if available.

    * Research the problem domain to acquire domain knowledge. 

    * If your problem is predicting house prices, do some research on real-estate for instance.
        - Wikipedia can be a good starting point, 
        - but books and journal articles will often have the best information.

    * Study previous work. Solution write-ups from past Kaggle competitions are a great resource.

    * Use data visualization. 
        - Visualization can reveal pathologies in the distribution of a feature or
        - complicated relationships that could be simplified. 
        - Be sure to visualize your dataset as you work through the feature engineering process.
"""

"""
Tips on Creating Features
    ---------------------------------------------------------
    It's good to keep in mind your model's own strengths and weaknesses when creating features. 

    Here are some guidelines:
        -------------------------
        * Linear models learn sums and differences naturally, but can't learn anything more complex.

        * Ratios seem to be difficult for most models to learn. 

        * Ratio combinations often lead to some easy performance gains.

        * Linear models and neural nets generally do better with normalized features. 

        * Neural nets especially need features scaled to values not too far from 0. 

        * Tree-based models (like random forests and XGBoost) can sometimes benefit from normalization, 
            - but usually much less so.

        * Tree models can learn to approximate almost any combination of features, 
            - but when a combination is especially important they can still benefit from having it explicitly created, especially when data is limited.

        * Counts are especially helpful for tree models, 
            - since these models don't have a natural way of aggregating information across many features at once.
"""


"""
Method Creating New Features
    -----------------------------------------------------------------
    * Mathematical Transformation:
        ------------------------------
        - Relationships among numerical features are often expressed through mathematical formulas, 
        - which you'll frequently come across as part of your domain research. 
        - In Pandas, you can apply arithmetic operations to columns just as if they were ordinary numbers.

    * Counts:
        ---------
        - Features describing the presence or absence of something often come in sets,
        - the set of risk factors for a disease, say.
        - You can aggregate such features by creating a count.

        - These features will be binary (1 for Present, 0 for Absent) or boolean (True or False). 
        - In Python, booleans can be added up just as if they were integers.
                e.g >>roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay","Junction"]
                    >> accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)
                    O/P : accidents["RoadwayFeatures"] = (count of object in each row)

        - You could also use a dataframe's built-in methods to create boolean values.
        - Many formulations lack one or more components (that is, the component has a value of 0). 
        - This will count how many components are in a formulation with the dataframe's built-in 
                -- greater-than gt method:
                 e.g >>components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water","FineAggregate"]
                 >> concrete["Components"] = concrete[components].gt(0).sum(axis=1)
                 O/P : concrete["Components"] = (count of object>0 in each row)


    * Building-Up and Breaking-Down Features:
        -----------------------------------------
        Often you'll have complex strings that can usefully be broken into simpler pieces. 
        Some common examples:

            - ID numbers: '123-45-6789'
            - Phone numbers: '(999) 555-0123'
            - Street addresses: '8241 Kaggle Ln., Goose City, NV'
            - Internet addresses: 'http://www.kaggle.com
            - Product codes: '0 36000 29145 2'
            - Dates and times: 'Mon Sep 30 07:06:05 2013'

        ----> Features like these will often have some kind of structure that you can make use of.
        US phone numbers, for instance, have an area code (the '(999)' part)
        that tells you the location of the caller. As always, some research can pay off here.
        ---->
        You could also join simple features into a composed feature 
        if you had reason to believe there was some interaction in the combination:
            e.g >> autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]


    Group Transforms:
        -------------------------------------------------------------------------
        - Finally we have Group transforms, 
        - which aggregate information across multiple rows grouped by some category.
        - With a group transform you can create features like:
        "the average income of a person's state of residence," or
        "the proportion of movies released on a weekday, by genre."
        - If you had discovered a category interaction, 
        - a group transform over that categry could be something good to investigate.

        - Using an aggregation function,
        - a group transform combines two features: 

        - a categorical feature that provides the grouping and 
        - another feature whose values you wish to aggregate.

        - For an "average income by state", 
            - you would choose State for the grouping feature, 
            - mean for the aggregation function,
                    - Other handy methods include [max, min, median, var, std, and count]. 
                    - Here's how you could calculate the frequency with 
                        which each state occurs in the dataset.
            - and Income for the aggregated feature. 
            
        - To compute this in Pandas, we use the groupby and transform methods:
            >> customer["StateFreq"]=(
                                        customer.groupby("State")["State"].transform("count")/ customer.State.count()
        - NOTE:
            - If you're using training and validation splits, to preserve their independence, 
            - * it's best to create a grouped feature using only the training set *
            - and then join it to the validation set. 
            - We can use the validation set's merge method 
            - after creating a unique set of values with drop_duplicates on the training set:                                                           )

"""

from sklearn.feature_selection import SelectKBest,chi2, f_classif,SelectFpr ,mutual_info_regression

def feature_SelectKBest(X_train,y_train,score_func = chi2, k=5):
    """
    params : X_train = Pandas dataframe of X
            y_train = Pandas dataframe of y
            
            score_func = Scores use for the univariate feature selection (chi2, f_classif,SelectFpr) 
                        default = chi2, 
            k= Number of features to be selected
                default = 5
    return :5best features
    """
    #from sklearn.feature_selection import SelectKBest,chi2, f_classif,SelectFpr

    import pandas as pd
    
    bestfeatures = SelectKBest(score_func= score_func, k=k)
    fit = bestfeatures.fit(X_train,y_train)

    dfscores = pd.DataFrame(fit.scores_)

    dfcolumns = pd.DataFrame(X_train.columns)

    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Features','Score'] #naming the dataframe columns

    print(featureScores.nlargest(k,'Score')) #print 5best features



def feature_Boruta(X_train_df,X_train_values, y_train_values,classifier):
    """
    params :- X_train_df = Pandas dataframe of X values
           :- X_train_values = numpy array of X values
           :- y_train_values = numpy array of y values
           :- classifier = Classifier to run  the test on i.e
                           #from sklearn.ensemble import RandomForestClassifier
                            #rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
                            
    NOTE: 
        ->BorutaPy accepts numpy arrays only,
        ----> if X_train and y_train #are pandas dataframes, 
        ---->then add .values attribute X_train.values in #that case
                X_train = X_train.values
                y_train = y_train.values
        ----->works with only Xgboots,randomforrest and extratrees

    return : new X_train now with selected features         
             
    """
    from boruta import BorutaPy
    #sampling in proportion to y labels
    
    #define Boruta feature selection method
    feat_selector = BorutaPy(classifier, n_estimators='auto', verbose=2, random_state=1)

    #find all relevant features - 5 features should be selected
    feat_selector.fit(X_train_values, y_train_values)

    #check selected features - first 5 features are selected
    feat_selector.support_

    #check ranking of features
    feat_selector.ranking_

    #call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X_train_values)

    #To get the new X_train now with selected features
    return X_train_df.columns[feat_selector.support_]




def feature_Wrapper_RFE(X_train,y_train,classifier,n_features):
    """
    params:
            :- X_train = Pandas dataframe of X_train
            :- y_train = Pandas dataframe of y_train
            :- classifier = Classifier to run  the test on i.e
                           #from sklearn.ensemble import RandomForestClassifier
                           #from sklearn.linear_model import LogisticRegression
                           #The choice of algorithm does not matter too much as long as it is skillful and consistent.
                           
    Output:
            :-Num Features: 3

            :-Selected Features: [ True False False False False True True False]

            :-Feature Ranking: [1 2 3 5 6 1 1 4]
     return :  X_train now with selected features columns
    """
    #Import your necessary dependencies
    from sklearn.feature_selection import RFE
    #You will use RFE with the  classifier to select the top n_features.

    #Feature extraction
    rfe = RFE(classifier, n_features)
    fit = rfe.fit(X_train, y_train)

    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))

    return X_train.columns[fit.support_]





def feature_RidgeRegression(X_train,y_train):
    """
      Ridge regression to determine the coefficient R2.
     
     params:
            :- X_train = Pandas dataframe of X_train
            :- y_train = Pandas dataframe of y_train
            
    helper_function :- A helper method for pretty-printing the coefficients
                            >> pretty_print_coefs(coefs, names = None, sort = False)
                            
                    :- pass Ridge model's coefficient terms to this little function
                            >> pretty_print_coefs(ridge.coef_)
                            
    interpretation:
            :-Below are some points that you should keep in mind while applying Ridge regression:
            
                - It is also known as L2-Regularization.

                - For correlated features, it means that they tend to get similar coefficients.

                - Feature having negative coefficients don't contribute that much.

                - But in a more complex scenario where you are dealing with lots of features
                
    output:
            Ridge model: 0.021 * X0 + 0.006 * X1 + -0.002 * X2 + 0.0 * X3 + -0.0 * X4 + 0.013 * X5 + 0.145 * X6 + 0.003 * X7
            
    return: Pandas series (data=ridge.coef_,index=columns,name= 'Ridge_Coef')
    """
    from sklearn.linear_model import Ridge
    #Also, check scikit-learn's official documentation on Ridge regression.
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train,y_train)

    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
    
    #A helper method for pretty-printing the coefficients
    def pretty_print_coefs(coefs, names = None, sort = False):
        if names == None:
            names = ["X%s" % x for x in range(len(coefs))]
        lst = zip(coefs, names)
        if sort:
            lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
        return " + ".join("%s * %s" % (round(coef, 3), name)
                                       for coef, name in lst)
    columns = X_train.columns
    print ("Ridge model:", pretty_print_coefs(ridge.coef_))
    df =pd.Series(data=ridge.coef_,index=columns,name= 'Ridge_Coef')
    return df
    
    

#from sklearn.feature_selection import mutual_info_regression

def make_mi_scores_Regression(X_train, y_train):
    """
    get the mutual importance score of each feature

        mi_scores = make_mi_scores_Regression(X, y)
        mi_scores[::3]  # show a few features with their MI scores

    params:
            :- X_train = Pandas dataframe of X_train
            :- y_train = Pandas dataframe of y_train
            
    NOTE :
        discrete_features = Bool with shape (n_features,) to determines whether to consider 
                                        all features discrete or continuous. 
                                        or array with indices of discrete features.

                            But i have set it to the former
    
    return : pandas series of mutual index scores       

    """
    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X_train.dtypes == int
    mi_scores = mutual_info_regression(X_train, y_train, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_train.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

#mi_scores = make_mi_scores_Regression(X, y)
#mi_scores[::3]  # show a few features with their MI scores



def plot_mi_scores(scores):
    """
    Plot Mutual Information Scores.
    
        plt.figure(dpi=100, figsize=(8, 5))
        plot_mi_scores(mi_scores)

    Param:
            :-Scores = Pandas series of mi_scores gotten from the make_mi_score function
                        i.e>>>mi_scores = make_mi_scores_Regression(X, y)
    
    output: Barchat of mi_scores and the respective feature
    """
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def make_mi_scores_Classif(X_train, y_train):
    """
    get the mutual importance score of each feature

        mi_scores = make_mi_scores_Classif(X, y)
        mi_scores[::3]  # show a few features with their MI scores

    params:
            :- X_train = Pandas dataframe of X_train
            :- y_train = Pandas dataframe of y_train
            
    NOTE :
        discrete_features = Bool with shape (n_features,) to determines whether to consider 
                                        all features discrete or continuous. 
                                        or array with indices of discrete features.

                            But i have set it to the former
    
    return : pandas series of mutual index scores       

    """
    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X_train.dtypes == int
    mi_scores = mutual_info_classif(X_train, y_train, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_train.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


