import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

#data cleaning
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder

from  sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,Binarizer,Normalizer

from sklearn.feature_selection import SelectPercentile, f_classif,chi2,SelectKBest

from sklearn.metrics import mean_absolute_error,r2_score ,median_absolute_error ,mean_squared_error,mean_absolute_percentage_error,max_error

from sklearn.metrics import confusion_matrix ,accuracy_score,precision_score,make_scorer,recall_score, f1_score,roc_curve,classification_report,roc_auc_score

from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict

from sklearn.pipeline import Pipeline



from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


from sklearn.compose import ColumnTransformer

def parse_hidden_layers(hidden_str):
    try:
        return tuple(int(x.strip()) for x in hidden_str.split(","))
    except:
        return (100,)  




st.title("🔴 ML App _ Supervised, Unsupervised & ANN  ")


df=None
st.header("🔷 upload file")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
btd=st.button("notes")
if btd:
    st.write("⚪ You must upload a CSV file first.")
    st.write("⚪ Then, perform data cleaning.")
    st.write("⚪ Next, choose a model based on your dataset.")
    st.write("⚪ After that, you can adjust the parameters and models freely until you reach the highest accuracy.")
    st.write("⚪ Finally, enter new values and the system will generate predictions for you.")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    original_df=df.copy()
    st.header("🔷 Preview of Data:")
    btn=st.button("show data")
    if btn :
        st.table(df.head(12))
        st.write("⚪ note that weak correlation with target will removed ")
        df_corr = df.copy()
        cat_cols = df_corr.select_dtypes(include=['object', 'string']).columns
        for col in cat_cols:
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))
        f, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(f)
    
    df.drop_duplicates()
    # حذف أي عمود نصه أو أكتر NaN
    df = df.dropna(thresh=len(df)//2 + 1, axis=1)
    # حذف أي عمود عدد القيم الفريدة فيه = عدد الصفوف (يعني كله unique تقريبًا)
    #df = df.loc[:, df.nunique() < len(df)]
   


    numerical = [col for col in df.columns if df[col].dtype != "O"]

    st.header("🔷 Outliers Handling")

    if st.button("Show Outliers "):
        fig, axes = plt.subplots(len(numerical), 1, figsize=(8, 4*len(numerical)))
        if len(numerical) == 1:
            axes = [axes]
        for i, var in enumerate(numerical):
            sns.boxplot(x=df[var], ax=axes[i])
            axes[i].set_title(f"{var} (Before Removal)")
        st.pyplot(fig)

    if st.button("Remove Outliers "):
        df_clean = df.copy()
        for var in numerical:
            Q1 = df_clean[var].quantile(0.25)
            Q3 = df_clean[var].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df_clean[(df_clean[var] < (Q1 - 1.5 * IQR)) | (df_clean[var] > (Q3 + 1.5 * IQR))]
            df_clean.drop(outliers.index, inplace=True)

        fig, axes = plt.subplots(len(numerical), 1, figsize=(8, 4*len(numerical)))
        if len(numerical) == 1:
            axes = [axes]
        for i, var in enumerate(numerical):
            sns.boxplot(x=df_clean[var], ax=axes[i])
            axes[i].set_title(f"{var} (After Removal)")
        st.pyplot(fig)

        st.success("Outliers removed successfully!")
        df = df_clean
        



    target_col = st.text_input("⚪Enter target column name:")


    if target_col in df.columns:
        y_raw = df[target_col]
        X_raw = df.drop(columns=[target_col])
        X_raw = X_raw.loc[:, X_raw.nunique() < len(X_raw)]

        st.success(f"Target column set to: {target_col}")
    else:
        st.warning("⚠️ Please enter a valid column name.")

















    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])   

    
    y_le = None
    y_is_encoded = False

    if y_raw.dtype == "object" or y_raw.dtype.name == "category":
        y_le = LabelEncoder()
        y = y_le.fit_transform(y_raw)
        y_is_encoded = True
    else:
        y = y_raw.values


   
    text_columns = X_raw.select_dtypes(include=['object', 'string']).columns
    numeric_columns = X_raw.select_dtypes(exclude=['object', 'string']).columns

    # بعد تحديد text_columns و numeric_columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(text_columns)),
            ("num", "passthrough", list(numeric_columns))
        ]
    )

    # fit the preprocessor
    preprocessor.fit(X_raw)

    # transform
    X_transformed = preprocessor.transform(X_raw)

    # get output column names in a robust way
    try:
        # sklearn >= 1.0: ColumnTransformer has get_feature_names_out
        new_columns = preprocessor.get_feature_names_out()
        new_columns = list(new_columns)
    except Exception:
        # fallback: handle case where 'cat' transformer wasn't fitted (e.g., no categorical cols)
        if len(text_columns) > 0 and hasattr(preprocessor.named_transformers_["cat"], "get_feature_names_out"):
            ohe = preprocessor.named_transformers_["cat"]
            # إذا الـ OHE مُدرّب
            if hasattr(ohe, "categories_") and len(ohe.categories_) > 0:
                ohe_features = ohe.get_feature_names_out(list(text_columns))
                new_columns = list(ohe_features) + list(numeric_columns)
            else:
                # لو OHE مش مُدرّب لأي سبب — رجّع للأعمدة الرقمية فقط
                new_columns = list(numeric_columns)
        else:
            # مفيش categorical columns
            new_columns = list(numeric_columns)

    # الآن أنشئ DataFrame
    X = pd.DataFrame(X_transformed, columns=new_columns, index=X_raw.index)



    corr_matrix = X.join(pd.Series(y, name=target_col)).corr()
    target_corr = corr_matrix[target_col].abs()

    useful_features = target_corr[target_corr >= 0.05].index

    useful_features = useful_features.drop(target_col, errors="ignore")

    x= X[useful_features]


    


    
    




    st.header("🔷 cleaning data before choosing model")
    st.write(" ⚪ Keep in mind that cleaning the data is optional, not required ")   

 
    col1, col2 ,col3 = st.columns(3)
    num_imputer=None
    with col1:
        st.markdown("1)) replacing missing values ")
        numeric_imp = st.selectbox("Numeric imputation", ["mean", "median","most frequent"], key="main_num_imp")
        if numeric_imp == "mean":
            num_imputer = SimpleImputer(strategy="mean")
        elif numeric_imp == "median":
            num_imputer = SimpleImputer(strategy="median")
        else :
            num_imputer = SimpleImputer(strategy="most_frequent")
        x=num_imputer.fit_transform(x)
    scall_data=None
    with col2:
        st.markdown("2)) scalling data")
        scaler_choice = st.selectbox(
            "Scaling method",
            ["StandardScaler", "MinMaxScaler", "MaxAbsScaler", "Binarizer", "Normalizer"],
            key="scaler_choice"
        )
        if scaler_choice == "StandardScaler":
            scall_data = StandardScaler()
        elif scaler_choice == "MinMaxScaler":
            scall_data = MinMaxScaler(feature_range=(0,1))
        elif scaler_choice == "MaxAbsScaler":
            normmz=st.selectbox("normalizer with",["l1","l2","max"])
            scall_data = MaxAbsScaler(norm=normmz)
        elif scaler_choice == "Binarizer":

            binarizerr=st.number_input("threshold",min_value=0,max_value=1000,value=1,step=0.1 )
            scall_data = Binarizer(threshold=binarizerr)
        elif scaler_choice == "Normalizer":
            normm=st.selectbox("normalizer with",["l1","l2","max"])
            scall_data = Normalizer(norm=normm)
        x=scall_data.fit_transform(x)        
     
    feature_selector=None
    with col3:
        st.markdown("3)) reduce features")
        
        rfeature = st.selectbox(
        "Choose feature selection method",
        ["SelectPercentile", "SelectKBest"],
        key="feature_method"
        )

        score_func_choice = st.selectbox(
            "Select score function",
            ["f_classif", "chi2"],
            key="score_func"
        )

        if score_func_choice == "f_classif":
            score_func = f_classif
        elif score_func_choice == "chi2":
            score_func = chi2

        if rfeature == "SelectPercentile":
            percentile_value = st.number_input("Select percentile",min_value=1,max_value=100,value=10,step=1,key="percentile_val")
            feature_selector = SelectPercentile(score_func=score_func, percentile=percentile_value)
        elif rfeature == "SelectKBest":
            k_value = st.number_input("Select number of features (k)",min_value=1,max_value=100,value=5,step=1,key="k_val")
            feature_selector = SelectKBest(score_func=score_func, k=k_value)
        feature_selector=feature_selector    
     
    st.header("🔷  Validation")
    val_method = st.radio("Validation method", ("Train/Test Split", "Cross-Validation"), key="val_method")
    num = st.number_input("Random state", 42, key="main_random") 
    if val_method=="Train/Test Split" :
        testsize = st.slider("Test size (for supervised)", 0.1, 0.5, 0.25, key="main_testsize")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=num)
    if val_method=="Cross-Validation" :
        cv = st.slider("Number of CV folds", 2, 10, 5, key="main_cv")   
            
      #  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testsize,random_state=num)
    


    st.sidebar.header("1) learning type")
    choice = st.sidebar.radio("Choose learning type", ("Supervised", "Unsupervised"))          
    if choice == "Supervised":
    
        sup_type = st.sidebar.radio("task ", ("Regression","Classification"))
        model = None  
         # ----------------- Regression -----------------
        if sup_type == "Regression":
            
            algo = st.sidebar.selectbox(
                "model",
                ["Linear Regression", "Ridge Regression", "Lasso Regression",
                    "Sgd regressor","Decision Tree regressor", "Random Forest Regressor",
                    "Gradient boosting regressor","KNN reg", "SVR","ANN reg"]
            )
            st.sidebar.success(f"you choose supervised → Regression → {algo}")
            st.sidebar.header("2) parameters")

          
            if algo == "Linear Regression":
                model = LinearRegression()

         
            elif algo == "Ridge Regression":
                alpha = st.sidebar.number_input("alpha", min_value=0.0, value=1.0, step=0.1)
                model = Ridge(alpha=alpha, random_state=num)

            
            elif algo == "Lasso Regression":
                alpha = st.sidebar.number_input("alpha", min_value=0.0, value=1.0, step=0.1)
                model = Lasso(alpha=alpha, random_state=num)

       
            elif algo == "Sgd regressor":
                max_iter = st.sidebar.number_input("max_iter", min_value=100, value=1000, step=100)
                loss = st.sidebar.selectbox("loss", ["squared_error","huber","epsilon_insensitive"])
                alpha = st.sidebar.number_input("alpha", min_value=0.0001, value=0.0001, step=0.0001, format="%f")
                model = SGDRegressor(loss=loss, max_iter=max_iter, alpha=alpha, random_state=num)

           
            elif algo == "Decision Tree regressor":
                max_depth = st.sidebar.number_input("max_depth", min_value=1, value=5, step=1)
                criterion = st.sidebar.selectbox("criterion", ["squared_error","friedman_mse","absolute_error"])
                model = DecisionTreeRegressor(max_depth=max_depth, criterion=criterion, random_state=num)

    
            elif algo == "Random Forest Regressor":
                n_estimators = st.sidebar.number_input("n_estimators", min_value=10, value=100, step=10)
                max_depth = st.sidebar.number_input("max_depth", min_value=1, value=5, step=1)
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=num)

          
            elif algo == "Gradient boosting regressor":
                n_estimators = st.sidebar.number_input("n_estimators", min_value=10, value=100, step=10)
                learning_rate = st.sidebar.number_input("learning_rate", min_value=0.001, value=0.1, step=0.01, format="%.3f")
                max_depth = st.sidebar.number_input("max_depth", min_value=1, value=3, step=1)
                model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                                    max_depth=max_depth, random_state=num)

         
            elif algo == "KNN reg":
                n_neighbors = st.sidebar.slider("n_neighbors", 1, 25, 5)
                weights = st.sidebar.selectbox("weights", ["uniform","distance"])
                model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

      
            elif algo == "SVR":
                C = st.sidebar.number_input("C", min_value=0.0001, value=1.0, step=0.1, format="%f")
                kernel = st.sidebar.selectbox("kernel", ["linear","poly","rbf","sigmoid"])
                epsilon = st.sidebar.number_input("epsilon", min_value=0.0, value=0.1, step=0.01)
                model = SVR(C=C, kernel=kernel, epsilon=epsilon)

           
            elif algo == "ANN reg":
                hidden = st.sidebar.text_input("Hidden layers", value="64,32")
                activation = st.sidebar.selectbox("activation", ["relu","tanh","logistic"])
                alpha = st.sidebar.number_input("alpha", min_value=0.0, value=0.0001, step=0.0001, format="%f")
                max_iter = st.sidebar.number_input("max_iter", min_value=100, value=300, step=50)
                model = MLPRegressor(
                    hidden_layer_sizes=parse_hidden_layers(hidden),
                    activation=activation, alpha=alpha, max_iter=max_iter, random_state=num
                )
        # ----------------- Classification -----------------
        elif sup_type == "Classification":
            algo = st.sidebar.selectbox(
                "model",
                ["Logistic Regression","Sgd classifier", "Decision Tree classifier", 
                    "Random Forest classifier", "Gradient boosting classifier",
                    "KNN class", "SVC", "Naive Bayes","Lda","ANN class","voting classifier"]
            )
            st.sidebar.success(f"you choose supervised → classification → {algo}")
            st.sidebar.header("2) parameters")
            if algo == "Logistic Regression":
                C = st.sidebar.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f",key=" c r ")
                max_iter = st.sidebar.number_input("max_iter", min_value=100, value=500, step=50)
                model = LogisticRegression(C=C, max_iter=max_iter, random_state=num)

            elif algo == "Sgd classifier":
                max_iter = st.sidebar.number_input("max_iter", min_value=100, value=1000, step=100)
                loss = st.sidebar.selectbox("loss", ["hinge","log_loss","modified_huber","squared_hinge"])
                alpha = st.sidebar.number_input("alpha", min_value=0.0001, value=0.04, step=0.01, format="%f")
                model = SGDClassifier(loss=loss, max_iter=max_iter, alpha=alpha, random_state=num)

            elif algo == "Decision Tree classifier":
                max_depth = st.sidebar.number_input("max_depth", min_value=1, value=5, step=1)
                criterion = st.sidebar.selectbox("criterion", ["gini","entropy","log_loss"])
                model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=num)

            
            elif algo == "Random Forest classifier":
                n_estimators = st.sidebar.number_input("n_estimators", min_value=10, value=100, step=10)
                max_depth = st.sidebar.number_input("max_depth", min_value=1, value=5, step=1)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=num)


            elif algo == "Gradient boosting classifier":
                n_estimators = st.sidebar.number_input("n_estimators", min_value=10, value=100, step=10)
                learning_rate = st.sidebar.number_input("learning_rate", min_value=0.001, value=0.1, step=0.01, format="%.3f")
                max_depth = st.sidebar.number_input("max_depth", min_value=1, value=3, step=1)
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                                    max_depth=max_depth, random_state=num)
            elif algo == "KNN class":
                n_neighbors = st.sidebar.slider("n_neighbors", 1, 100, 5)
                weights = st.sidebar.selectbox("weights", ["uniform","distance"])
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

        
            elif algo == "SVC":
                C = st.sidebar.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f")
                kernel = st.sidebar.selectbox("kernel", ["rbf","linear","poly","sigmoid"])
                probability = st.sidebar.checkbox("Enable probability", value=True)
                model = SVC(C=C, kernel=kernel, probability=probability, random_state=num)

            elif algo == "Naive Bayes":
                var_smoothing = st.sidebar.number_input("var_smoothing", min_value=1e-10, value=1e-9, format="%e")
                choosing_technique=st.selectbox("choosing_technique",["GaussianNB","MultinomialNB","BernoulliNB"])
                if choosing_technique == "GaussianNB" :
                    model = GaussianNB(var_smoothing=var_smoothing)
                if choosing_technique == "MultinomialNB" :
                    model = MultinomialNB(var_smoothing=var_smoothing)
                if choosing_technique == "BernoulliNB" :
                    model = BernoulliNB(var_smoothing=var_smoothing)

            elif algo == "Lda":
                solver = st.sidebar.selectbox("solver", ["svd","lsqr","eigen"])
                model = LinearDiscriminantAnalysis(solver=solver)

            elif algo == "ANN class":
                hidden = st.sidebar.text_input("Hidden layers", value="100")
                activation = st.sidebar.selectbox("activation", ["relu","tanh","logistic"])
                alpha = st.sidebar.number_input("alpha", min_value=0.0, value=0.0001, step=0.0001, format="%f")
                max_iter = st.sidebar.number_input("max_iter", min_value=50, value=300, step=10)
                model = MLPClassifier(
                    hidden_layer_sizes=parse_hidden_layers(hidden),
                    activation=activation, alpha=alpha, max_iter=max_iter, random_state=num)
        
            elif algo == "voting classifier":
                    
                st.sidebar.subheader("⚪ LogisticRegression")
                C = st.sidebar.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f")
                max_iter = st.sidebar.number_input("max_iter", min_value=100, value=500, step=50)
                lr= LogisticRegression(C=C, max_iter=max_iter, random_state=num)


                st.sidebar.subheader("⚪ Sgd classifier")
                max_iter = st.sidebar.number_input("max_iter", min_value=100, value=1000, step=100)
                loss = st.sidebar.selectbox("loss", ["hinge","log_loss","modified_huber","squared_hinge"])
                alpha = st.sidebar.number_input("alpha", min_value=0.0001, value=0.04, step=0.01, format="%f")
                sgd = SGDClassifier(loss=loss, max_iter=max_iter, alpha=alpha, random_state=num)



                st.sidebar.subheader("⚪ Decision Tree classifier")
                max_depth = st.sidebar.number_input("max_depth", min_value=1, value=5, step=1,key="tree_max_depth")
                criterion = st.sidebar.selectbox("criterion", ["gini","entropy","log_loss"])
                dt = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=num)



                st.sidebar.subheader("⚪ Random Forest classifier")
                n_estimators = st.sidebar.number_input("n_estimators", min_value=10, value=100, step=10,key="rf_estimators")
                max_depth = st.sidebar.number_input("max_depth", min_value=1, value=5, step=1,key="rf_max_depth")
                rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=num)


                st.sidebar.subheader("⚪ Gradient boosting classifier")
                n_estimators = st.sidebar.number_input("n_estimators", min_value=10, value=100, step=10,key="gb_estimators")
                learning_rate = st.sidebar.number_input("learning_rate", min_value=0.001, value=0.1, step=0.01, format="%.3f")
                max_depth = st.sidebar.number_input("max_depth", min_value=1, value=3, step=1,key="gb_max_depth")
                gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                                    max_depth=max_depth, random_state=num)

                st.sidebar.subheader("⚪ KNN class")
                n_neighbors = st.sidebar.slider("n_neighbors", 1, 100, 5)
                weights = st.sidebar.selectbox("weights", ["uniform","distance"])
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

                st.sidebar.subheader("⚪ SVC")
                C = st.sidebar.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f",key="c svc")
                kernel = st.sidebar.selectbox("kernel", ["rbf","linear","poly","sigmoid"])
                probability = st.sidebar.checkbox("Enable probability", value=True)
                svc = SVC(C=C, kernel=kernel, probability=probability, random_state=num)

                st.sidebar.subheader("⚪ Lda")
                solver = st.sidebar.selectbox("solver", ["svd","lsqr","eigen"])
                lda = LinearDiscriminantAnalysis(solver=solver)





                voting = st.sidebar.selectbox("voting", ["hard","soft"])
                model = VotingClassifier(estimators=[
                    ('lr', lr), 
                    ('sgd', sgd),
                    ('dt', dt),
                    ('rfc', rfc),
                    ('gbc', gbc),
                    ('knn',knn),
                    ('lda', lda)


                ], voting=voting)

            



       



    elif choice == "Unsupervised":
        algo = st.sidebar.selectbox(
            "model",
            ["KMeans", "DBSCAN", "Agglomerative Clustering"]
        )
        st.sidebar.success(f"you choose Unsupervised → {algo}")
        st.sidebar.header("2) parameters")

        model = None 
        n_components = st.sidebar.slider("Number of components", 1, min(x.shape[1], 10), 2)
        svd_solver = st.sidebar.selectbox("SVD Solver", ["auto", "full", "arpack", "randomized"], index=0)
        pca = PCA(n_components=n_components, svd_solver=svd_solver)
        transformed = pca.fit_transform(x)
        if algo == "KMeans":
            n_clusters = st.sidebar.slider("Number of clusters", 2, 20, 3)
            init = st.sidebar.selectbox("Initialization method", ["k-means++", "random"], index=0)
            max_iter = st.sidebar.number_input("Max iterations", min_value=100, value=300, step=50)
            random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)
            model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=random_state)

        elif algo == "DBSCAN":
            eps = st.sidebar.number_input("eps", min_value=0.1, value=0.5, step=0.1)
            min_samples = st.sidebar.number_input("min_samples", min_value=1, value=5, step=1)
            metric = st.sidebar.selectbox("Metric", ["euclidean", "manhattan", "cosine"], index=0)
            model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

        elif algo == "Agglomerative Clustering":
            n_clusters = st.sidebar.slider("Number of clusters", 2, 20, 3)
            metric = st.sidebar.selectbox("metric", ["euclidean", "manhattan", "cosine"], index=0)
            linkage = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"], index=0)
            model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)


    model=model    
    
     
      
    
    
    
   
    st.divider()
    st.subheader("🔷 Model Evaluation")



    if choice == "Supervised" and val_method=="Train/Test Split":
        
        

        model.fit(x_train,y_train)     
        y_pred = model.predict(x_test)   




        if sup_type == "Classification":
            

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fg,ax = plt.subplots(figsize=(15,15))
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("actual")
            ax.set_xticks(range(len(np.unique(y_test))))
            ax.set_yticks(range(len(np.unique(y_test))))
    
            sns.heatmap(cm, annot=True,ax=ax)
            st.pyplot(fg)

            
            st.text("Classification report:")
            st.code(classification_report(y_test, y_pred))

        # ---------------- Regression ----------------
        else:
            mse   = mean_squared_error(y_test, y_pred)
            mae   = mean_absolute_error(y_test, y_pred)
            medae = median_absolute_error(y_test, y_pred)
            r2    = r2_score(y_test, y_pred)
            mape  = mean_absolute_percentage_error(y_test, y_pred) * 100  
            maxer = max_error(y_test, y_pred)

            st.metric("R2 (test accuracy)", f"{r2:.4f}")
            st.metric("MSE", f"{mse:.4f}")
            st.metric("MAE", f"{mae:.4f}")
            st.metric("MedAE", f"{medae:.4f}")
            st.metric("MAPE (%)", f"{mape:.2f}")
            st.metric("Max Error", f"{maxer:.4f}")

            # Actual vs Predicted
            fig_fit, ax_fit = plt.subplots()
            ax_fit.scatter(y_test, y_pred, alpha=0.7,c="g", label="Data points")
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax_fit.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect fit (y = x)")
            ax_fit.set_xlabel("Actual")
            ax_fit.set_ylabel("Predicted")
            ax_fit.set_title("Actual vs Predicted")
            ax_fit.legend()
            st.pyplot(fig_fit)

            resid = y_test - y_pred
            fig, ax = plt.subplots()
            ax.scatter(y_pred, resid)
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Predicted")
            st.pyplot(fig)

            
            if isinstance(model, MLPRegressor) and hasattr(model, "loss_curve_"):
                fig2, ax2 = plt.subplots()
                ax2.plot(model.loss_curve_)
                ax2.set_title("Training Loss (MLP)")
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Loss")
                st.pyplot(fig2) 
    if choice == "Supervised" and val_method=="Cross-Validation":
        
        if sup_type == "Classification":
        
            scorers = {
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score, average="weighted", zero_division=0),
                "recall": make_scorer(recall_score, average="weighted", zero_division=0),
                "f1": make_scorer(f1_score, average="weighted", zero_division=0),
            }
          
            if len(np.unique(y)) == 2:
                scorers["roc_auc"] = make_scorer(roc_auc_score, needs_proba=True)

            scores = cross_validate(model, x, y, cv=cv, scoring=scorers)
         
            st.metric("Accuracy (test accuracy)", f"{scores['test_accuracy'].mean():.4f}")
            st.metric("Precision", f"{scores['test_precision'].mean():.4f}")
            st.metric("Recall", f"{scores['test_recall'].mean():.4f}")
            st.metric("F1 Score", f"{scores['test_f1'].mean():.4f}")
            if "test_roc_auc" in scores:
                st.metric("ROC AUC", f"{scores['test_roc_auc'].mean():.4f}")

        # ---------------- Regression ----------------
        else:
            scorers = {
                "r2 (test accuracy)": make_scorer(r2_score),
                "mse": make_scorer(mean_squared_error),
                "mae": make_scorer(mean_absolute_error),
                "medae": make_scorer(median_absolute_error),
                "mape": make_scorer(mean_absolute_percentage_error),
                "max_error": make_scorer(max_error),
            }

            scores = cross_validate(model, x, y, cv=cv, scoring=scorers)

            st.metric("MSE", f"{scores['test_mse'].mean():.4f}")
            st.metric("MAE", f"{scores['test_mae'].mean():.4f}")
            st.metric("MedAE", f"{scores['test_medae'].mean():.4f}")
            st.metric("R2", f"{scores['test_r2 (test accuracy)'].mean():.4f}")
            st.metric("MAPE (%)", f"{scores['test_mape'].mean()*100:.2f}")
            st.metric("Max Error", f"{scores['test_max_error'].mean():.4f}")     

    if choice == "Unsupervised" :
        

        with st.spinner("Fitting clustering model..."):
            
            model.fit(x)

            labels = model.labels_

        st.success("Clustering done ✅")
        st.write("Cluster label counts:")
        st.write(pd.Series(labels).value_counts().sort_index())

        from sklearn.metrics import silhouette_score
        try:
       
            transformed = model.transform(x)
            if len(np.unique(labels)) > 1 and transformed.shape[0] > len(np.unique(labels)):
                sil = silhouette_score(transformed, labels)
                st.metric("Silhouette Score", f"{sil:.4f}")
        except Exception as e:
            st.warning(f"Silhouette not available: {e}")


        try:
            pca_vis = PCA(n_components=2, random_state=num)
            reduced = pca_vis.fit_transform(transformed)
            fig, ax = plt.subplots()
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10")
            ax.set_title("PCA (2D) of Clusters")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            st.pyplot(fig)
        except Exception:
            st.info("PCA plot not available.")


    st.divider()
    st.subheader("🔷 Predict on Custom Input")
    st.caption("📝 Provide feature values as in the original dataset.")
    with st.form("predict_form"):
        input_data = {}
        



        for col in original_df.drop(columns=[target_col]).columns: 
            if pd.api.types.is_numeric_dtype(original_df[col]):
                default_val = float(original_df[col].median()) if pd.notnull(original_df[col].median()) else 0.0
                val = st.number_input(f"{col}", value=default_val)
            else:
                opts = sorted(original_df[col].astype(str).dropna().unique().tolist())
                opts = opts if opts else ["missing"]
                val = st.selectbox(f"{col}", opts)

            input_data[col] = val

        submitted = st.form_submit_button("🚀 Predict")

        if submitted:
            input_df = pd.DataFrame([input_data])

            if val_method == "Cross-Validation":
        
                cv_preds = cross_val_predict(model, x, y, cv=cv)
                last_pred = cv_preds[-1]

                if y_is_encoded and y_le is not None:
                    last_pred_decoded = y_le.inverse_transform([int(last_pred)])[0]
                else:
                    last_pred_decoded = last_pred

    
                st.success(f"🎯 Cross-Validation Prediction: {last_pred_decoded}")

            else:
              
                df_temp = preprocessor.transform(input_df)

                df_temp = pd.DataFrame(df_temp, columns=new_columns)


                df_temp = df_temp[useful_features.drop(target_col, errors="ignore")]

                
                pred_encoded = model.predict(df_temp)  # مصفوفة (حتى لو عنصر واحد)

  
                if choice == "Supervised" and sup_type == "Classification" and y_is_encoded and y_le is not None:
                    try:
                        pred_decoded = y_le.inverse_transform(pred_encoded.astype(int))
                    except Exception:
                        pred_decoded = y_le.inverse_transform(pred_encoded)

                    st.success(f"🎉 Prediction: {pred_decoded[0]}")

          
                    if hasattr(model["model"], "predict_proba"):
                        proba = model.predict_proba(df_temp)
                        class_labels = model["model"].classes_

                        if y_is_encoded:
                            try:
                                class_labels = y_le.inverse_transform(class_labels.astype(int))
                            except Exception:
                                class_labels = y_le.inverse_transform(class_labels)

                        st.write("🔎 Class probabilities:")
                        st.write(pd.DataFrame(proba, columns=[f"class_{c}" for c in class_labels]))


                elif choice == "Supervised" and sup_type == "Classification":
                    st.success(f"🎉 Prediction: {pred_encoded[0]}")

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(df_temp)
                        class_labels = model.classes_
                        st.write("🔎 Class probabilities:")
                        st.write(pd.DataFrame(proba, columns=[f"class_{c}" for c in class_labels]))

                else:
                    pred_value = pred_encoded[0]
                    st.success(f"🎉 Prediction: {pred_value}")

                    if isinstance(model, MLPRegressor) and hasattr(model, "loss_curve_"):
                        fig2, ax2 = plt.subplots()
                        ax2.plot(model.loss_curve_)
                        ax2.set_title("Training Loss (MLP)")
                        ax2.set_xlabel("Iteration")
                        ax2.set_ylabel("Loss")
                        st.pyplot(fig2)


        







































