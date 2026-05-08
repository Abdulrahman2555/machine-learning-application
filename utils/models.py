import streamlit as st
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

from utils.helpers import parse_hidden_layers


def build_supervised_model(sup_type, num):
    model = None

    # ─── Regression ───
    if sup_type == "Regression":
        algo = st.sidebar.selectbox(
            "model",
            ["Linear Regression", "Ridge Regression", "Lasso Regression",
             "Sgd regressor", "Decision Tree regressor", "Random Forest Regressor",
             "Gradient boosting regressor", "KNN reg", "SVR", "ANN reg"]
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
            loss = st.sidebar.selectbox("loss", ["squared_error", "huber", "epsilon_insensitive"])
            alpha = st.sidebar.number_input("alpha", min_value=0.0001, value=0.0001, step=0.0001, format="%f")
            model = SGDRegressor(loss=loss, max_iter=max_iter, alpha=alpha, random_state=num)

        elif algo == "Decision Tree regressor":
            max_depth = st.sidebar.number_input("max_depth", min_value=1, value=5, step=1)
            criterion = st.sidebar.selectbox("criterion", ["squared_error", "friedman_mse", "absolute_error"])
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
            weights = st.sidebar.selectbox("weights", ["uniform", "distance"])
            model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

        elif algo == "SVR":
            C = st.sidebar.number_input("C", min_value=0.0001, value=1.0, step=0.1, format="%f")
            kernel = st.sidebar.selectbox("kernel", ["linear", "poly", "rbf", "sigmoid"])
            epsilon = st.sidebar.number_input("epsilon", min_value=0.0, value=0.1, step=0.01)
            model = SVR(C=C, kernel=kernel, epsilon=epsilon)

        elif algo == "ANN reg":
            hidden = st.sidebar.text_input("Hidden layers", value="64,32")
            activation = st.sidebar.selectbox("activation", ["relu", "tanh", "logistic"])
            alpha = st.sidebar.number_input("alpha", min_value=0.0, value=0.0001, step=0.0001, format="%f")
            max_iter = st.sidebar.number_input("max_iter", min_value=100, value=300, step=50)
            model = MLPRegressor(
                hidden_layer_sizes=parse_hidden_layers(hidden),
                activation=activation, alpha=alpha, max_iter=max_iter, random_state=num
            )

    # ─── Classification ───
    elif sup_type == "Classification":
        algo = st.sidebar.selectbox(
            "model",
            ["Logistic Regression", "Sgd classifier", "Decision Tree classifier",
             "Random Forest classifier", "Gradient boosting classifier",
             "KNN class", "SVC", "Naive Bayes", "Lda", "ANN class", "voting classifier"]
        )
        st.sidebar.success(f"you choose supervised → classification → {algo}")
        st.sidebar.header("2) parameters")

        if algo == "Logistic Regression":
            C = st.sidebar.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f", key=" c r ")
            max_iter = st.sidebar.number_input("max_iter", min_value=100, value=500, step=50)
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=num)

        elif algo == "Sgd classifier":
            max_iter = st.sidebar.number_input("max_iter", min_value=100, value=1000, step=100)
            loss = st.sidebar.selectbox("loss", ["hinge", "log_loss", "modified_huber", "squared_hinge"])
            alpha = st.sidebar.number_input("alpha", min_value=0.0001, value=0.04, step=0.01, format="%f")
            model = SGDClassifier(loss=loss, max_iter=max_iter, alpha=alpha, random_state=num)

        elif algo == "Decision Tree classifier":
            max_depth = st.sidebar.number_input("max_depth", min_value=1, value=5, step=1)
            criterion = st.sidebar.selectbox("criterion", ["gini", "entropy", "log_loss"])
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
            weights = st.sidebar.selectbox("weights", ["uniform", "distance"])
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

        elif algo == "SVC":
            C = st.sidebar.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f")
            kernel = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
            probability = st.sidebar.checkbox("Enable probability", value=True)
            model = SVC(C=C, kernel=kernel, probability=probability, random_state=num)

        elif algo == "Naive Bayes":
            var_smoothing = st.sidebar.number_input("var_smoothing", min_value=1e-10, value=1e-9, format="%e")
            choosing_technique = st.selectbox("choosing_technique", ["GaussianNB", "MultinomialNB", "BernoulliNB"])
            if choosing_technique == "GaussianNB":
                model = GaussianNB(var_smoothing=var_smoothing)
            if choosing_technique == "MultinomialNB":
                model = MultinomialNB(var_smoothing=var_smoothing)
            if choosing_technique == "BernoulliNB":
                model = BernoulliNB(var_smoothing=var_smoothing)

        elif algo == "Lda":
            solver = st.sidebar.selectbox("solver", ["svd", "lsqr", "eigen"])
            model = LinearDiscriminantAnalysis(solver=solver)

        elif algo == "ANN class":
            hidden = st.sidebar.text_input("Hidden layers", value="100")
            activation = st.sidebar.selectbox("activation", ["relu", "tanh", "logistic"])
            alpha = st.sidebar.number_input("alpha", min_value=0.0, value=0.0001, step=0.0001, format="%f")
            max_iter = st.sidebar.number_input("max_iter", min_value=50, value=300, step=10)
            model = MLPClassifier(
                hidden_layer_sizes=parse_hidden_layers(hidden),
                activation=activation, alpha=alpha, max_iter=max_iter, random_state=num)

        elif algo == "voting classifier":
            st.sidebar.subheader("⚪ LogisticRegression")
            C = st.sidebar.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f")
            max_iter = st.sidebar.number_input("max_iter", min_value=100, value=500, step=50)
            lr = LogisticRegression(C=C, max_iter=max_iter, random_state=num)

            st.sidebar.subheader("⚪ Sgd classifier")
            max_iter = st.sidebar.number_input("max_iter", min_value=100, value=1000, step=100)
            loss = st.sidebar.selectbox("loss", ["hinge", "log_loss", "modified_huber", "squared_hinge"])
            alpha = st.sidebar.number_input("alpha", min_value=0.0001, value=0.04, step=0.01, format="%f")
            sgd = SGDClassifier(loss=loss, max_iter=max_iter, alpha=alpha, random_state=num)

            st.sidebar.subheader("⚪ Decision Tree classifier")
            max_depth = st.sidebar.number_input("max_depth", min_value=1, value=5, step=1, key="tree_max_depth")
            criterion = st.sidebar.selectbox("criterion", ["gini", "entropy", "log_loss"])
            dt = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=num)

            st.sidebar.subheader("⚪ Random Forest classifier")
            n_estimators = st.sidebar.number_input("n_estimators", min_value=10, value=100, step=10, key="rf_estimators")
            max_depth = st.sidebar.number_input("max_depth", min_value=1, value=5, step=1, key="rf_max_depth")
            rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=num)

            st.sidebar.subheader("⚪ Gradient boosting classifier")
            n_estimators = st.sidebar.number_input("n_estimators", min_value=10, value=100, step=10, key="gb_estimators")
            learning_rate = st.sidebar.number_input("learning_rate", min_value=0.001, value=0.1, step=0.01, format="%.3f")
            max_depth = st.sidebar.number_input("max_depth", min_value=1, value=3, step=1, key="gb_max_depth")
            gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                             max_depth=max_depth, random_state=num)

            st.sidebar.subheader("⚪ KNN class")
            n_neighbors = st.sidebar.slider("n_neighbors", 1, 100, 5)
            weights = st.sidebar.selectbox("weights", ["uniform", "distance"])
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

            st.sidebar.subheader("⚪ SVC")
            C = st.sidebar.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f", key="c svc")
            kernel = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
            probability = st.sidebar.checkbox("Enable probability", value=True)
            svc = SVC(C=C, kernel=kernel, probability=probability, random_state=num)

            st.sidebar.subheader("⚪ Lda")
            solver = st.sidebar.selectbox("solver", ["svd", "lsqr", "eigen"])
            lda = LinearDiscriminantAnalysis(solver=solver)

            voting = st.sidebar.selectbox("voting", ["hard", "soft"])
            model = VotingClassifier(estimators=[
                ('lr', lr),
                ('sgd', sgd),
                ('dt', dt),
                ('rfc', rfc),
                ('gbc', gbc),
                ('knn', knn),
                ('lda', lda)
            ], voting=voting)

    return model


def build_unsupervised_model(x):
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

    return model, transformed
