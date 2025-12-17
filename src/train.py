import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def split_umap_data(df):
    unique_id = df['clusterid'].unique()
    if len(unique_id) != 1:
        raise ValueError("Dataframe contains more clusterid. Please, narrow down to one.")
    
    label = unique_id[0]  # e.g. 'as'

    # Split
    X = df[['UMAP_1', 'UMAP_2']]
    y = df['Cluster']

    # Stratify only when every class contain at least 3 elements
    y_counts = np.bincount(y + 1)  # can be -1, bugs can be avoided by +1 shift
    if np.min(y_counts[y_counts > 0]) < 3:
        stratify_param = None
    else:
        stratify_param = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return {
        f'X_{label}_train': X_train,
        f'X_{label}_test': X_test,
        f'y_{label}_train': y_train,
        f'y_{label}_test': y_test,
        f'X_{label}_train_scaled': X_train_scaled,
        f'X_{label}_test_scaled': X_test_scaled,
        f'scaler_{label}': scaler
    }

def knn_with_outlier_filtering(data_dict, label, nu=0.03, k_range=(1, 30, 2), cv=5):
    X_train_scaled = data_dict[f'X_{label}_train_scaled']
    y_train        = data_dict[f'y_{label}_train']
    X_test_scaled  = data_dict[f'X_{label}_test_scaled']
    y_test         = data_dict[f'y_{label}_test']

    # One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    ocsvm.fit(X_train_scaled)

    # GridSearchCV
    param_grid = {
        'n_neighbors': np.arange(*k_range),
        'metric': ['euclidean', 'manhattan']
    }
    scorer = make_scorer(f1_score, average='micro')
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        verbose=0,
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_k = grid_search.best_params_['n_neighbors']
    best_metric = grid_search.best_params_['metric']
    print(f"Optimal K: {best_k}, Metric: {best_metric}")

    # Final model
    final_knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
    final_knn.fit(X_train_scaled, y_train)

    # Prediction
    y_pred_final = []
    for point in X_test_scaled:
        if ocsvm.predict([point])[0] == -1:
            y_pred_final.append(-1)
        else:
            y_pred_final.append(final_knn.predict([point])[0])

    # Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final))
    print('Accuracy:', round(accuracy_score(y_test, y_pred_final), 4))
    print('Macro F1:', round(f1_score(y_test, y_pred_final, average='macro'), 4))
    return final_knn, y_pred_final

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor='black',
        annot_kws={"size": 14}
    )

    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("Ground Truth", fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def random_forest_with_outlier_filtering(data_dict, label, nu=0.03, n_estimators=100, max_depth=None, random_state=42):
    
    X_train_scaled = data_dict[f'X_{label}_train_scaled']
    y_train        = data_dict[f'y_{label}_train']
    X_test_scaled  = data_dict[f'X_{label}_test_scaled']
    y_test         = data_dict[f'y_{label}_test']

    # Outlier detection: One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    ocsvm.fit(X_train_scaled)
    inlier_mask = ocsvm.predict(X_train_scaled) == 1

    # Inliers
    X_train_filtered = X_train_scaled[inlier_mask]
    y_train_filtered = y_train[inlier_mask]

    # Random Forest modell train
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    rf.fit(X_train_filtered, y_train_filtered)

    # Prediction
    y_pred_final = []
    for point in X_test_scaled:
        if ocsvm.predict([point])[0] == -1:
            y_pred_final.append(-1)
        else:
            y_pred_final.append(rf.predict([point])[0])

    # Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final))
    print('Accuracy:', round(accuracy_score(y_test, y_pred_final), 4))
    print('Macro F1:', round(f1_score(y_test, y_pred_final, average='macro'), 4))

    return rf, y_pred_final

def xgboost_with_outlier_filtering(data_dict, label, nu=0.05, n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42):
    
    X_train = data_dict[f"X_{label}_train_scaled"]
    X_test  = data_dict[f"X_{label}_test_scaled"]
    y_train = data_dict[f"y_{label}_train"]
    y_test  = data_dict[f"y_{label}_test"]

     # Outlier detection: One-Class SVM
    ocsvm = OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
    ocsvm.fit(X_train)

    is_inlier_train = ocsvm.predict(X_train) == 1
    is_inlier_test = ocsvm.predict(X_test) == 1

    # Inliers
    X_train_clean = X_train[is_inlier_train]
    X_test_clean = X_test[is_inlier_test]

    y_train_clean = np.array(y_train)[is_inlier_train]
    y_test_clean  = np.array(y_test)[is_inlier_test]

    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_clean)
    y_test_enc  = le.transform(y_test_clean)
    num_classes = len(le.classes_)

    # XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=random_state
    )

    # Train
    xgb_model.fit(X_train_clean, y_train_enc)

    # Prediction and back transformation
    y_pred_enc = xgb_model.predict(X_test_clean)
    y_pred = le.inverse_transform(y_pred_enc)

    # Evaluation
    print("\nXGBoost + OneClassSVM results (only Inliers):")
    print(f"Accuracy: {accuracy_score(y_test_clean, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_test_clean, y_pred, average='macro'):.4f}")
    print(classification_report(y_test_clean, y_pred))

    # Return cleaned lists
    return xgb_model, y_pred, y_test_clean

def xgboost_plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    labels = sorted(set(y_true))

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.show()
