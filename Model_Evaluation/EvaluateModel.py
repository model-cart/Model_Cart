from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, roc_curve, auc, precision_score, recall_score, f1_score
import custom_evaluation_methods


def eval_performance(algorithm, X_test, y_test):
    '''evaluate the model's performance using metrics such as accuracy, confusion matrix, classification report, kappa value '''
    # Predict the classes of the test set
    y_pred = algorithm.predict(X_test)
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Calculate the confusion matrix of the model
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    #
    # Calculate the classification report of the model
    cr = classification_report(y_test, y_pred)
    print("Classification Report:\n", cr)

    # Calculate the kappa value of the model
    kappa = cohen_kappa_score(y_test, y_pred)
    print("Kappa Value:", kappa)

    # Calculate the precision score of the model
    precision = precision_score(y_test, y_pred, average='weighted')

    # Calculate the recall_score  of the model
    recall = recall_score(y_test, y_pred, average='weighted')

    # Calculate the f1_score of the model
    f1 = f1_score(y_test, y_pred, average='weighted')


    # Calculate the AUC of the model
    # y_prob = algorithm.predict_proba(X_test)
    # fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    # auc_score = auc(fpr, tpr)
    # print("Area Under Curve (AUC):", auc_score)
    return accuracy, kappa, cr, cm, precision, recall, f1