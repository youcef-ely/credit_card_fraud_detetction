from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt, seaborn as sns
import pandas as pd


def roc_auc_curve(test_target, y_prob):
  fpr, tpr, _ = roc_curve(test_target, y_prob)
  plt.figure()
  plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (area = %0.2f)' % roc_auc_score(test_target, y_prob))
  plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc = "lower right")
  plt.show()

def heatmap(confusionMatrix, params: dict = {}):
  fig = plt.figure()
  sns.heatmap(confusionMatrix, annot = True, fmt = '.0f', cmap = 'Blues', **params)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.show()



def feature_importance(feature_importances, columns, figsize, params = {}):
  feature_importance_dict = {name: importance for name, importance in zip(columns, feature_importances)}
  feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns = ['name', 'importance']) \
                                  .sort_values(by = 'importance', ascending = False)

  feature_importance_df.plot(kind = 'barh', x = 'name', y = 'importance', figsize = figsize, **params)
  plt.show()