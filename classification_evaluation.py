def evaluation(model=model_0, Features=X_val, Label=y_val, DL=False):

  if DL:
    model_prob = model.predict(X_val)
    preds = tf.squeeze(tf.round(model_prob))
  else:
    preds = model.predict(X_val)

  dic = {
      "Accuracy": accuracy_score(y_val, preds),
      "precision": precision_score(y_val, preds),
      "f1_score": f1_score(y_val, preds),
      "recall_score": recall_score(y_val, preds)
  }

  return dic