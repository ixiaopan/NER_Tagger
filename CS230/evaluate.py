import numpy as np

def evaluate(model, data_loader, params, loss_fn, metrics_func):
  model.eval()

  eval_log = []

  for inputs, labels in data_loader:
    out = model(inputs)
    loss = loss_fn(out, labels)

    log_per_batch = { name: metrics_func[name](out, labels) for name in metrics_func.keys() }
    log_per_batch['loss'] = loss.item()
    eval_log.append(log_per_batch)
  
  return { m: round(np.mean([x[m] for x in eval_log ]), 4) for m in eval_log[0].keys() }

