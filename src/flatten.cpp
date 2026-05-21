#include "../inc/flatten.hpp"

Tensor Flatten::Forward(const Tensor &input) {
  batch_cache = input.getBatchSize();
  channels_cache = input.getChannels();
  rows_cache = input.getRows();
  columns_cache = input.getColumns();
  size_t total_features = channels_cache * rows_cache * columns_cache;
  Tensor output(batch_cache, total_features, 1, 1);
  output.SetData(input.GetData());
  return output;
}

Tensor Flatten::Backward(const Tensor &grad_out) {
  Tensor output(batch_cache, channels_cache, rows_cache, columns_cache);
  output.SetData(grad_out.GetData());
  return output;
}
