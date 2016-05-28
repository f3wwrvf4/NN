#pragma once

#include <vector>

namespace NN
{
class MNIST
{
  std::vector<float>* data;
  std::vector<float>* label;
  int img_count;
  int img_row;
  int img_col;

public:
  MNIST();
  ~MNIST();

  void loadTrainData();
  void loadTestData() {/* todo */}

  int count() const
  {
    return img_count;
  }

  int imageRow() const
  {
    return img_row;
  }
  int imageCol() const
  {
    return img_col;
  }

  const std::vector<float>& imageData(int idx) const
  {
    return data[idx];
  }
  const std::vector<float>& labelData(int idx) const
  {
    return label[idx];
  }

private:
  int reverseByte(int num);
};

}