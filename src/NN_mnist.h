#pragma once

#include <vector>
#include <NN_math.h>

namespace NN
{
class MNIST
{
public:
  static const int DataSize = 26*26;
  static const int LabelSize = 10;

  MNIST();
  ~MNIST();

  void LoadData();

  int GetTrainDataCount() const;
  const Matrix& GetTrainInputData(int idx, int count, NN::Matrix* mat) const;
  const Matrix& GetTrainOutputData(int idx, int count, NN::Matrix* mat) const;

  int GetTestDataCount() const;
  const Matrix& GetTestInputData(int idx, int count, NN::Matrix* mat) const;
  const Matrix& GetTestOutputData(int idx, int count, NN::Matrix* mat) const;

  void shuffle();


private:
  std::vector<float>* data;
  std::vector<float>* label;
  int img_count;
  int img_row;
  int img_col;


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
  int reverseByte(int num);
};

}