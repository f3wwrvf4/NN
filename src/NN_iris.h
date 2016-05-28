#pragma once

#include <vector>
#include <NN_math.h>

namespace NN
{
class Iris
{
public :
  static const int DataSize = 4;
  static const int LabelSize = 3;

  Iris();
  ~Iris();

  void LoadData();

  int GetTrainDataCount() const;
  const Matrix& GetTrainInputData(int idx, int count, NN::Matrix* mat) const;
  const Matrix& GetTrainOutputData(int idx, int count, NN::Matrix* mat) const;

  int GetTestDataCount() const;
  const Matrix& GetTestInputData(int idx, int count, NN::Matrix* mat) const;
  const Matrix& GetTestOutputData(int idx, int count, NN::Matrix* mat) const;

  void shuffle();

private:

  struct Data
  {
    Data(float in[3], float ou[3])
    {
      for (int i = 0; i < 3; ++i) {
        input[i] = in[i];
        out[i] = ou[i];
      }
    }
    float input[3];
    float out[3];
  };

  std::vector<Data> trainData;
  std::vector<Data> testData;
};

}