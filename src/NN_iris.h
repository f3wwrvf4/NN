#pragma once

#include <vector>
#include <NN_math.h>
#include <NN_fw.h>

namespace NN
{
class Iris
{
public:
  static const int DataSize = 4;
  static const int LabelSize = 3;

  typedef ContentData<DataSize, LabelSize> Content;

  static void LoadTrainData(Content& data);
  static void LoadTestData(Content& data);

private:
  static void LoadData(Content& data, bool isTrain);
};

}