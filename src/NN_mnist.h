#pragma once

#include <vector>
#include <NN_math.h>
#include <NN_fw.h>

namespace NN
{
class MNIST
{
public:
  static const int DataSize = 26*26;
  static const int LabelSize = 10;
  typedef ContentData<DataSize, LabelSize> Content;

  static void LoadTrainData(Content& data);
  static void LoadTestData(Content& data);

private:
  static void loadData(const char* data_path, const char* label_path, Content&);
  static int reverseByte(int num)
  {
    char bt[4];
    char* pt = (char*)(&num);

    bt[3] = *pt++;
    bt[2] = *pt++;
    bt[1] = *pt++;
    bt[0] = *pt;

    return *(int*)(bt);
  }


#if 0
  void LoadTrainData();
  void LoadTestData();

  int GetTrainDataCount() const { return (int)trainData.size(); }
  const Matrix& GetTrainInputData(int idx, int count, NN::Matrix* mat) const;
  const Matrix& GetTrainOutputData(int idx, int count, NN::Matrix* mat) const;

  int GetTestDataCount() const { return (int)testData.size(); }
  const Matrix& GetTestInputData(int idx, int count, NN::Matrix* mat) const;
  const Matrix& GetTestOutputData(int idx, int count, NN::Matrix* mat) const;

  void shuffle();

private:

  struct Data
  {
    Data(float in[DataSize], float ou[LabelSize])
    {
      if(in)
        for (int i = 0; i < DataSize; ++i) {
          input[i] = in[i];
        }
      if(ou)
        for (int i = 0; i < LabelSize; ++i) {
          out[i] = ou[i];
        }
    }
    float input[DataSize];
    float out[LabelSize];
  };


  std::vector<Data> trainData;
  std::vector<Data> testData;
  int img_count;
  int img_row;
  int img_col;

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

#endif
};

}